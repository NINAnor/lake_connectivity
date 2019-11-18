#!/usr/bin/python

# Call script like this: ./connectivity.py --wrid=12345 --cores=10
# where 12345 represents the water region ID to process
# and 10 represents the number of cores that can be used for parallelised
# functions (mainly in the network analysis)


################################################################################
# Load required libraries
import argparse
import itertools
import os
import sys
import subprocess
from cStringIO import StringIO
from itertools import combinations
from itertools import islice
from itertools import chain
import time
import re
import shutil
import math
import multiprocessing as mp

import logging

import numpy as np
import psycopg2
from psycopg2 import extras
from igraph import *
import pandas as pd

# imported from db_creditals.py which has to be chmod 600 and added to gitignore
# It defines the variables host user db password
import db_creditals


# Add parser function for boolean arguments
# https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# Get command line arguments
parser = argparse.ArgumentParser(description='Compute connectivity between lakes along river network.')
# parser.add_argument("wrid", type=int, default=None,
parser.add_argument("--host", default='localhost', metavar='host',
                    help="Host adress / IP for the PostgreSQL database")
parser.add_argument("--port", type=int, default=5432, metavar='port',
                    help="Name of the PostgreSQL database")
parser.add_argument("--db", default=None, metavar='db',
                    help="Name of the PostgreSQL database")
parser.add_argument("--user", default=None, metavar='user',
                    help="Name of the PostgreSQL database user")
parser.add_argument("--wrid", type=int, default=None, metavar='wrid',
                    help="water region ID from Hydrography.waterregions_dem_10m_nosefi")
parser.add_argument("--cores", type=int, nargs='?', default=1, metavar='cores',
                    help="Number of cores to use at maximum")
parser.add_argument("--schema_prefix", nargs='?', default="temporary",
                    help="Full GRASS map name of the terrain model to use")
parser.add_argument("--dem", nargs='?', default="dem_10m_nosefi_float@g_Elevation_Fenoscandia",
                    help="Full GRASS map name of the terrain model to use")
parser.add_argument("--memory", type=int, nargs='?', default=10000, metavar='memory',
                    help="Amount of memory in MB to be used for stream extraction")
parser.add_argument("--terrain_stream", type=str2bool, nargs='?', default=False, metavar='terrain_stream',
                    help="Generate stream network from terrain model")
parser.add_argument("--focal_lakes", nargs='?', default=None, metavar='focal_lakes',
                    help="Comma separated list of lakes to process. E.g.: 1,6,8,9,17")
parser.add_argument("--snap", type=int, default=1, metavar='snap',
                    help="Snap distance in whole meters at which lakes are connected to river networks")
parser.add_argument("--grassdb", nargs='?', default="/data/scratch",
                    help="Path to GRASS database (where GRASS data is stored)")
parser.add_argument("--location", nargs='?', default="stefan.blumentrath",
                    help="Name of the GRASS location to work in (where DEM is located)")
parser.add_argument("--mapset_prefix", nargs='?', default="p_INVAFISH",
                    help="Prefix of the GRASS mapset to work in (will be created if it does not exist). All GRASS data end up here")
parser.add_argument("--sections", nargs='?', default="gis,network_compilation,lake_combinations,lake_summary", metavar='sections',
                    help="Comma separated list of sections to process. Allowed values: gis,network_compilation,lake_combinations,lake_summary.")
parser.add_argument("--graph_dir", nargs='?', default="/data/R/Avd15GIS/Prosjekter/INVAFISH/networks",
                    help="Directory where Python igraph network objects should be stored")
parser.add_argument("--stream_table", nargs='?', default="Hydrography.Streams_Norway_2017",
                    help="Fully qualified (with schema) name of PostGIS table with stream network (schema.table)")
parser.add_argument("--catchment_table", nargs='?', default="Hydrography.waterregions_dem_10m_nosefi",
                    help="Fully qualified (with schema) name of PostGIS table with catchment areas (schema.table)")
parser.add_argument("--lake_table", nargs='?', default="nofa.lake",
                    help="Fully qualified (with schema) name of PostGIS table with geometries (schema.table)")
parser.add_argument("--verbose", type=str2bool, nargs='?', default=False, metavar='verbose',
                    help="Give extra output from processes (PostGIS and GRASS)")

args = parser.parse_args()


def formatPGnames(name):
    name = name.strip('"')
    new_name = name.split('.') if len(name.split('.')) == 2 else ['public', name]
    return new_name

# Get PostGIS Input
stream_table = formatPGnames(args.stream_table)
catchment_table = formatPGnames(args.catchment_table)
lake_table = formatPGnames(args.lake_table)

# Get water region ID to process
wrid = args.wrid

# Get distance for snapping lakes to river networks
# Most relevant for terrain derived networks
snap_lakes = args.snap

# Get terrain model to use for analysis
if len(args.dem.split('@')) == 2:
    dem = args.dem.split('@')[0]
    req_mapsest = args.dem.split('@')[1]
else:
    dem = args.dem.split('@')[0]
    req_mapsest = False

# Get number of cores to use for processing
cores = args.cores

# Get amount of memory in MB to be used for stream extraction
memory = args.memory

# Get name of schema where results are supposed to be stored
schema_prefix = args.schema_prefix

# Derive stream network from terrain model (or not)
terrain_stream = args.terrain_stream

if args.focal_lakes:
    try:
        focal_lakes = map(str, map(int, args.focal_lakes.split(',')))
    except:
        print('Error: Could convert focal lake ids to integer...')
        exit(0)
else:
    focal_lakes = False

# Get GRASS related settings
# Path to GRASS database (where GRASS data is stored)
grassdb = args.grassdb

# GRASS location
location = args.location

# GRASS mapset (working directory where GRASS maps end up)
mapset_prefix = args.mapset_prefix
mapset = "{}_{}".format(mapset_prefix, wrid)

# Check if recreation of GIS data should be skipped
sections = args.sections.split(',')

# Set information message level
verbose = args.verbose
quiet = False if verbose else True

graph_dir = args.graph_dir

# Give message about cores used
print("Using up to {} cores".format(cores))

start_time_total = time.time()


# Connect to PostGIS
# Set connection parameters
pg_host = args.host
pg_port = args.port
pg_db = args.db
pg_user = args.user

pgpass = os.path.join(os.environ['HOME'],'.pgpass')
if not os.path.exists(pgpass):
    print("ERROR: Required .pgpass file missing in $HOME")
    exit(0)

pgp_match = '{}:{}:{}:{}:'.format(pg_host, pg_port, pg_db, pg_user)
pgp_line = None
try:
    with open(pgpass, 'r') as pgp:
        for line in pgp:
            if pgp_match in line:
                pgp_line = line
                break
except:
    print("ERROR: Could not read file .pgpass file...")
    exit(0)

if not pgp_line:
    print("Combination of host, database, port and user not found in .pgpass file...")
    exit(0)

pg_password = pgp_line.strip(os.linesep).split(':')[-1]


"""
pg_host = db_creditals.host
pg_db = db_creditals.db
pg_user = db_creditals.user
pg_password = db_creditals.password
"""

tablespace = 'connectivity'

con_string = "host='" + pg_host + "' dbname='" + pg_db + "' user='" + pg_user + "' password='" + pg_password + "'"
print(con_string)

try:
    con = psycopg2.connect(con_string)
except:
    print("Unable to connect to the database")
    exit(0)

con.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)

# Connect to GRASS

# GRASS binary
grass7bin = subprocess.check_output(['which', 'grass']).split('\n')[0]

# Full path to GRASS mapset
grasspath = os.path.join(grassdb, location, mapset)

if 'gis' in sections or 'network_compilation' in sections:
    # Check if GRASS DB exists
    if not os.path.exists(grassdb):
        print('Error: Directory "{}" for GRASS GIS database not found.'.format(grassdb))
        exit(1)

    # Check if location exists
    if not os.path.exists(os.path.join(grassdb, location)):
        print('Error: Directory "{}" for GRASS GIS location not found.'.format(location))
        exit(1)

    if 'gis' not in sections and not os.path.exists(grasspath):
        print('Warning: GIS data preparation is to be skiped, but GRASS mapset {} cannot be found for reading the network data.'.format(grasspath))
        exit(1)
    else:
        # Create mapset if it does not exist
        if not os.path.exists(grasspath):
            try:
                os.system('{0} -text -c -e {1}'.format(grass7bin, grasspath))
            except:
                print('Error: Could not start GRASS session. Please check your GRASS related input.')
                exit(1)

    # query GRASS 7 itself for its GISBASE
    startcmd = [grass7bin, '--config', 'path']

    p = subprocess.Popen(startcmd, shell=False,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    if p.returncode != 0:
        print >>sys.stderr, "ERROR: Cannot find GRASS GIS 7 start script (%s)" % startcmd
        sys.exit(-1)

    # Path to GRASS libraries (where GRASS is installed)
    gisbase = out.strip('\n\r')

    # Set GISBASE environment variable
    os.environ['GISBASE'] = gisbase

    # the following not needed with trunk
    os.environ['PATH'] += os.pathsep + os.path.join(gisbase, 'extrabin')

    # add path to GRASS addons
    home = os.path.expanduser("~")
    os.environ['PATH'] += os.pathsep + os.path.join(home,
                                                    '.grass7',
                                                    'addons',
                                                    'scripts')

    # define GRASS-Python environment
    gpydir = os.path.join(gisbase, "etc", "python")
    sys.path.append(gpydir)

    # Check if location exists
    if not os.path.exists(graph_dir):
        print('Error: Directory "{}" for storing network objects not found.'.format(graph_dir))
        exit(1)

    # DATA
    # Set GISDBASE environment variable
    os.environ['GISDBASE'] = grassdb

    # import GRASS Python bindings (see also pygrass)
    import grass.script as grass
    import grass.script.setup as gsetup

    # launch session
    gsetup.init(gisbase, grassdb, location, mapset)

    genv = grass.gisenv()

    print('Current GRASS GIS 7 environment:')
    print("""GRASS database: {0}\nLOCATION: {1}\nMAPSET: {2}""".format(genv['GISDBASE'], genv['LOCATION_NAME'], genv['MAPSET']))

    # grass.run_command('g.gisenv', set='GRASS_VECTOR_TMPDIR_MAPSET=0')
    # grass.run_command('g.gisenv', set='TMPDIR=/tmp')

    grass.run_command('db.login', host=pg_host, driver='pg', database=pg_db,
                      user=pg_user, password=pg_password, overwrite=True)

    # Set compression options for GRASS maps
    os.environ['GRASS_COMPRESSOR'] = 'LZ4'
    os.environ['GRASS_COMPRESS_NULLS'] = '1'

    # Initalise garbage list
    tmp_vmaps = []
    tmp_rmaps = []

# Define which river network to use (names of hypertable objects are not
# by schema name separated)
stream_master_table = 'Streams_{}'.format(schema_prefix)
streaminput = "Streams_WR{}".format(wrid)
graph_name = 'WR{0}_network.pickle'.format(wrid)

if terrain_stream:
    stream_master_table += '_{}'.format(dem)
    streaminput += "_{}".format(dem)
    graph_name = 'WR{0}_{1}_network.pickle'.format(wrid, dem)

# Define and initialise hard coded variables
pg_schema = schema_prefix
pg_result_schema = schema_prefix
pg_result_table_master = "{}_lake_connectivity".format(schema_prefix)
pg_result_table_summary_master = "{}_lake_connectivity_summary".format(schema_prefix)
pg_tmp_schema = '{}_{}'.format(schema_prefix, wrid)

nd_str = '-9999\t-9999\t-9999\t-9999\t-9999\t-9999\t-9999\t-9999\t-9999\t-9999\t-9999'

# Create schemas if necessary
print('Creating schemas if not exist')

# Check if required extensions exist in DB
cur = con.cursor()
cur.execute("""SELECT extname FROM pg_extension""")
pg_extensions = cur.fetchall()
if not ('postgis',) in pg_extensions:
    print('ERROR: Could not find extension "postgis" in database {}'.format(pg_db))
    if os.path.exists(grasspath):
        shutil.rmtree(grasspath)
    exit(1)

if not ('timescaledb',) in pg_extensions:
    print('ERROR: Could not find extension "timescaledb" in database {}'.format(pg_db))
    if os.path.exists(grasspath):
        shutil.rmtree(grasspath)
    exit(1)


tries = 0
schema_exists = None
while tries < 5 and not schema_exists:
    tries = tries + 1
    cur = con.cursor()
    cur.execute("""SELECT schema_name FROM information_schema.schemata WHERE schema_name = '{}';""".format(pg_result_schema))
    schema_exists = cur.fetchone()
    if not schema_exists:
        try:
            create_tmp_schema = """CREATE SCHEMA IF NOT EXISTS "{0}";
                GRANT USAGE ON SCHEMA "{0}" TO app_reader;
                GRANT USAGE ON SCHEMA "{0}" TO nofa_reader;
                ALTER DEFAULT PRIVILEGES IN SCHEMA "{0}" GRANT SELECT ON TABLES TO app_reader;
                ALTER DEFAULT PRIVILEGES IN SCHEMA "{0}" GRANT SELECT ON TABLES TO nofa_reader;
                ALTER DEFAULT PRIVILEGES IN SCHEMA "{0}" GRANT SELECT, USAGE ON SEQUENCES TO app_reader;
                ALTER DEFAULT PRIVILEGES IN SCHEMA "{0}" GRANT SELECT, USAGE ON SEQUENCES TO nofa_reader;
                """.format(pg_result_schema)
            cur.execute(create_tmp_schema)
        except:
            pass
if not schema_exists:
    print('Error: Could not create schema {0}. Aborting after 5 tries'.format(pg_result_schema))
    exit(1)

# Define functions
def abort_script(connection, error_message):
    """Abort script and clean remove produced data
    This function does not take any parameters and does not return any results
    All possibly created data for the water region will be removed
    :returns: None
    :Example:
    >>> abort_script(con, "Give reason why script execution is aborted.")
    """

    cur = connection.cursor()
    cur.execute("""DROP SCHEMA IF EXISTS "{0}" CASCADE;""".format(pg_tmp_schema))

    print("ERROR: {}".format(error_message))
    if os.path.exists(grasspath):
        shutil.rmtree(grasspath)
    exit(1)


def per2ideg(x):
    """Convert slope in percentage to degrees * 100 rounded to integer
    :param x: slope in percent (scaled from 0 to 1)
    :returns: slope in degree * 100
    :type x: int, float
    :rtype: int
    :Example:
    >>> per2ideg(100)
    45
    """

    if x is None:
        ideg = None
    else:
        ideg = int(round(math.degrees(math.atan(x)) * 100))
    return ideg


def w_avg(x, y):
    """Compute average of x wheighted by y, ignoring None.
    :param x: a list of numerics
    :param y: a list of numerics
    :returns: average of x wheighted by y
    :type x: list, int, float
    :type y: list, int, float
    :rtype: float
    :Example:
    >>> w_avg([1, 3, 4, 1, 3], [3, 1, 1, 10, 1])
    1.4375
    .. note::
        Lists for x and y are assumed to be of equal length
    """
    if float(sum(pd.Series(y)[pd.Series(x).isnull() == False].dropna())) == 0:
        wavg = 0
    else:
        wavg = float(sum((pd.Series(x) * pd.Series(y)).dropna())) / float(sum(pd.Series(y)[pd.Series(x).isnull() == False].dropna()))
    return wavg


def cleanup_script(connection, error_message):
    """Cleanup procedure at exit after successful execution
    :param connection: a PostgreSQL connection
    :param error_message: message to be printed at exit
    :returns: None
    :type connection: psycopg2 connection object
    :type error_message: string
    :returns: None
    :Example:
    >>> cleanup_script(con, "Give reason why script execution is finished.")
    """

    cur = connection.cursor()
    cur.execute("""DROP TABLE IF EXISTS "{0}"."Lakes_WR{1}";
        DROP TABLE IF EXISTS "{0}"."Streams_WR{1}";""".format(pg_tmp_schema,
                                                              wrid))
    print("ERROR: {}".format(error_message))
    if len(tmp_vmaps) > 0:
        grass.run_command('g.remove', type='vector', name=tmp_vmaps,
                          flags='f', quiet=True)
    if len(tmp_rmaps) > 0:
        grass.run_command('g.remove', type='raster', name=tmp_rmaps,
                          flags='f', quiet=True)
    if os.path.exists(grasspath):
        shutil.rmtree(grasspath)
    exit(1)


def remove_tmp(connection, wrid):
    """Clean up temnporary data after successful script execution
    Only temporary data for the water region will be removed
    :param connection: a psycopg2 connection object
    :param wrid: Water region ID of the current run
    :returns: None
    :Example:
    >>> remove_tmp(con, wrid)
    """

    print("Script finished successfully for Water region {}. Cleaning up temporary data.".format(wrid))
    cur = connection.cursor()
    cur.execute("""DROP TABLE IF EXISTS "{0}"."Lakes_WR{1}";
                DROP TABLE IF EXISTS "{0}"."Streams_WR{1}";""".format(pg_tmp_schema, wrid))
    if len(tmp_vmaps) > 0:
        grass.run_command('g.remove', type='vector', name=tmp_vmaps,
                          flags='f', quiet=True)
    if len(tmp_rmaps) > 0:
        grass.run_command('g.remove', type='raster', name=tmp_rmaps,
                          flags='f', quiet=True)


def sendToPG(results):
    """Send results for individual lake connections to specific PostgreSQL table
    :param results: A list of numbers that characterize the connection.
    :returns: None
    :Example:
    >>> sendToPG(1, c(1:36))
    ..Notes::
        Length of list "results" and position of the values have to match exactly with the
        columns in the respective PG table
    """

    cur = connection.cursor()
    cur.execute(insert_statement, results)


def getRowEstimate(connection, schema, table_name):
    """Get a fast, approximate row count estimate for a specific PostgreSQL table
    Please note that quoting of identifiers is not necessary.
    :param connection: a psycopg2 database connection object.
    :param schema: PostgreSQL schema in which the table is found.
    :param table_name: Text string; name of a PostgreSQL table.
    :returns: re Estimate of row count, or -1 if table is not found if table is not found
    :Example:
    >>> getRowEstimate(con, '\"MySchema\".\"MyTable\"')
    51
    """

    cur = connection.cursor()
    cur.execute("""SELECT reltuples::bigint AS estimate FROM pg_class WHERE oid = to_regclass('"{0}"."{1}"');""".format(schema, table_name))
    re = int(cur.fetchone()[0])
    return re


def countSmall(connection, schema, table_name):
    """Get exact row count for a smaller (<10 rows) PostgreSQL table
    Note that quoting of identifiers is not necessary.
    :param connection: A PostgreSQL database connection object.
    :param schema: PostgreSQL schema in which the table is found.
    :param table_name: Text string; name of a PostgreSQL table.
    :returns: Exact count for tables with less than 10 rows, or -1 if table is not found.
    :Examples
    >>> countSmall(con, 'MySchema', 'MyTable')
    7
    """

    cur = connection.cursor()
    cur.execute("""SELECT count(*) FROM (SELECT 1 FROM "{0}"."{1}" LIMIT 10) t;""".format(schema, table_name))
    rc = int(cur.fetchone()[0])
    return rc


def pg2GRASS(schema, table_name, snap=0.001, where=None):
    """Import data from PostGIS into GRASS (topological format)
    Note that quoting of identifiers is not necessary.
    Input name will be input name.
    :param schema PostgreSQL schema in which the table is found.
    :param table_name Text string; name of a PostgreSQL table.
    :Example:
    >>> pg2GRASS('MySchema', 'MyTable')
    True
    """

    # "overwrite" gets rid of tables if they are already there so they can be
    # updated, "verbose" brings in extra detail
    grass.run_command("v.in.ogr", flags=["o"], overwrite=True, verbose=verbose,
                      quiet=quiet, output=table_name, snap=snap,
                      input="PG:dbname={} host={} user={}".format(pg_db,
                                                                  pg_host,
                                                                  pg_user),
                      layer='{}.{}'.format(schema, table_name),
                      where=where)


def existsPG(connection, schema, table_name):
    """Check if table exists in given schema in PostgreSQL
    Note that quoting of identifiers is not necessary.
    :param connection: PostgreSQL schema in which the table is supposed to be found.
    :param schema: PostgreSQL schema in which the table is supposed to be found.
    :param table_name: Text string; name of a PostgreSQL table to look for.
    :returns: Logical value if table exists (True) or not (False).
    :Example:
    >>> existsPG(con, 'MySchema', 'MyTable')
    True
    """

    cur = connection.cursor()
    cur.execute("""SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE  table_schema = '{0}' AND table_name = '{1}');""".format(schema, table_name))
    e = cur.fetchone()[0]
    return e


def existsGRASS(map, type):
    """Check if GRASS map exists
    :param map: Map name (without @mapset).
    :param type: Type of the map to check (either 'vector' or 'raster').
    :returns: Logical value if map exists (True) or not (False).
    :Example:
    >>> existsGRASS('dem_10m_nosefi_float','raster')
    True
    """

    if type == 'raster':
        mtype = 'cellhd'
    elif type == 'vector':
        mtype = 'vector'

    try:
        res = grass.parse_command("g.findfile", element=mtype, file=map)
        if res['fullname']:
            e = True
    except:
        e = False
    return e


def get_type(x):
    """
    :Example:
    >>> get_type(1.5)
    """

    try:
        if float(x).is_integer():
            try:
                y = int(float(x))
            except:
                y = long(float(x))
        else:
            y = float(x)
    except:
        y = unicode(x)
    return y


def lake_connection(lake):
    """Compute connectivity for combination of lakes
    Intended for parallel execution
    :param lake: name (id) of the lake to compute connectivity from
    :returns: None (results are sendt PosgreSQL)
    :Example:
    >>> lake_connection('12345')
    """

    # n = 0
    idx = lakes.index(lake) + 1
    rel_lakes = lakes[idx:]
    print('Processing lake: {}'.format(lake))
    output = StringIO()
    paths = cg.get_shortest_paths(lake, rel_lakes, mode="all", output="epath",
                                  weights="dir_weight")

    # Loop over paths if necessary / appropriate
    if len(paths) > 0:
        # Loop over accessible lakes
        for i in xrange(len(paths)):
            # if n == 1:
            #     output.seek(0)
            #     print(output.getvalue())
            #     output.seek(0)
            # n = n + 1

            # Get relevant subgraph
            sg = cg.subgraph_edges(cg.es[paths[i]])
            # current wrid
            output.write('{}\t'.format(wrid))
            # from_lake
            output.write(lake + '\t')
            # to_lake
            output.write(rel_lakes[i] + '\t')
            # cluster
            output.write('{}\t'.format(c))
            # lakes_along
            output.write(','.join(sg.vs.select(area_ha_gt=0)['name']) + '\t')
            # confluences_along
            output.write(','.join(sg.vs.select(typ=3)['name']) + '\t')
            # total_stream_length
            output.write('{}\t'.format(int(round(sum(sg.es['length_m'])))))
            # total_slope_max_max
            output.write('{}\t'.format(per2ideg(max(pd.Series(sg.es['slope_maximum']).dropna()))))

            for d in ['upstream', 'downstream']:
                sgd = sg.subgraph_edges(sg.es.select(direction=d))
                # Compute upstream/downstream part of the connection
                if sgd.ecount() > 0:
                    s_length = sgd.es['length_m']
                    # {direction}_length
                    output.write('{}\t'.format(int(round(sum(s_length)))))
                    # {0}_altitude_min
                    output.write('{}\t'.format(int(round(min(sgd.es['altitude_minimum'])))))
                    # {0}_altitude_mean
                    output.write('{}\t'.format(int(round(w_avg(sgd.es['altitude_average'], s_length)))))
                    # {0}_altitude_max
                    output.write('{}'.format(int(round(max(sgd.es['altitude_maximum'])))))
                    # {0}_slope_min
                    # output.write('{}\t'.format(per2ideg(min(sgd.es['slope_minimum']))))
                    # {0}_slope_perc_10
                    # output.write('{}\t'.format(per2ideg(w_avg(sgd.es['slope_minimum'], s_length))))
                    # {0}_slope_first_quart
                    # output.write('{}\t'.format(per2ideg(w_avg(sgd.es['slope_first_quartile'], s_length))))
                    # {0}_slope_mean
                    output.write('\t{}\t'.format(per2ideg(w_avg(sgd.es['slope_average'], s_length))))
                    # {0}_slope_third_quart
                    output.write('{}\t'.format(per2ideg(w_avg(sgd.es['slope_third_quartile'], s_length))))
                    # {0}_slope_perc_90
                    output.write('{}\t'.format(per2ideg(w_avg(sgd.es['slope_percentile_90'], s_length))))
                    # {0}_slope_perc_90_max
                    output.write('{}\t'.format(per2ideg(max(sgd.es['slope_percentile_90']))))
                    # {0}_slope_max
                    output.write('{}\t'.format(per2ideg(w_avg(sgd.es['slope_maximum'], s_length))))
                    # {0}_slope_max_max
                    output.write('{}\t'.format(per2ideg(max(sgd.es['slope_maximum']))))
                    # {0}_slope_stddev
                    output.write('{}'.format(per2ideg(w_avg(sgd.es['slope_stddev'], s_length))))
                    # {0}_slope_variance
                    # output.write('{}'.format(per2ideg(w_avg(sgd.es['slope_variance'], s_length))))
                else:
                    # Write NULL for every coluumn of the given direction
                    output.write(nd_str)

                if d == 'upstream':
                    output.write('\t')
                else:
                    output.write('\n')

    output.seek(0)
    # Send result to PostGIS
    try:
        con = psycopg2.connect(con_string)
    except:
        print("Error: Unable to connect to the database")

    con.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)

    cur = con.cursor()

    cur.copy_from(output, '"{0}"."{1}"'.format(pg_result_schema,
                                                   pg_result_table_master))

# .........................................................
# Prepare data in PostGIS  ------
# .........................................................
# if __name__ == '__main__':
pool = mp.Pool(processes=cores)
try:
    con = psycopg2.connect(con_string)
except:
    print("Unable to connect to the database")
    exit(1)

# Check if user has privileges to create schemata in PG

pg_priv_check = """SELECT has_database_privilege('{}', '{}', 'CREATE');""".format(pg_user, pg_db)
cur = con.cursor()
cur.execute(pg_priv_check)
pg_priv_check = cur.fetchone()
if pg_priv_check == 'False':
    print('User {1} lacks privileges to create schema in database {0}\n \
    Privileges can be granted as follows:\n \
    GRANT CREATE ON DATABASE {0} TO "{}";'.format(pg_db, pg_user))
    exit(1)

con.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)

for table in [stream_table, lake_table, catchment_table]:
    if not existsPG(con, table[0], table[1]):
        print("Cannot find table {}.".format())
        exit(1)

if 'gis' in sections:
    try:
        con = psycopg2.connect(con_string)
    except:
        print("Unable to connect to the database")
        exit(1)

    con.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)

    print('Starting GIS data preparation')

    start_time_gis = time.time()

    try:
        create_tmp_schema = """CREATE SCHEMA IF NOT EXISTS "{0}";
            GRANT USAGE ON SCHEMA "{0}" TO app_reader;
            GRANT USAGE ON SCHEMA "{0}" TO nofa_reader;
            ALTER DEFAULT PRIVILEGES IN SCHEMA "{0}" GRANT SELECT ON TABLES TO app_reader;
            ALTER DEFAULT PRIVILEGES IN SCHEMA "{0}" GRANT SELECT ON TABLES TO nofa_reader;
            ALTER DEFAULT PRIVILEGES IN SCHEMA "{0}" GRANT SELECT, USAGE ON SEQUENCES TO app_reader;
            ALTER DEFAULT PRIVILEGES IN SCHEMA "{0}" GRANT SELECT, USAGE ON SEQUENCES TO nofa_reader;
            """.format(pg_tmp_schema)
        cur.execute(create_tmp_schema)
    except:
        pass

    if not schema_exists:
        print('Error: Could not create schema {0}. Aborting after 5 tries'.format(pg_tmp_schema))
        exit(1)

    # if recreate_gis or
    #    existsGRASS("Streams_WR", wrid, "_network", sep=''), 'vector')==False or
    #    existsGRASS("Lakes_WR{}_bounds".format(wrid), 'vector')==False or
    #    existsGRASS("Lakes_WR", wrid, "_erings", sep=''), 'vector')==False or
    #    existsGRASS("Lakes_WR", wrid, sep=''), 'vector')==False):

    # Create master table for stream network
    hypercolumns_edges = """gid, cat,"""
    create_stream_master = """CREATE TABLE "{0}"."{1}_network_edges"
    (
        gid integer,
        cat integer,\n""".format(pg_schema, stream_master_table)

    for m in ['altitude', 'slope']:
        for s in ['minimum', 'maximum', 'average', 'stddev', 'variance', 'first_quartile', 'median', 'third_quartile', 'percentile_90']:
            create_stream_master += """{0}_{1} double precision,\n""".format(m, s)
            hypercolumns_edges += """{0}_{1}, """.format(m, s)

    create_stream_master += """length_m double precision,"""
    hypercolumns_edges += """length_m, """

    for s in ['minimum', 'maximum', 'average', 'stddev', 'variance', 'first_quartile', 'median', 'third_quartile', 'percentile_90']:
        create_stream_master += """{0}_{1}_deg double precision,\n""".format('slope', s)
        hypercolumns_edges += """{0}_{1}_deg, """.format('slope', s)

    create_stream_master += """geom geometry(MultiLineString,25833),\n
        wrid integer);"""
    hypercolumns_edges += """geom, wrid"""

    # Create hypertable
    # one chunk per wrid
    create_hypertable = """SET search_path = {0},timescaledb,public;
    SELECT create_hypertable('"{1}_network_edges"', 'wrid', chunk_time_interval => 1);""".format(pg_schema, stream_master_table)


    # Define privelidges
    grant = """GRANT SELECT ON TABLE "{0}"."{1}_network_edges" TO nofa_reader;""".format(pg_schema, stream_master_table)

    index = """CREATE INDEX IF NOT EXISTS "{1}_network_edges_geom_idx" ON "{0}"."{1}_network_edges" USING gist (geom);""".format(pg_schema, stream_master_table)

    # print(create_stream_master)
    cur = con.cursor()

    if not existsPG(con, pg_schema, "{}_network_edges".format(stream_master_table)):
        cur.execute(create_stream_master)
        cur.execute(create_hypertable)
        cur.execute(index)
        cur.execute(grant)


    hypercolumns_vertices = """wrid, gid, lakeid, area_ha, geom"""

    if not existsPG(con, pg_schema, "{}_network_vertices".format(stream_master_table)):

        # Create master table for stream network
        create_vertices_master = """CREATE TABLE "{0}"."{1}_network_vertices"
        (wrid integer NOT NULL,
        gid integer NOT NULL,
        lakeid integer,
        area_ha double precision,
        geom geometry(Point,25833),
        CONSTRAINT "{1}_network_vertices_pkey" PRIMARY KEY (wrid, gid)
        )""".format(pg_schema, stream_master_table)

        # Create hypertable
        # one chunk per wrid
        create_hypertable = """SET search_path = {0},timescaledb,public;
        SELECT create_hypertable('"{1}_network_vertices"', 'wrid', chunk_time_interval => 1);""".format(pg_schema, stream_master_table)


        # Define privelidges
        grant = """GRANT SELECT ON TABLE "{0}"."{1}_network_vertices" TO nofa_reader;""".format(pg_schema, stream_master_table)

        index = """CREATE INDEX IF NOT EXISTS "{1}_network_vertices_geom_idx" ON "{0}"."{1}_network_vertices" USING gist (geom);
        -- CREATE INDEX "{1}_network_vertices_wrid_idx" ON "{0}"."{1}_network_vertices" USING brin (wrid);
        CREATE INDEX IF NOT EXISTS "{1}_network_vertices_gid_idx" ON "{0}"."{1}_network_vertices" USING brin (gid);""".format(pg_schema, stream_master_table)

        # print(create_stream_master)
        cur = con.cursor()
        cur.execute(create_vertices_master)
        cur.execute(create_hypertable)
        cur.execute(index)
        cur.execute(grant)

    # Add required GRASS mapsets to search_path
    if req_mapsest:
        if req_mapsest != 'PERMANENT':
            grass.run_command("g.mapsets", mapset=req_mapsest, operation='add')

    # Check if input terrain model exists
    if not existsGRASS(dem, 'raster'):
        print("ERROR: Could not find terrain model: {}".format(dem))
        exit(1)

    # Initialise connection
    # con = poolCheckout(pool)
    # con = dbConnect(pg_drv,dbname=pg_db,user=pg_user, password=pg_password,host=pg_host)

    if not terrain_stream:
        # Overlay river network by major watersheds / catchments
        # (for limiting connectivity analysis to within catchment connectivity)
        # remove  "WHERE vassomr = '123'" in order to run on the entire dataset
        # this joins stream IDs with Water Region IDs, only needs to be used if tables need to be updated
        cur = con.cursor()
        cur.execute("""DROP TABLE IF EXISTS "{0}"."Streams_WR{1}";""".format(pg_tmp_schema, wrid))
        cur.execute("""CREATE TABLE "{0}"."Streams_WR{1}" AS SELECT a.*, max(b.gid) AS "waterRegionID", array_agg(b.gid) AS "waterRegionIDs" FROM
                              "{2}"."{3}" AS a,
                              (SELECT gid, geom FROM "{4}"."{5}" WHERE gid in ({1})) AS b
                              WHERE ST_Intersects(a.geom,b.geom)
                              GROUP BY a.ogc_fid, b.gid
                              ORDER BY a.ogc_fid, b.gid;""".format(pg_tmp_schema, wrid, stream_table[0], stream_table[1], catchment_table[0], catchment_table[1]))

        # Create spatial index and analyse table in order to speed up subsequent queries
        # (VACUUM ANALYZE needed in order to make PostGIS use the spatial index)
        # Only works if you run series of commands directly above this
        # Basically the index helps speed things up by letting the connection know if a stream is close to our watershed and worth checking further to see if there is intersection
        cur = con.cursor()
        cur.execute("""CREATE INDEX "Streams_WR{1}_gist" ON "{0}"."Streams_WR{1}" USING gist (geom);""".format(pg_tmp_schema, wrid))
        cur.execute("""ALTER TABLE "{0}"."Streams_WR{1}" CLUSTER ON "Streams_WR{1}_gist";""".format(pg_tmp_schema, wrid))
        cur.execute("""VACUUM FULL ANALYZE "{0}"."Streams_WR{1}";""".format(pg_tmp_schema, wrid))

        ####################################################################################################################################################################################################################
        # Skip if no streams selectd
        # Get row estimate of stream table
        n = getRowEstimate(con, pg_tmp_schema, streaminput)
        if n < 10:
            # Get exact row count of stream table
            n = countSmall(con, pg_tmp_schema, streaminput)
            if n == 0:
                abort_script(con, "Warning: No streams in waterregion. Cleaning up and exiting.")

    ####################################################################################################################################################################################################################

    # Same as above but for lakes
    # Overlay lakes by major watersheds / catchments
    # (for limiting connectivity analysis to within catchment connectivity)
    # remove  "WHERE vassomr = '123'" in order to run on the entire dataset
    cur.execute("""DROP TABLE IF EXISTS "{0}"."Lakes_WR{1}";
                   CREATE TABLE "{0}"."Lakes_WR{1}" AS SELECT a.id AS lakeid, a.geom, max(b.gid) AS "waterRegionID", array_agg(b.gid) AS "waterRegionIDs" FROM
                          "{2}"."{3}" AS a,
                          (SELECT gid, geom FROM "{4}"."{5}" WHERE gid in ({1})) AS b
                          WHERE ST_Intersects(a.geom, b.geom)
                          GROUP BY a.id, a.geom
                          ORDER BY a.id;""".format(pg_tmp_schema, wrid, lake_table[0], lake_table[1], catchment_table[0], catchment_table[1]))

    # Create spatial index and analyse table in order to speed up subsequent queries (VACUUM ANALYZE needed in order to make PostGIS use the spatial index)
    cur.execute("""CREATE UNIQUE INDEX "Lakes_WR{1}_idx" ON "{0}"."Lakes_WR{1}" USING btree (lakeid);""".format(pg_tmp_schema, wrid))
    cur.execute("""ALTER TABLE "{0}"."Lakes_WR{1}" CLUSTER ON "Lakes_WR{1}_idx";""".format(pg_tmp_schema, wrid))
    cur.execute("""CREATE INDEX "Lakes_WR{1}_gist" ON "{0}"."Lakes_WR{1}" USING gist (geom);""".format(pg_tmp_schema, wrid))
    cur.execute("""ALTER TABLE "{0}"."Lakes_WR{1}" CLUSTER ON "Lakes_WR{1}_gist";""".format(pg_tmp_schema, wrid))
    cur.execute("""VACUUM FULL ANALYZE "{0}"."Lakes_WR{1}";""".format(pg_tmp_schema, wrid))

    ####################################################################################################################################################################################################################
    # Skip if less than two lakes selectd (not be necessary if this check is included in the selection of wrids to process)
    # Get row estimate of lakes table
    n = getRowEstimate(con, pg_tmp_schema, "Lakes_WR{}".format(wrid))
    if n < 10:
        # Get exact row count of lakes table
        n = countSmall(con, pg_tmp_schema, "Lakes_WR{}".format(wrid))
        if n < 2:
            abort_script(con, "Warning: Less than two lakes in waterregion. Cleaning up and exiting.")

    ####################################################################################################################################################################################################################

    # Remove interior rings (islands) from lake polygons, because NVEs rivernetwork is crossing islands (which is only a technical artifact)
    cur.execute("""DROP TABLE IF EXISTS "{0}"."Lakes_WR{1}_erings";""".format(pg_tmp_schema, wrid))
    cur.execute("""CREATE TABLE "{0}"."Lakes_WR{1}_erings" AS SELECT lakeid, geom FROM
                           (SELECT lakeid, ST_Union(ST_BuildArea(ST_ExteriorRing(geom))) AS geom FROM
                           (SELECT lakeid, (ST_Dump(geom)).geom AS geom FROM "{0}"."Lakes_WR{1}") AS a GROUP BY lakeid) AS b;""".format(pg_tmp_schema, wrid))

    # Create spatial index and analyse table in order to speed up subsequent queries (VACUUM ANALYZE needed in order to make PostGIS use the spatial index)
    cur.execute("""CREATE UNIQUE INDEX "Lakes_WR{1}_erings_idx" ON "{0}"."Lakes_WR{1}_erings" USING btree (lakeid);""".format(pg_tmp_schema, wrid))
    cur.execute("""ALTER TABLE "{0}"."Lakes_WR{1}_erings" CLUSTER ON "Lakes_WR{1}_erings_idx";""".format(pg_tmp_schema, wrid))
    cur.execute("""CREATE INDEX "Lakes_WR{1}_erings_gist" ON "{0}"."Lakes_WR{1}_erings" USING gist (geom);""".format(pg_tmp_schema, wrid))
    cur.execute("""ALTER TABLE "{0}"."Lakes_WR{1}_erings" CLUSTER ON "Lakes_WR{1}_erings_gist";""".format(pg_tmp_schema, wrid))
    cur.execute("""VACUUM FULL ANALYZE "{0}"."Lakes_WR{1}_erings";""".format(pg_tmp_schema, wrid))

    # Build network in GRASS GIS  ------

    # Derive stream network from terrain model if requested
    if terrain_stream:
        # Import catchment polygon
        pg2GRASS(catchment_table[0], catchment_table[1], snap=0.001,
                 where='gid = {}'.format(wrid))
        """grass.run_command("v.in.ogr", flags="o", overwrite=True,
                          verbose=verbose, quiet=quiet,
                          input='PG:dbname={} host={} user={}'.format(pg_db, pg_host, pg_user),
                          layer='{}.{}'.format(catchment_table[0],
                                               catchment_table[1]),
                          output="Catchment{}".format(wrid),
                          where='gid = {}'.format(wrid), snap=0.001)"""

        tmp_vmaps.append(catchment_table[1])

        # Create a 30m buffer around catchment
        grass.run_command("v.buffer", overwrite=True, verbose=verbose,
                          quiet=quiet, distance=30,
                          input=catchment_table[1],
                          output="{}_buffer".format(catchment_table[1]))

        tmp_vmaps.append("{}_buffer".format(catchment_table[1]))

        # Set computational region to catchmentbuffer
        grass.run_command("g.region", flags="p", align=dem,
                          vector="{}_buffer".format(catchment_table[1]))

        # User buffer as mask
        grass.run_command("r.mask", overwrite=True, verbose=verbose,
                          quiet=quiet,
                          vector="{}_buffer".format(catchment_table[1]))

        tmp_rmaps.append("MASK")

        # Define parameters for stream extraction
        min_seg_length_m = 150
        threshold = 2000
        # Get resolution of the DEM
        reginfo = grass.parse_command("g.region", flags="g")
        nsres = float(reginfo['nsres'])
        ewres = nsres = float(reginfo['ewres'])
        min_seg_length_pixel = int(round(min_seg_length_m / mean([ewres, nsres]), 0))

        # Perform stream extraction
        grass.run_command("r.stream.extract", overwrite=True, verbose=verbose,
                          quiet=quiet, elevation=dem, threshold=threshold,
                          stream_length=min_seg_length_pixel,
                          memory=memory, stream_vector=streaminput,
                          stream_raster=streaminput, direction='drain_dir')

        tmp_rmaps.append(streaminput)
        tmp_vmaps.append(streaminput)

        # Check if lines in river network
        stream_input_num = grass.parse_command('v.info', flags='t', map=streaminput)
        if int(stream_input_num['lines']) == 0:
            abort_script(con, "Warning: No streams derived for waterregion. Cleaning up and exiting.")

        if not grass.shutil_which('r.stream.slope'):
            grass.run_command('g.extension', extension='r.stream.slope',
                              operation='add')

        grass.run_command("r.stream.slope", overwrite=True, verbose=verbose,
                          quiet=quiet, elevation=dem, direction='drain_dir',
                          gradient='{}_gradient'.format(streaminput))

        tmp_rmaps.append('{}_gradient'.format(streaminput))

        grass.run_command("g.remove", overwrite=True, quiet=True, type='raster',
                          flags='f', name='drain_dir')

    else:
        # Import stream network
        pg2GRASS(pg_tmp_schema, streaminput, snap=1.0)

        # Snap vertices of lines
        grass.run_command('v.edit', map=streaminput, id='0-9999999',
                          tool='snap', threshold='-1,1', type='line')

    # Import relevant tables from PostGIS to GRASS
    # Import exterior rings of lakes
    pg2GRASS(pg_tmp_schema, "Lakes_WR{}_erings".format(wrid))
    tmp_vmaps.append("Lakes_WR{}_erings".format(wrid))


    # Import lakes with islands
    pg2GRASS(pg_tmp_schema, "Lakes_WR{}".format(wrid))
    tmp_vmaps.append("Lakes_WR{}".format(wrid))

    # Merge adjacent lines as far as possible (in order to have only single line strings between lakes / fork points (lake - lake, lake - fork point, fork point - fork point))
    # If you have a stream that has been broken into two sections by a rogue node somewhere, merges them into 1
    grass.run_command("v.build.polylines", overwrite=True, verbose=verbose,
                      quiet=quiet, input=streaminput, cats="first",
                      output="Streams_WR{}_polyline".format(wrid))
    tmp_vmaps.append("Streams_WR{}_polyline".format(wrid))

    # Remove lines within lakes from river network
    grass.run_command("v.overlay", flags="t", overwrite=True, verbose=verbose,
                      quiet=quiet, ainput="Streams_WR{}_polyline".format(wrid),
                      atype="line", binput="Lakes_WR{}_erings".format(wrid),
                      output="Streams_WR{}_notLake".format(wrid),
                      operator="not", olayer="0,0,0")
    tmp_vmaps.append("Streams_WR{}_notLake".format(wrid))

    # Add categories
    grass.run_command("v.category", input="Streams_WR{}_notLake".format(wrid),
                      layer='1', option='add', overwrite=True,
                      output="Streams_WR{}_notLake_cat".format(wrid))
    tmp_vmaps.append("Streams_WR{}_notLake_cat".format(wrid))

    # Create a network dataset from river line strings by adding nodes at start, end, and fork points
    grass.run_command("v.net", flags="c", overwrite=True, verbose=verbose,
                      quiet=quiet, arc_layer='1', node_layer='2',
                      input="Streams_WR{}_notLake_cat".format(wrid),
                      output="{}_network".format(streaminput),
                      operation="nodes")

    # create a database table for edges
    grass.run_command("v.db.addtable", verbose=verbose, quiet=quiet,
                      map="{}_network".format(streaminput), layer=1,
                      table="{}_network".format(streaminput))

    # Assign ID of closest lake to nodes in network
    # create a database table for vertices
    grass.run_command("v.db.addtable", verbose=verbose, quiet=quiet,
                      map="{}_network".format(streaminput), layer=2,
                      table="Streams_WR{}_network_vertices".format(wrid),
                      columns="lakeid integer")

    # Check if still lines in river network
    stream_input_num = grass.parse_command('v.info', flags='t', map="{}_network".format(streaminput))
    if int(stream_input_num['lines']) == 0:
        abort_script(con, "Warning: No streams derived for waterregion. Cleaning up and exiting.")

    # Load ID of lakes within 1m distance to vertices into database of vertices (column = lake_id)
    grass.run_command("v.distance", overwrite=True, verbose=verbose,
                      quiet=quiet, from_="{}_network".format(streaminput),
                      from_layer="2", column="lakeid",
                      to="Lakes_WR{}_erings".format(wrid), to_column="lakeid",
                      upload="to_attr", dmax=snap_lakes)

    # Mark lake vertices
    # create a column for vertices
    # grass.run_command("v.db.addcolumn", flags=c("verbose"),
    #           map="{}_network".format(streaminput), layer='2',
    #           columns="is_lake smallint")
    # grass.run_command("v.db.update", map="{}_network".format(streaminput), layer="2", column="is_lake", value='1', where="lakeid IS NOT NULL",redirect=True,legacyExec=True)
    # grass.run_command("v.db.update", map="{}_network".format(streaminput), layer="2", column="is_lake", value='0', where="lakeid IS NULL",redirect=True,legacyExec=True)

    # Fill lake_id column for vertices with no lake within 1m distance with individual values from "cat" column
    max_lake_id = max(map(int,
                          grass.read_command("v.db.select", flags="c",
                                             quiet=True,
                                             map="Lakes_WR{}_erings".format(wrid),
                                             layer="1",
                                             columns="lakeid").split('\n')[:-1]))

    # grass.run_command("v.db.select", flags=c("c", "quiet"),map="{}_network".format(streaminput), layer="2", columns="lakeid,is_lake", where='lakeid < 5000000', redirect=True, legacyExec=True)
    grass.run_command("v.db.update", verbose=verbose, quiet=quiet,
                      map="{}_network".format(streaminput), layer="2",
                      column="lakeid", where="lakeid IS NULL",
                      query_column="cat+{}".format(max_lake_id))

    # Calculate stream/lake data based on spatial data -----

    # Calculate lake size
    grass.run_command("v.db.addcolumn", verbose=verbose, quiet=quiet,
                      map="Lakes_WR{}".format(wrid),
                      columns="area_ha double precision")
    grass.run_command("v.to.db", quiet=True,
                      map="Lakes_WR{}".format(wrid),
                      option="area", columns="area_ha", units="hectares")

    # Join lake attributes to network
    grass.run_command("v.db.join", quiet=True,
                      map="{}_network".format(streaminput), layer=2,
                      column='lakeid',
                      other_table="Lakes_WR{}".format(wrid),
                      other_column='lakeid',
                      subset_columns="area_ha")

    # Charaterize river line strings
    # Convert line strings from vector to raster using ID ("cat")
    if verbose:
        grass.run_command("g.region", flags="p",
                          vector=["Lakes_WR{}_erings".format(wrid),
                                  "{}_network".format(streaminput)],
                          n="n+100", s="s-100", e="e+100", w="w-100",
                          align=dem)
    else:
        grass.run_command("g.region",
                          vector=["Lakes_WR{}_erings".format(wrid),
                                  "{}_network".format(streaminput)],
                          n="n+100", s="s-100", e="e+100", w="w-100",
                          align=dem)
    if terrain_stream:
        stream_raster = streaminput
    else:
        grass.run_command("v.to.rast", overwrite=True, verbose=verbose,
                          quiet=quiet, input="{}_network".format(streaminput),
                          type="line", use="cat", flags='d',
                          output="{}_network_cat".format(streaminput))
        stream_raster = "{}_network_cat".format(streaminput)

        # Calculate slope in river network
        # Get resolution of the DEM
        reginfo = grass.parse_command("g.region", flags="ug")
        nsres = float(reginfo['nsres'])
        ewres = nsres = float(reginfo['ewres'])
        # First we look at the pixels directly left,right,above,below central pixel, see if the river is present there,
        # take the altitude change, divide by 10(metres)
        stream_raster
        grass.run_command("r.mapcalc", overwrite=True, verbose=verbose,
                          quiet=quiet,
                          expression="""Streams_WR{0}_network_local_slope_direct=if(\
        isnull({4}),null(),\
        if(nmin(if(isnull({4}[1,0]),9999,{1}[1,0]),\
        if(isnull({4}[-1,0]),9999,{1}[-1,0]),\
        if(isnull({4}[0,1]),9999,{1}[0,1]),\
        if(isnull({4}[0,-1]),9999,{1}[0,-1]))>{1},-9999,({1}-nmin(\
        if(isnull({4}[1,0]),9999,{1}[1,0]),\
        if(isnull({4}[-1,0]),9999,{1}[-1,0]),\
        if(isnull({4}[0,1]),9999,{1}[0,1]),\
        if(isnull({4}[0,-1]),9999,{1}[0,-1])))/float({2}+{3})/2.0))
        Streams_WR{0}_network_local_slope_diagonal=if(\
        isnull({4}),null(),\
        if(nmin(if(isnull({4}[1,1]),9999,{1}[1,1]),\
        if(isnull({4}[-1,1]),9999,{1}[-1,1]),\
        if(isnull({4}[1,-1]),9999,{1}[1,-1]),\
        if(isnull({4}[-1,-1]),9999,{1}[-1,-1]))>{1},-9999,({1}-nmin(\
        if(isnull({4}[1,1]),9999,{1}[1,1]),\
        if(isnull({4}[-1,1]),9999,{1}[-1,1]),\
        if(isnull({4}[1,-1]),9999,{1}[1,-1]),\
        if(isnull({4}[-1,-1]),9999,{1}[-1,-1])))/sqrt({2}+{3})))""".format(wrid, dem, nsres, ewres, stream_raster))
        # now take the maximum of all of them
        grass.run_command("r.mapcalc", overwrite=True, verbose=verbose,
                          quiet=quiet,
                          expression="""{1}_network_local_slope=max(\
        if(Streams_WR{0}_network_local_slope_direct==-9999,0,Streams_WR{0}_network_local_slope_direct),\
        if(Streams_WR{0}_network_local_slope_diagonal==-9999,0,Streams_WR{0}_network_local_slope_diagonal))""".format(wrid, streaminput))

        grass.run_command("g.remove", type="raster", flags="f",
                          pattern="Streams_WR{}_network_local_slope_di*".format(wrid))

    # Calculate univariate statistics on terrain model for every line string ("cat" as key column)
    # First converts line into series of rasters, then gives univar stats for that series
    grass.run_command("v.rast.stats", flags="cd", overwrite=True,
                      verbose=verbose, quiet=quiet,
                      map="{}_network".format(streaminput), raster=dem,
                      method=["minimum", "maximum", "average", "stddev",
                              "variance", "first_quartile", "median",
                              "third_quartile", "percentile"],
                      percentile=90, column_prefix="altitude")

    if terrain_stream:
        # Calculate univariate statistics on slope for every line string ("cat" as key column)
        grass.run_command("v.rast.stats", flags="cd", overwrite=True,
                          verbose=verbose, quiet=quiet,
                          map="{}_network".format(streaminput),
                          raster="{}_gradient".format(streaminput),
                          method=["minimum", "maximum", "average", "stddev",
                                  "variance", "first_quartile", "median",
                                  "third_quartile", "percentile"],
                          percentile=90, column_prefix="slope")
    else:
        # Calculate univariate statistics on slope for every line string ("cat" as key column)
        grass.run_command("v.rast.stats", flags="cd", overwrite=True,
                          verbose=verbose, quiet=quiet,
                          map="{}_network".format(streaminput),
                          raster="{}_network_local_slope".format(streaminput),
                          method=["minimum", "maximum", "average", "stddev",
                                  "variance", "first_quartile", "median",
                                  "third_quartile", "percentile"],
                          percentile=90, column_prefix="slope")

    # Add edges (connections between adjacent lakes)
    # add categories for boundaries of the input vector map, in layer 2:
    grass.run_command("v.category", overwrite=True, verbose=verbose,
                      quiet=quiet, input="Lakes_WR{}".format(wrid),
                      output="Lakes_WR{}_bounds".format(wrid), layer='2',
                      type='boundary', option='add')

    # tmp_vmaps.append("Lakes_WR{}_bounds".format(wrid))

    # add a table with columns named "left" and "right" to layer 2 of the input vector map:
    grass.run_command("v.db.addtable", map="Lakes_WR{}_bounds".format(wrid),
                      layer=2, columns="left integer,right integer,left_lake integer,right_lake integer")
    # Upload categories of left and right areas:
    grass.run_command("v.to.db", map="Lakes_WR{}_bounds".format(wrid),
                      option='sides', columns=["left", "right"], layer='2',
                      verbose=verbose, quiet=quiet)
    # Upload lake ids
    grass.run_command("db.select", sql="""UPDATE "Lakes_WR{0}_bounds_2" SET left_lake = (SELECT lakeid FROM Lakes_WR{0} WHERE left = "Lakes_WR{0}".cat);""".format(wrid))
    grass.run_command("db.select", sql="""UPDATE "Lakes_WR{0}_bounds_2" SET right_lake = (SELECT lakeid FROM Lakes_WR{0} WHERE right = "Lakes_WR{0}".cat);""".format(wrid))

    # Calculate lenght in meter for every line string ("cat" as key column)
    grass.run_command("v.db.addcolumn", quiet=True,
                      map="{}_network".format(streaminput),
                      columns="length_m double precision")
    grass.run_command("v.to.db", quiet=True,
                      map="{}_network".format(streaminput), type="line",
                      option="length", columns="length_m", units="meters")

    # Export resulting network to PostGIS
    grass.run_command('v.out.ogr', overwrite=True, verbose=verbose, quiet=quiet,
                      format='PostgreSQL', type='line', layer='1',
                      input="{}_network".format(streaminput),
                      output='PG:host= {} dbname={} user={}'.format(pg_host, pg_db, pg_user),
                      output_layer='{0}.{1}_network_edges'.format(pg_tmp_schema,
                                                                  streaminput),
                      flags='m', lco=['LAUNDER=NO',
                                      'GEOMETRY_NAME=geom',
                                      'FID=gid'])

    grass.run_command('v.out.ogr', overwrite=True, verbose=verbose, quiet=quiet,
                      format='PostgreSQL', type='point', layer='2',
                      input="{}_network".format(streaminput),
                      output='PG:host= {} dbname={} user={}'.format(pg_host, pg_db, pg_user),
                      output_layer='{0}.{1}_network_vertices'.format(pg_tmp_schema,
                                                                     streaminput),
                      lco=['LAUNDER=NO', 'GEOMETRY_NAME=geom', 'FID=gid'])

    # Add empty columns to match hypertable structure
    for s in ['minimum', 'maximum', 'average', 'stddev', 'variance',
              'first_quartile', 'median', 'third_quartile', 'percentile_90']:
        cur.execute("""ALTER TABLE "{0}"."{1}_network_edges" ADD COLUMN {2}_{3}_deg double precision;""".format(pg_tmp_schema, streaminput, 'slope', s))

    cur.execute("""ALTER TABLE "{0}"."{1}_network_edges" ADD COLUMN wrid integer;""".format(pg_tmp_schema, streaminput))
    cur.execute("""ALTER TABLE "{0}"."{1}_network_vertices" ADD COLUMN wrid integer;""".format(pg_tmp_schema, streaminput))

    # After export to PG
    for s in ['minimum', 'maximum', 'average', 'stddev', 'variance',
              'first_quartile', 'median', 'third_quartile', 'percentile_90']:
        # Compute degrees
        cur.execute("""UPDATE "{0}"."{1}_network_edges" SET {2}_{3}_deg = degrees(atan({2}_{3}));""".format(pg_tmp_schema, streaminput, 'slope', s))

    cur.execute("""UPDATE "{0}"."{1}_network_edges" SET {2} = {3};""".format(pg_tmp_schema, streaminput, 'wrid', wrid))
    cur.execute("""UPDATE "{0}"."{1}_network_vertices" SET {2} = {3};""".format(pg_tmp_schema, streaminput, 'wrid', wrid))

    # Link table to streams master
    cur.execute("""DELETE FROM "{0}"."{1}_network_edges" WHERE wrid = {2};""".format(pg_schema, stream_master_table, wrid))
    cur.execute("""DELETE FROM "{0}"."{1}_network_vertices" WHERE wrid = {2};""".format(pg_schema, stream_master_table, wrid))
    add_data_to_hypertable = """INSERT INTO "{0}"."{3}_network_edges" SELECT {4} FROM "{1}"."{2}_network_edges";
    INSERT INTO "{0}"."{3}_network_vertices" SELECT {5} FROM "{1}"."{2}_network_vertices";""".format(pg_schema, pg_tmp_schema, streaminput, stream_master_table, hypercolumns_edges, hypercolumns_vertices)
    cur.execute(add_data_to_hypertable)

    remove_tmp(con, wrid)
    cur.execute("""DROP SCHEMA "{0}" CASCADE;""".format(pg_tmp_schema))
    con.close()
    proc_time_gis = time.time() - start_time_gis
    print('Done with GIS data preparation. Took: {} sec.'.format(int(proc_time_gis)))
else:
    print('Skipping GIS data preparation on request.')
    if not existsGRASS("{}_network".format(streaminput), 'vector') and 'network_compilation' in sections:
        print("Could not find map: Streams_WR{}_network".format(wrid))
        exit(1)

################################################################################
if 'network_compilation' in sections:
    try:
        con = psycopg2.connect(con_string)
    except:
        logging.info("Unable to connect to the database")

    con.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)

    cur = con.cursor()

    print("Preparing network data...")
    start_time_net = time.time()
    net = pd.read_csv(StringIO(grass.read_command('v.net', input="{}_network".format(streaminput),
                                                  points="{}_network".format(streaminput), arc_layer=1, node_layer=2,
                                                  operation='report', quiet=True).rstrip('\n')), sep=' ', header=-1,
                      dtype={0: np.int32, 1: np.int32, 2: np.int32})
    net.columns = ['line_cat', 'from', 'to']

    # Vertex attributes
    lake_attr = pd.read_csv(StringIO(grass.read_command("v.db.select", quiet=True,
                                                        map="{}_network".format(streaminput), layer=2,
                                                        columns="cat,lakeid,area_ha", separator=",").rstrip('\n')),
                            dtype={'cat': np.int32, 'lakeid': np.int32, 'area_h': np.float64})

    # Collaps nodes for lakes (merge inlet(s) and outlet(s))
    lid = lake_attr[['cat', 'lakeid']]
    lid.columns = ['from', 'from_lakeid']
    net = pd.merge(net, lid, how='left', on='from')
    lid.columns = ['to', 'to_lakeid']
    net = pd.merge(net, lid, how='left', on='to')

    # Create Graph object
    g = Graph().as_directed()

    # Add vertices with names
    vertices = list(lake_attr['lakeid'])
    vertices.sort()
    vertices = set(vertices)
    g.add_vertices(map(str, list(vertices)))

    # Add vertex attributes
    header = tuple(lake_attr.columns)[1:]
    # for a in header:
    #     g.vs[a] = 0

    for i, r in lake_attr.iterrows():
        for a in header:
            g.vs.select(name=str(get_type(r['lakeid'])))[a] = get_type(r[a])

    # Add edges
    edges = []
    # Create edge list using lakeid (in order to merge all nodes in lakes)
    for i, r in net.iterrows():
        # From- and to-vertices for edges
        edges.append((str(r['from_lakeid']), str(r['to_lakeid']),))

    # Add edgelist to graph
    g.add_edges(edges)

    # Extract stream features (as edge attributes) into data frame
    stream_attr = pd.read_csv(StringIO(grass.read_command("v.db.select", quiet=True,
                                                          map="{}_network".format(streaminput), layer=1,
                                                          separator=",").rstrip('\n')), dtype={'cat': np.int32})

    # Set Nan values to 0 that can occure for short line stings (below or around resolution)
    stream_attr = stream_attr.fillna(0)

    # Add edge attributes
    header = tuple(stream_attr.columns)
    for a in header:
        # g.es[a] = 0
        g.es.set_attribute_values(a, stream_attr[a])

    g.es['direction'] = 'downstream'

    # Remove possible loops
    g.delete_edges([e.index for e in g.es.select(_is_loop=True)])

    # Add edges for adjacent lakes if required
    ed = pd.read_csv(StringIO(grass.read_command("v.db.select",
                                                 map="Lakes_WR{}_bounds".format(wrid),
                                                 columns="left_lake,right_lake",
                                                 layer=2,
                                                 where="left>0 AND right>0",
                                                 separator=",").rstrip('\n')),
                     dtype={'left_lake': np.int32, 'right_lake': np.int32})

    # Add edges for adjacent lakes (if such exist)
    if len(ed['left_lake']) > 0:
        # Create attribute template
        at_template = {}
        for a in g.es.attributes():
            at_template[a] = 0

        # Define direction in template
        at_template['direction'] = 'downstream'

        # Add possibly missing vertices to network
        new_vids = list(set(list(ed['left_lake']) + list(ed['right_lake'])))
        for v in new_vids:
            if str(v) not in g.vs['name']:
                g.add_vertices(str(v))

        # Add only new edges
        for i, r in ed.iterrows():
            from_lakeid = str(r['left_lake'] if r['left_lake'] > r['right_lake'] else r['right_lake'])
            to_lakeid = str(r['left_lake'] if r['left_lake'] < r['right_lake'] else r['right_lake'])

            # Add edges for adjacent lakes in both directions
            for direct in [(to_lakeid, from_lakeid), (from_lakeid, to_lakeid)]:
                try:
                    # Check if edge exists
                    edge_id = g.get_eid(direct[0], direct[1])
                except:
                    # Get ID for edge to add
                    edge_id = g.ecount()

                    # Add new edge
                    g.add_edge(direct[0], direct[1])

                    # Add edge attributes from template
                    g.es[edge_id].update_attributes(at_template)

    # Add weights for direction
    g.es['dir_weight'] = [0] * g.ecount()
    g.es.select(direction='upstream')['dir_weight'] = 1

    # Remove possible duplicate edges
    # g.simplify(combine_edges=max)

    # Assign cluster membership to vertices
    g.vs['cluster'] = g.as_undirected().clusters().membership

    # Compute incoming degree centrality
    # sources have incoming degree centrality of 0
    g.vs['indegree'] = g.degree(mode="in")

    # Compute outgoing degree centrality
    # outlets have outgoing degree centrality of 0
    g.vs['outdegree'] = g.degree(mode="out")

    # Classify vertices as
    #     1 = Sources (without incomming edges),
    #     2 = Connections (one incoming and one outgoing edge),
    #     3 = Fork (two or more incoming edges),
    #     4 = Outlets (without outgoing edges)
    #     0 = Unclassified
    g.vs['typ'] = 0
    # Sources (without incoming edges)
    g.vs.select(indegree=0)['typ'] = 1
    # Connections (one incoming and one outgoing edge)
    g.vs.select(indegree=1).select(outdegree=1)['typ'] = 2
    # Confluences 2 or more incoming edges
    g.vs.select(indegree_gt=1)['typ'] = 3
    # Outlets (without outgoing edges)
    g.vs.select(outdegree=0)['typ'] = 4
    # Isolated nodes (without incoming or outgoing edges)
    g.vs.select(outdegree=0).select(indegree=0)['typ'] = 5

    # Remove vertices without edges
    # g.delete_vertices([v.index for v in g.vs.select(_degree=0)])

    # Sum lake area for each cluster and assign it to vertices/lakes in network
    # Count lakes for each cluster and assign it to vertices/lakes in network
    g.vs['cluster_lake_area_ha'] = 0
    g.vs['cluster_lake_n'] = 0

    # Note: Consider parallelisation
    clusters = set(g.vs['cluster'])
    for c in clusters:
        g.vs.select(cluster=c)['cluster_lake_area_ha'] = sum(g.vs.select(cluster=c).select(area_ha_gt=0)['area_ha'])
        g.vs.select(cluster=c)['cluster_lake_n'] = len(set(g.vs.select(cluster=c).select(area_ha_gt=0)['lakeid']))

    # Remove clusters that dont contain lakes
    g.delete_vertices([v.index for v in g.vs.select(cluster_lake_n_lt=2)])

    if g.ecount() < 1:
        abort_script(con, "Warning: Network did not contain any edges. Aborting script.")
    elif max(g.vs['cluster_lake_n']) < 2:
        abort_script(con, "Warning: Largest cluster in network contained less than two lakes. Aborting script.")

    # Add reverse edges
    eids = range(g.ecount())

    # Note: Consider parallelisation
    for e in eids:
        at = g.es[e].attributes()
        at['direction'] = 'upstream'
        max_edge_id = g.ecount()
        g.add_edge(g.es[e].target, g.es[e].source)
        g.es[max_edge_id].update_attributes(at)

    con.close()

    if not os.path.exists(graph_dir):
        os.mkdir(graph_dir)

    g.write_pickle(os.path.join(graph_dir, graph_name))

    proc_time_net = time.time() - start_time_net
    print('Done with network data preparation. Took: {} sec.'.format(int(proc_time_net)))

################################################################################
if 'lake_combinations' in sections:
    if 'network_compilation' not in sections:
        g = Graph.Read_Pickle(os.path.join(graph_dir, graph_name))

    # Get unique list of clusters
    clusters = set(g.vs['cluster'])

    print("Strating connectivity analysis for lake combinations for each of the {} clusters".format(len(clusters)))

    try:
        con = psycopg2.connect(con_string)
    except:
        logging.info("Unable to connect to the database")

    con.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)

    cur = con.cursor()

    # Create destination table if necessary
    if not existsPG(con, pg_result_schema, pg_result_table_master):

        # Make sure result tables exist
        create_master = """SET search_path TO "{0}", public;
        CREATE TABLE IF NOT EXISTS "{1}"(
           wrid integer NOT NULL,
           from_lake integer NOT NULL,
           to_lake integer NOT NULL,
           cluster smallint NOT NULL,
           lakes_along text,
           confluences_along text,
           total_stream_length integer,
           total_slope_max_max smallint,
           upstream_length integer,
           upstream_altitude_min smallint,
           upstream_altitude_mean smallint,
           upstream_altitude_max smallint,
           -- upstream_slope_min smallint,
           -- upstream_slope_perc_10 smallint,
           -- upstream_slope_first_quart smallint,
           upstream_slope_mean smallint,
           upstream_slope_third_quart smallint,
           upstream_slope_perc_90 smallint,
           upstream_slope_perc_90_max smallint,
           upstream_slope_max smallint,
           upstream_slope_max_max smallint,
           upstream_slope_stddev smallint,
           -- upstream_slope_variance smallint,
           """.format(pg_result_schema, pg_result_table_master)
        create_master += """downstream_length integer,
           downstream_altitude_min smallint,
           downstream_altitude_mean smallint,
           downstream_altitude_max smallint,
           -- downstream_slope_min smallint,
           -- downstream_slope_perc_10 smallint,
           -- downstream_slope_first_quart smallint,
           downstream_slope_mean smallint,
           downstream_slope_third_quart smallint,
           downstream_slope_perc_90 smallint,
           downstream_slope_perc_90_max smallint,
           downstream_slope_max smallint,
           downstream_slope_max_max smallint,
           downstream_slope_stddev smallint --,
           -- downstream_slope_variance smallint
           """.format(pg_result_schema, pg_result_table_master)
        create_master += """) TABLESPACE {0};
        COMMENT ON COLUMN "{1}".wrid IS 'WaterRegionID';
        COMMENT ON COLUMN "{1}".from_lake IS 'WaterRegionID';
        COMMENT ON TABLE "{1}" IS
        'Lake connectivity hyper-table';""".format(tablespace,
                                                     pg_result_table_master)

        cur = con.cursor()
        cur.execute(create_master)

        cur = con.cursor()

        # Create indices on partitioned table
        cur.execute("""SET search_path TO "{0}", public;
        -- BRIN indices save a lot of disk space but have a poor performance on unordered data. So it works fine in upstream direction, but not downstreams.
        -- In addition btree allows index-only-scans (table does not have to be accessed at all) which are not possible with BRIN
        -- See: http://dev.sortable.com/brin-indexes-in-postgres-9.5/
        CREATE INDEX IF NOT EXISTS "{1}_upstream_max_max" ON "{1}" USING brin (from_lake, upstream_slope_max_max, upstream_length, to_lake);
        CREATE INDEX IF NOT EXISTS "{1}_downstream_max_max" ON "{1}" USING btree (to_lake, downstream_slope_max_max, downstream_length, from_lake);
        -- CREATE INDEX IF NOT EXISTS "{1}_from_lake" ON  "{1}" USING brin (from_lake);
        -- CREATE INDEX IF NOT EXISTS "{1}_to_lake" ON  "{1}" USING brin (to_lake);
        -- CREATE INDEX IF NOT EXISTS "{1}_upstream_length" ON  "{1}" USING brin (upstream_length);
        -- CREATE INDEX IF NOT EXISTS "{1}_downstream_length" ON  "{1}" USING brin (downstream_length);
        -- CREATE INDEX IF NOT EXISTS "{1}_upstream_slope_max_max" ON  "{1}" USING brin (upstream_slope_max_max);
        -- CREATE INDEX IF NOT EXISTS "{1}_downstream_slope_max_max" ON  "{1}" USING brin (downstream_slope_max_max);
        -- CREATE INDEX IF NOT EXISTS "{1}_upstream_slope_mean" ON  "{1}" USING brin (upstream_slope_mean);
        -- CREATE INDEX IF NOT EXISTS "{1}_downstream_slope_mean" ON  "{1}" USING brin (downstream_slope_mean);
        """.format(pg_result_schema,  pg_result_table_master))

        # Create TimescaleDB hyper table
        cur = con.cursor()
        cur.execute("""SELECT table_name FROM _timescaledb_catalog.hypertable WHERE table_name = '{}';""".format(pg_result_table_master))
        is_hyper = cur.fetchone()
        if not is_hyper:
            cur.execute("""SET search_path TO "{0}", public;
            SELECT create_hypertable('{1}', 'wrid', chunk_time_interval => 1, if_not_exists => TRUE);
            SELECT attach_tablespace('{2}', '{1}', if_not_attached => TRUE);""".format(pg_result_schema, pg_result_table_master, tablespace))

    cur.execute("""DELETE FROM "{0}"."{1}" WHERE wrid = {2};""".format(pg_schema, pg_result_table_master, wrid))


    con.close()

    for c in clusters:
        print("Analysing cluster {}...".format(c))
        start_time_con = time.time()

        # Create subgraph for the cluster to operate on
        cg = g.vs.select(cluster=c).subgraph()
        if focal_lakes:
            lakes = map(str, sorted(map(int,
                                        cg.vs.select(name_in=focal_lakes)['name'])))
        else:
            lakes = map(str, sorted(map(int,
                                        cg.vs.select(area_ha_gt=0)['name'])))

        # Check if cluster contains at least two lakes
        if len(lakes) <= 1:
            print('Warning: Less than two lakes to process in cluster {}. Skipping...'.format(c))
            continue

        # Simplyfy subgraph to only contain edges that connect lakes
        cg = cg.subgraph(set(chain.from_iterable(cg.as_undirected().get_shortest_paths(lakes[0],
                                                                                       lakes[1:],
                                                                                       mode='ALL',
                                                                                       output='vpath'))))

        # Check if no lakes got lost
        l_n = cg.vs['cluster_lake_n'][0]
        if len(lakes) != l_n:
            print("Warning: computation of lake number incorrect...")

        # Give some info messages
        print("Number of vertices in cluster {0} is: {1}".format(c, cg.vcount()))
        print("Number of edges in cluster {0} is: {1}".format(c, cg.ecount()))
        print("Number of lakes in cluster {0} is: {1}".format(c, len(lakes)))
        lcn = math.factorial(len(lakes)) // (math.factorial(2) * math.factorial(len(lakes)-2))
        print("Number of lake combinations in cluster {0} to process is: {1}".format(c, lcn))

        if lcn % cores > 0:
            chunk_size = lcn / cores + 1
        else:
            chunk_size = lcn / cores

        if lcn < cores:
            cores = lcn
            chunk_size = 1

        from multiprocessing import Pool
        if __name__ == '__main__':
            p = Pool(cores)
            p.map(lake_connection, lakes[:-1])

        proc_time_con = time.time() - start_time_con
        print('Done with connectivity analysis for cluster {0}. Took: {1} sec.'.format(c, int(proc_time_con)))

    start_time_db = time.time()

    try:
        con = psycopg2.connect(con_string)
    except:
        logging.info("Unable to connect to the database")

    con.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)

    cur = con.cursor()

    proc_time_db = time.time() - start_time_db
    print("Done with catchment {0}, creating indices on result table. Took: {1} sec.".format(wrid, proc_time_db))

    con.close()
else:
    print('Skipping analysis of connectivity between lake combinations on request.')

#############################################################################################################################
if 'lake_summary' in sections:
    if 'network_compilation' not in sections and 'lake_combinations' not in sections:
        g = Graph.Read_Pickle(os.path.join(graph_dir, graph_name))

    lakes = [l.index for l in g.vs.select(area_ha_gt=0)]

    print("Strating connectivity analysis for lakes of the {} clusters".format(len(lakes)))
    start_time_summary = time.time()
    try:
        con = psycopg2.connect(con_string)
    except:
        logging.info("Unable to connect to the database")

    con.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)

    cur = con.cursor()

    # Make sure result tables exist
    if not existsPG(con, pg_result_schema, pg_result_table_summary_master):
        create_summary_master = """SET search_path TO "{0}", public;
        CREATE TABLE IF NOT EXISTS "{1}"(
           wrid integer NOT NULL,
           cluster integer NOT NULL,
           "lakeID" integer NOT NULL,
            typ smallint,
            indegree smallint,
            outdegree smallint,
            neighborhood_size integer,
            cluster_lake_area_ha real,
            cluster_lake_n integer,
            downstream_lakes text,
            downstream_lakes_n integer,
            downstream_lakes_area_ha real,
            first_downstream_lake integer,
            -- downstream_stream_length_km real,
            upstream_lakes text,
            upstream_lakes_n integer,
            upstream_lakes_area_ha real --,
            -- upstream_stream_length_km real
        );
        CREATE INDEX IF NOT EXISTS "{0}_{1}" ON "{1}" USING btree ("lakeID");
        COMMENT ON COLUMN "{1}".wrid IS 'WaterRegionID';
        COMMENT ON COLUMN "{1}"."lakeID" IS 'References column "id" from "{2}"."{3}"';
        COMMENT ON TABLE "{1}" IS
        'Lake connectivity summary hyper table';""".format(pg_result_schema,
                                                     pg_result_table_summary_master,
                                                     lake_table[0],
                                                     lake_table[1])

        cur = con.cursor()
        cur.execute(create_summary_master)

        cur.execute("""SELECT table_name FROM _timescaledb_catalog.hypertable WHERE table_name = '{}';""".format(pg_result_table_summary_master))
        is_hyper = cur.fetchone()
        if not is_hyper:
            cur.execute("""SET search_path TO "{0}", public;
            SELECT create_hypertable('{1}', 'wrid', chunk_time_interval => 1, if_not_exists => TRUE);
            SELECT attach_tablespace('{2}', '{1}', if_not_attached => TRUE);""".format(pg_result_schema, pg_result_table_summary_master, tablespace))

    cur.execute("""DELETE FROM "{0}"."{1}" WHERE wrid = {2};""".format(pg_schema, pg_result_table_summary_master, wrid))

    # Prepare an INSERT statement template
    insert_summary_statement = """INSERT INTO "{0}"."{1}" VALUES (\
    %(wrid)s,\
    %(cluster)s,\
    %(lakeid)s,\
    %(typ)s,\
    %(indegree)s,\
    %(outdegree)s,\
    %(neghborhood_size)s,\
    %(cluster_lake_area_ha)s,\
    %(cluster_lake_n)s,\
    %(downstream_lakes)s,\
    %(downstream_lakes_n)s,\
    %(downstream_lakes_area_ha)s,\
    %(first_downstream_lake)s,\
    %(upstream_lakes)s,\
    %(upstream_lakes_n)s,\
    %(upstream_lakes_area_ha)s\
    );""".format(pg_result_schema, pg_result_table_summary_master)

    g.delete_edges(g.es.select(direction='downstream'))

    # Compute number of vertices that can be reached from each vertex
    # Indicates upstream or downstream position of a node
    g.vs['nbh'] = g.neighborhood_size(mode='out', order=g.diameter())

    cur = con.cursor()
    for l in lakes:
        result_dict = {'wrid': wrid,
                       'cluster': g.vs[l]['cluster'],
                       'lakeid': g.vs[l]['lakeid'],
                       'typ': g.vs[l]['typ'],
                       'indegree': g.vs[l]['indegree'],
                       'outdegree': g.vs[l]['outdegree'],
                       'neghborhood_size': g.vs[l]['nbh'],
                       'cluster_lake_area_ha': g.vs[l]['cluster_lake_area_ha'],
                       'cluster_lake_n': g.vs[l]['cluster_lake_n']
                       }

        downstream_lakes = g.vs[g.neighborhood(g.vs[l], mode='in',
                                               order=g.diameter())].select(area_ha_gt=0)
        # Source lake is included in neighborhood
        if len(downstream_lakes) > 1:
            result_dict['downstream_lakes'] = ','.join(map(str, downstream_lakes['lakeid'][1:]))
            result_dict['downstream_lakes_n'] = len(downstream_lakes) - 1
            result_dict['downstream_lakes_area_ha'] = sum(downstream_lakes['area_ha'][1:])
            result_dict['first_downstream_lake'] = downstream_lakes['lakeid'][1:][0]
            # downstream_stream_length_km real,
        else:
            result_dict['downstream_lakes'] = None
            result_dict['downstream_lakes_n'] = None
            result_dict['downstream_lakes_area_ha'] = None
            result_dict['first_downstream_lake'] = None

        upstream_lakes = g.vs[g.neighborhood(g.vs[l], mode='out',
                                             order=g.diameter())].select(area_ha_gt=0)
        if len(upstream_lakes) > 1:
            result_dict['upstream_lakes'] = ','.join(map(str, upstream_lakes['lakeid'][1:]))
            result_dict['upstream_lakes_n'] = len(upstream_lakes['lakeid']) - 1
            result_dict['upstream_lakes_area_ha'] = sum(upstream_lakes['area_ha'][1:])
            # upstream_stream_length_km real
        else:
            result_dict['upstream_lakes'] = None
            result_dict['upstream_lakes_n'] = None
            result_dict['upstream_lakes_area_ha'] = None

        cur.execute(insert_summary_statement, result_dict)

    con.close()
    proc_time_summary = time.time() - start_time_summary
    print('Done with connectivity summary analysis for wrid {0}. Took: {1} sec.'.format(wrid, int(proc_time_summary)))
    # Append to pandas data frame

    # Write to StringIO

    # Copy to PG

    # Or use insert solution
else:
    print('Skipping analysis of lake connectivity summary on request.')

################################################################################
if 'lake_community' in sections:
    # lakes = [l.index for l in g.vs.select(area_ha_gt=0)]

    print("Strating connectivity analysis for lakes of the {} clusters".format(len(lakes)))
    start_time_summary = time.time()
    try:
        con = psycopg2.connect(con_string)
    except:
        logging.info("Unable to connect to the database")

    con.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)

    cur = con.cursor()

    # Make sure result tables exist
    create_community_master = """SET search_path TO "{0}", public;
    CREATE TABLE IF NOT EXISTS "{1}"(
       wrid integer NOT NULL,
       cluster integer NOT NULL,
       "lakeID" integer NOT NULL,
        typ smallint,
        indegree smallint,
        outdegree smallint,
        neighborhood_size integer,
        cluster_lake_area_ha real,
        cluster_lake_n integer,
        downstream_lakes text,
        downstream_lakes_n integer,
        downstream_lakes_area_ha real,
        first_downstream_lake integer,
        -- downstream_stream_length_km real,
        upstream_lakes text,
        upstream_lakes_n integer,
        upstream_lakes_area_ha real --,
        -- upstream_stream_length_km real
    );
    COMMENT ON COLUMN "{1}".wrid IS 'WaterRegionID';
    COMMENT ON COLUMN "{1}"."lakeID" IS 'References column "id" from nofa.lake';
    COMMENT ON TABLE "{1}" IS
    'Lake connectivity summary hyper table';""".format(pg_result_schema,
                                                 pg_result_table_community_master)

    cur = con.cursor()
    cur.execute(create_summary_master)

    # Prepare an INSERT statement template
    insert_community_statement = """INSERT INTO "{0}"."{1}_{2}" VALUES (\
    %(wrid)s,\
    %(cluster)s,\
    %(lakeid)s,\
    %(typ)s,\
    %(indegree)s,\
    %(outdegree)s,\
    %(neghborhood_size)s,\
    %(cluster_lake_area_ha)s,\
    %(cluster_lake_n)s,\
    %(downstream_lakes)s,\
    %(downstream_lakes_n)s,\
    %(downstream_lakes_area_ha)s,\
    %(first_downstream_lake)s,\
    %(upstream_lakes)s,\
    %(upstream_lakes_n)s,\
    %(upstream_lakes_area_ha)s\
    );""".format(pg_result_schema, pg_result_table_community_master, wrid)

    slope = 1.00
    while slope >= 0.0 and g.ecount() > 0:
        g.delete_edges(g.es.select(slope_max_max_lt=slope))
        g.vs['cluster_slope{}'.format(str(slope).replace('.', '_'))] = g.as_undirected().clusters().membership
        slope = slope - 0.005

        cur.execute(insert_community_statement, result)

    con.close()
    proc_time_summary = time.time() - start_time_summary
    print('Done with analysis of lake communities for wrid {0}. Took: {1} sec.'.format(wrid, int(proc_time_summary)))
    # Append to pandas data frame

    # Write to StringIO

    # Copy to PG

    # Or use insert solution
else:
    print('Skipping analysis of lake communities on request.')


proc_time_total = time.time() - start_time_total
print("Done with waterrgion {0}. Took: {1} sec in total.".format(wrid,
                                                                 proc_time_total))
