# Analysing lake connectivity with connectivity.py

## Requirements
The following non-standard python libraries are required to run the tool:
 - [numpy](https://numpy.org/): For matrix operations
 - [psycopg2](https://pypi.org/project/psycopg2/): For access to PostgreSQL/PostGIS databases
 - [igraph](https://igraph.org/python/): For network analysis
 - [pandas](https://pandas.pydata.org/): For more convenient matrix operations

In addtion, access to a [PostgreSQL](https://www.postgresql.org/) database with the extensions
[PostGIS](https://postgis.net/) [TimescaleDB](https://www.timescale.com/) is required as well as
[GRASS GIS (>=7.8)](https://grass.osgeo.org/) for hydrological analysis and preparation of the
network data.

Make sure that these are installed on your system.

Connection information for accessing the required PostgreSQL database are taken from a password file
([.pgpass](https://www.postgresql.org/docs/9.3/libpq-pgpass.html)) that resides in the users home
directory.

## Functionality

### Processing sections

The tool consists of four sections
 - **gis**: Combines lakes and stream network data and produces the required attributes for further
 analystis. Results are written into a TimescaleDB Hypertable in the target database, as well
 as into a GRASS GIS mapset. This process benefits mostly from Random Access Memory (RAM)
 (see *memory* option). it is not parallelised internally, due to the nature of the analysis,
 so the *cores* option has no effect. However, the process can be parallelized by running it
 on several catchments in parallel (see examples below). This is mainly a pre-processing step.
 - **network_compilation**: Generates an igraph network dataset from the spatial data and saves it
 as zipped and pickled Python object. This is mainly a pre-processing step.
 - **lake_combinations**: Analyses slope characteristics of the streams connecting all pairs of lakes
 in both upstream and downstream direction. This is the most CPU intensive section and benefits
 from more cores (see *memory* option) allocated to the process.
 - **lake_summary**: Summarizes the situation of a lake in the river network 

### Final results

#### lake_combinations

| wrid   | from_lake | to_lake | cluster | lakes_along                                                                                                                                                                                                                                                                                     | confluences_along                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | total_stream_length | total_slope_max_max | upstream_length | upstream_altitude_min | upstream_altitude_mean | upstream_altitude_max | upstream_slope_mean | upstream_slope_third_quart | upstream_slope_perc_90 | upstream_slope_perc_90_max | upstream_slope_max | upstream_slope_max_max | upstream_slope_stddev | downstream_length | downstream_altitude_min | downstream_altitude_mean | downstream_altitude_max | downstream_slope_mean | downstream_slope_third_quart | downstream_slope_perc_90 | downstream_slope_perc_90_max | downstream_slope_max | downstream_slope_max_max | downstream_slope_stddev | 
|--------|-----------|---------|---------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------|---------------------|-----------------|-----------------------|------------------------|-----------------------|---------------------|----------------------------|------------------------|----------------------------|--------------------|------------------------|-----------------------|-------------------|-------------------------|--------------------------|-------------------------|-----------------------|------------------------------|--------------------------|------------------------------|----------------------|--------------------------|-------------------------| 
| 252756 | 2779430   | 2779746 | 0       | 2821359,4108168,4108282,2859813,2566059,4106300,4106306,2828828,3516970,3516983,4107589,3518140,4108576,3519063,3520175,3520414,3521868,2572511,2901561,3524207,2937404,2842661,2646927,2779430,2779746,3516399,2915913                                                                         | 4108168,4119106,4108282,4119517,2566059,4106300,4106306,4119563,3516983,4107589,4119825,3518140,4108576,3519063,3520175,3520414,3521868,2572511,3524207,4119993,4121141,4121313,4117282,4117329,4117330,4117392,4117407,4117408,4117418,4117422,4117431,4117432,4117463,4117492,4117637,4117643,4117661,4117730,4117733,4117734,4117738,4117763,4117764,4117772,4117790,4117853,4117859,4117872,4117873,4117938,4117939,4117997,4118013,4118060,4118108,4118114,4118115,4118118,4118146,4118228,4118365,4118366,4118446,4118550,4118608,4118634,4118656,4118657,4118666,4118667,4118844,4118890,4118995,4119094,4119117,4119178,4119179,4119204,4119280,4119348,4119386,4119450,4119558,4119559,4119562,4119613,4119640,4119644,4120149,4120258,4120266,4120276,4120332,4120506,4120550,4120551,4120622,4120623,4120814,4121016,4121219,4121566,3516399,4123168                                                         | 79759               | 7263                | 75400           | 137                   | 236                    | 828                   | 275                 | 371                        | 790                    | 4703                       | 2056               | 7263                   | 407                   | 4359              | 140                     | 203                      | 257                     | 775                   | 948                          | 2173                     | 3642                         | 4298                 | 5463                     | 1082                    | 
| 252756 | 2779430   | 2779931 | 0       | 2821359,4108168,3514284,2828828,3516970,2829726,3518140,3521868,2866509,2802611,2937404,2844205,2779430,2779931                                                                                                                                                                                 | 4108168,3514284,2829726,3518140,4119997,3521868,2866509,4119574,2802611,4119993,4121141,4117316,4117375,4117391,4117392,4117400,4117422,4117431,4117432,4117484,4117492,4117498,4117529,4117537,4117538,4117559,4117597,4117646,4117647,4117733,4117734,4117763,4117764,4117790,4117818,4117872,4117873,4117890,4117930,4117931,4118001,4118013,4118108,4118143,4118144,4118173,4118194,4118328,4118346,4118347,4118394,4118474,4118491,4118550,4118558,4118666,4118667,4118716,4118731,4118748,4118750,4118769,4118886,4118890,4118936,4119013,4119055,4119177,4119332,4119386,4119572,4119674,4119996,4120276,4120382,4120506,4120550,4120551,4121566,2844205                                                                                                                                                                                                                                                         | 59555               | 7263                | 55196           | 137                   | 214                    | 780                   | 323                 | 453                        | 846                    | 6732                       | 2216               | 7263                   | 437                   | 4359              | 140                     | 203                      | 257                     | 775                   | 948                          | 2173                     | 3642                         | 4298                 | 5463                     | 1082                    | 
| 252756 | 2779430   | 2787776 | 0       | 2787776,2821359,4108168,2828828,3516970,3518140,2899032,3521868,2605774,2803480,2937404,2709913,2615085,2779430,2748738                                                                                                                                                                         | 2787776,4108168,3518140,3521868,4120850,4117878,4119575,4119916,4120261,4120276,4120577,4120578,4121566                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | 7898                | 5628                | 3861            | 140                   | 230                    | 296                   | 633                 | 907                        | 1629                   | 5037                       | 2896               | 5628                   | 808                   | 4037              | 141                     | 207                      | 257                     | 697                   | 843                          | 2038                     | 2822                         | 4179                 | 5188                     | 1019                    | 
| 252756 | 2770379   | 2774450 | 0       | 2774450,2759088,4108063,3520392,2770379,3514424                                                                                                                                                                                                                                                 | 2774450,4108063,3520392,3514424                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | 1715                | 5506                | 636             | 612                   | 620                    | 623                   | 294                 | 384                        | 1014                   | 1014                       | 1738               | 1738                   | 463                   | 1079              | 611                     | 671                      | 770                     | 725                   | 760                          | 1741                     | 1969                         | 4908                 | 5506                     | 1143                    | 
| 252756 | 2779430   | 2780019 | 0       | 2819577,2624000,2821359,2724908,4108168,4108282,3514635,2859813,2566059,4106300,4106306,4106379,2828828,3516970,3517616,4107589,3518140,4108576,3519063,2994782,4109332,3520149,3520414,3521868,2572511,2901561,3524207,2902009,2937404,2645756,2646927,2779430,2780019,3516399,2750099,2915913 | 2819577,2624000,4108168,4119106,4126913,4108282,3514635,4119517,2566059,4106300,4106306,4119563,4106379,4119633,3517616,4107589,4119825,3518140,4108576,3519063,4109332,3520149,3520414,3521868,2572511,3524207,4119993,4121141,4121313,4117329,4117330,4117392,4117407,4117408,4117418,4117422,4117431,4117432,4117463,4117492,4117637,4117643,4117661,4117730,4117733,4117734,4117735,4117736,4117738,4117763,4117764,4117772,4117790,4117853,4117859,4117872,4117873,4117938,4117939,4117997,4118013,4118060,4118108,4118114,4118115,4118118,4118146,4118228,4118365,4118366,4118446,4118550,4118608,4118634,4118656,4118657,4118666,4118667,4118844,4118890,4118995,4119094,4119117,4119178,4119179,4119280,4119348,4119386,4119450,4119558,4119559,4119562,4119613,4119640,4119644,4120149,4120258,4120266,4120276,4120332,4120506,4120550,4120551,4120622,4120623,4120814,4121016,4121219,4121566,3516399         | 84776               | 7263                | 80417           | 137                   | 273                    | 1124                  | 311                 | 429                        | 859                    | 5588                       | 2169               | 7263                   | 441                   | 4359              | 140                     | 203                      | 257                     | 775                   | 948                          | 2173                     | 3642                         | 4298                 | 5463                     | 1082                    | 
| 252756 | 2779430   | 2780066 | 0       | 2821359,2625643,3544418,3513125,4108168,3514474,2696995,2828828,3516970,3518140,4108063,4108456,2863646,2700297,2603890,2735021,3521868,2572427,2802611,2575378,2937404,2844205,2779430,2780066                                                                                                 | 3544418,4118862,3513125,4108168,3514474,3518140,4108063,4108456,3521868,4119574,2802611,4119993,4121141,4117289,4117290,4117316,4117356,4117357,4117374,4117375,4117391,4117392,4117400,4117422,4117431,4117432,4117473,4117484,4117492,4117494,4117498,4117537,4117538,4117555,4117559,4117597,4117646,4117647,4117733,4117734,4117763,4117764,4117769,4117770,4117788,4117790,4117794,4117818,4117848,4117872,4117873,4117890,4117930,4117931,4118001,4118013,4118021,4118022,4118062,4118063,4118067,4118081,4118082,4118108,4118113,4118173,4118194,4118328,4118346,4118347,4118368,4118394,4118474,4118491,4118549,4118550,4118558,4118666,4118667,4118716,4118731,4118748,4118750,4118769,4118863,4118886,4118890,4118936,4119006,4119013,4119055,4119177,4119332,4119386,4119494,4119572,4119653,4119674,4119689,4119797,4119929,4119937,4120276,4120382,4120506,4120550,4120551,4120619,4121314,4121566,2844205 | 75245               | 7263                | 70886           | 137                   | 267                    | 849                   | 278                 | 364                        | 732                    | 5237                       | 2085               | 7263                   | 407                   | 4359              | 140                     | 203                      | 257                     | 775                   | 948                          | 2173                     | 3642                         | 4298                 | 5463                     | 1082                    | 
| 252756 | 2779430   | 2780150 | 0       | 2621760,2821359,4108168,2828828,3516970,3518140,3521868,2937404,2779430,2780150                                                                                                                                                                                                                 | 4108168,4119682,3518140,3521868,4117492,4117759,4118403,4118666,4118667,4119046,4120276,4121566                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | 10477               | 7263                | 6118            | 138                   | 198                    | 406                   | 962                 | 1367                       | 2534                   | 3559                       | 5388               | 7263                   | 1387                  | 4359              | 140                     | 203                      | 257                     | 775                   | 948                          | 2173                     | 3642                         | 4298                 | 5463                     | 1082                    | 
| 252756 | 2779430   | 2780164 | 0       | 2821359,3544418,3513125,4108168,2828828,3516970,3518140,4108063,4108456,3521868,2572427,2802611,2575378,2937404,2578180,2844205,2779430,2780164                                                                                                                                                 | 3544418,3513125,4108168,3518140,4108063,4108456,3521868,4119574,2802611,4120740,4119993,4121141,4117289,4117290,4117316,4117374,4117375,4117391,4117392,4117400,4117422,4117431,4117432,4117473,4117484,4117492,4117498,4117537,4117538,4117555,4117559,4117597,4117646,4117647,4117733,4117734,4117763,4117764,4117769,4117770,4117788,4117790,4117794,4117818,4117848,4117872,4117873,4117890,4117930,4117931,4118001,4118013,4118021,4118022,4118062,4118063,4118108,4118113,4118173,4118194,2578180,4118328,4118346,4118347,4118394,4118474,4118491,4118549,4118550,4118558,4118666,4118667,4118716,4118731,4118748,4118750,4118769,4118886,4118890,4118936,4119013,4119055,4119177,4119332,4119386,4119494,4119572,4119653,4119674,4119689,4119937,4120276,4120382,4120506,4120550,4120551,4120619,4121103,4121566,2844205                                                                                         | 69821               | 7263                | 65462           | 137                   | 233                    | 635                   | 259                 | 333                        | 684                    | 5237                       | 2077               | 7263                   | 394                   | 4359              | 140                     | 203                      | 257                     | 775                   | 948                          | 2173                     | 3642                         | 4298                 | 5463                     | 1082                    | 
| 252756 | 2779430   | 2780433 | 0       | 2821359,4108168,2828828,3516970,3518140,3520804,3521868,2937404,2779430,2780433,2620072                                                                                                                                                                                                         | 4108168,3518140,3520804,3521868,4120276,4121566,2620072                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | 5591                | 5188                | 1554            | 139                   | 191                    | 228                   | 720                 | 901                        | 2375                   | 3017                       | 3656               | 4453                   | 1067                  | 4037              | 141                     | 207                      | 257                     | 697                   | 843                          | 2038                     | 2822                         | 4179                 | 5188                     | 1019                    | 
| 252756 | 2779430   | 2780496 | 0       | 2821359,4108168,3514287,2828828,3516970,3518140,4108264,3521868,2802611,2937404,2844205,2779430,2681493,2780496                                                                                                                                                                                 | 4108168,3514287,3518140,4108264,3521868,4119574,2802611,4119993,4121141,4117276,4117289,4117290,4117316,4117346,4117374,4117375,4117391,4117392,4117400,4117422,4117431,4117432,4117473,4117484,4117492,4117498,4117537,4117538,4117547,4117559,4117597,4117646,4117647,4117712,4117733,4117734,4117763,4117764,4117790,4117793,4117794,4117818,4117872,4117873,4117890,4117930,4117931,4118001,4118013,4118021,4118022,4118108,4118173,4118194,4118328,4118346,4118347,4118394,4118474,4118491,4118550,4118558,4118666,4118667,4118716,4118731,4118748,4118750,4118769,4118798,4118886,4118890,4118936,4119013,4119055,4119177,4119244,4119297,4119332,4119386,4119494,4119572,4119674,4119937,4120276,4120382,4120506,4120550,4120551,4120752,4121125,4121566,2844205                                                                                                                                                 | 68567               | 7263                | 64208           | 0                     | 225                    | 640                   | 254                 | 337                        | 696                    | 2921                       | 2075               | 7263                   | 387                   | 4359              | 140                     | 203                      | 257                     | 775                   | 948                          | 2173                     | 3642                         | 4298                 | 5463                     | 1082                    | 

#### lake_summary

| wrid   | cluster | lakeID  | typ | indegree | outdegree | neighborhood_size | cluster_lake_area_ha | cluster_lake_n | downstream_lakes                                                        | downstream_lakes_n | downstream_lakes_area_ha | first_downstream_lake | upstream_lakes                  | upstream_lakes_n | upstream_lakes_area_ha | 
|--------|---------|---------|-----|----------|-----------|-------------------|----------------------|----------------|-------------------------------------------------------------------------|--------------------|--------------------------|-----------------------|---------------------------------|------------------|------------------------| 
| 264544 | 0       | 2767558 | 2   | 1        | 1         | 4                 | 538.329              | 50             | 4109257,3270026,3531975,3531973,4109256,3531971                         | 6                  | 238.912                  | 4109257               |                                 |                  |                        | 
| 264544 | 0       | 2715165 | 3   | 2        | 1         | 3                 | 538.329              | 50             | 3531976,3270065,3270052,3270035,3531975,3531973,4109256,3531971         | 8                  | 232.088                  | 3531976               |                                 |                  |                        | 
| 264544 | 0       | 2728478 | 2   | 1        | 1         | 3                 | 538.329              | 50             | 2962746,2550508,3520779,3522434,3531969,4109254,3531971                 | 7                  | 197.399                  | 2962746               | 2721986,276                     | 2                | 1.74474                | 
| 264544 | 0       | 2708055 | 2   | 1        | 1         | 2                 | 538.329              | 50             | 3518222,3520779,3522434,3531969,4109254,3531971                         | 6                  | 207.86                   | 3518222               |                                 |                  |                        | 
| 264544 | 0       | 2625118 | 2   | 1        | 1         | 5                 | 538.329              | 50             | 3522434,3531969,4109254,3531971                                         | 4                  | 155.449                  | 3522434               | 3517217,263                     | 2                | 14.663                 | 
| 264544 | 0       | 2942567 | 2   | 1        | 1         | 5                 | 538.329              | 50             | 2920141,3518222,3520779,3522434,3531969,4109254,3531971                 | 7                  | 207.961                  | 2920141               | 2549523,284                     | 2                | 0.75399                | 
| 264544 | 0       | 2840175 | 2   | 1        | 1         | 2                 | 538.329              | 50             | 2549523,2942567,2920141,3518222,3520779,3522434,3531969,4109254,3531971 | 9                  | 208.431                  | 2549523               |                                 |                  |                        | 
| 264544 | 0       | 2764485 | 1   | 0        | 1         | 1                 | 538.329              | 50             | 2721986,2728478,2962746,2550508,3520779,3522434,3531969,4109254,3531971 | 9                  | 198.292                  | 2721986               |                                 |                  |                        | 
| 264544 | 0       | 3531974 | 3   | 2        | 1         | 6                 | 538.329              | 50             | 3270025,3531973,4109256,3531971                                         | 4                  | 188.779                  | 3270025               | 3270024,3270038,3270040,3270044 | 4                | 4.40346                | 
| 264544 | 0       | 3531968 | 2   | 1        | 1         | 2                 | 538.329              | 50             | 4109256,353                                                             | 2                  | 150.425                  | 4109256               |                                 |                  |                        | 

## Examples on how to use connectivity.py on the LINUX command line

### Get a unique list of wrids from DB for lakes in Agder
#### Select by lakes (and administrative unit)
```
wrids=$(psql -d nofa -W -A -t -F' ' -c "
SELECT DISTINCT ON (a.gid) 
    a.gid
FROM
    \"Hydrography\".\"waterregions_dem_10m_nosefi\" AS a,
    (SELECT geom FROM nofa.lake WHERE county IN ('Vest-Agder', 'Aust-Agder')) AS b
WHERE
    ST_Intersects(a.geom, b.geom);")
```
### Select by administrative unit
```
wrids=$(psql -d nofa -W -A -t -F' ' -c "
SELECT DISTINCT ON (a.gid)
    a.gid
FROM
    \"Hydrography\".\"waterregions_dem_10m_nosefi\" AS a,
    (SELECT
        geom
    FROM
        \"AdministrativeUnits\".\"Fenoscandia_Municipality_polygon\"
    WHERE
        county IN ('Vest-Agder', 'Aust-Agder')) AS b
WHERE
    ST_Intersects(a.geom, b.geom);")
```

### Parallelise in shell (Linux command line); one wrid / core
This is most likely more efficient for a larger number of smaller catchments cause only (parts of) the network analysis are parallelised in the script
```
# Define number of cores to use
cores=20
# Send commands to be processed to xargs for parallel execution with the given number of cores
echo "$wrids" | \
awk '{print "python connectivity.py --db=nofa --schema_prefix=lake_connectivity --wrid=" $1 " --terrain_stream=True &> con" $1 ".log"}' | \
xargs -I{} -P$cores bash -c "{}"

# Or with a manual given list of wrids
echo "143415
143439
149606
150937
150938
150963" | \
awk '{print "python connectivity.py --db=nofa --schema_prefix=lake_connectivity --wrid=" $1 " --terrain_stream=True &> con" $1 ".log"}' | \
xargs -I{} -P$cores bash -c "{}"
```


### Trouble shooting
```
# Get list of logfiles with python errors
grep -i trace *.log

# Get list of logfiles with expected (or unexpected error messages
grep -i error *.log

# Get list of wrids failing with Python error
wrids=$(grep -i trace *.log | cut -f1 -d':' | \
sed -e 's/con//g;s/\.log//g')

cores=$(echo "$wrids" | wc -l)
if [ $cores -gt 20 ] ; then
cores=20
fi
```

Now fix the issues that appeared and re-run sections of the script.
```
# Rerun the wrids that encountered errors (after reasons for that were fixed)
echo "$wrids" | \
awk '{print "python connectivity.py --db=nofa --schema_prefix=lake_connectivity --wrid=" $1 " --terrain_stream=True --sections=lake_combinations,lake_summary &> con" $1 ".log"}' | \
xargs -I{} -P$cores bash -c "{}"
```


### Analyse connections only for a set of lakes of interest (focal lakes)
First column contains the wrid, second lakes within that wrid that should be processed
```
echo "38747 3656360,3656435,3870645,3870967,3893451,3896026,3896027,4105006,4114082,4114090
48528 3652904,3871245,3893623,4104968
66022 3894622,3894627,3897106,4106973,4113077,4115880
67361 2548414,3732613,3873645,4107561
70686 3742191,3776826,3779240,3779255,3901995,3902190,3902264,3902265,3902731,3902738,3905719,3905752,3905753,3905772,3905844,3917600,4108385,4111597,4111613,4113036,4113108,4114440,4114796,4115450
82389 3901343,4115047
98655 3546496,3860697,3862018,3916185,4096958,4102973,4104410,4106222,4110601,4111877,4117398
107799 2548736,3901231,3905692,4113905
114537 2548519,3544967,3906289,3916129,4104355,4114494
117011 3879915,3905882,3905894,3916140
119424 3732933,3778302,3905686,3905829,3917745,4114784
119810 3860465,3879531,3880048,4106213,4110539,4113817
120091 3548123,3550084,3860884,3862801,3881153,3881186,3881187,3881266,3883569,3883574,4098419,4104444,4107505,4110301
121254 3905654,3905657
131095 3545750,3546484,3860617,3860636,3860672,3860794,3880207,3880989
131736 3879965,3879966,3880947,4114398,4114415
158660 3582840,3861140,3862658,3862660,3881716,3881719,3882002,3883516,3883522,3884089,3884090,3884234,3884265,3916352
162902 3861223,3861262,3868058,3881885,3881953,3889926,4099658,4110148
167089 3909801,4106118
169834 3596851,3864328
176618 3889283,3889320,3889327,3889328
178504 3624274,3889361
191459 3868535,4099722
246667 3870298,3891632,4099829
264568 3818211,3910794
279515 3807091,3909435,3909531,3918087
280863 3818369,3876555
281116 3818884,3876595
281651 3878451,3913886
283347 3911976,4105717" | awk '{print "python connectivity.py --db=nofa --schema_prefix=lake_connectivity --wrid=" $1 " --terrain_stream=True --sections=lake_combinations --focal_lakes=" $2 " --memory=10000 --verbose=True &> connectivity_sweden/con_focal_lakes" $1 ".log"}' | xargs -I{} -P30 bash -c "{}"
```
