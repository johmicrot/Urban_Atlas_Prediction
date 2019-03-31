#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Creator: John Rothman"""

import os
import glob
import urbanatlas as ua
import classes

Country = 'ENTER COUNTRY HERE'
window_km = 50
img_area = (224 * 1.19 / 1000) ** 2  # in km^2, at zoom level 17

# Thresh_frac represents which polygons we will select
# if we say thresh_frac = 0.5, that means the polygons have to have atleast 50% of the area as the
# google earth image downloaded
THRESH_FRAC = 1
thresh_area = img_area * THRESH_FRAC
all_classes = classes.get_classes()

N_SAMPLES_PER_CITY = 20000
N_SAMPLES_PER_CLASS = N_SAMPLES_PER_CITY / len(all_classes)
MAX_SAMPLES_PER_POLY = 15
OUT_PATH_CSV = 'extracted_data/%s/data_points/' % Country
all_shapefiles = 'Cities/%s/Shapefiles_with_pop/*.shp' % Country

if not os.path.exists(OUT_PATH_CSV):
    os.makedirs(OUT_PATH_CSV)


def generate_datapoints(city, out_path_csv, curr_classes, shapefn, city_name):
    print("\nProcessing %s" % city)

    mycity = ua.UAShapeFile(city, name=city_name)
    if mycity._gdf is None:
         return "Error reading shapefile %s" % shapefn
    lonmin, latmin, lonmax, latmax = mycity._bounds
    naive_city_center = ((latmin+latmax)/2.0, (lonmin+lonmax)/2.0)
    window = (window_km, window_km)
    mycity_crop = mycity.crop_centered_window(naive_city_center, window)
    locations_train = mycity_crop.generate_sampling_locations(all_classes = curr_classes,
                                                              thresh_area=thresh_area,
                                                              n_samples_per_class=N_SAMPLES_PER_CLASS,
                                                              max_samples=MAX_SAMPLES_PER_POLY)
    locations_train.to_csv("%s/%s.csv" % (out_path_csv, city_name), encoding='utf-8')
    print('Successfully saved %s file' % city_name)


for city in glob.glob(all_shapefiles):
    shapefn = city.split('/')[-1]
    city_name = shapefn[:-4]
    savename = "%s/%s_train.csv" % (OUT_PATH_CSV, city_name)
    if not os.path.isfile(savename):
        generate_datapoints(city, OUT_PATH_CSV, all_classes, shapefn, city_name)
    else:
        print('already processed %s' % savename)
