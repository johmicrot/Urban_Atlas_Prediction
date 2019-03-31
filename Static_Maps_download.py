#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Creator: John Rothman"""

import numpy as np
import pandas as pd
import os
import glob
from pysatapi.googlemaps.GoogleMaps import GoogleMaps
import classes

COUNTRY = 'ENTER COUNTRY NAME HERE'

all_classes = classes.get_classes()
GMKEY= 'google key goes here'
gmClient = GoogleMaps(key=GMKEY)

# settings for the google api
MAX_REQUESTS = 50000
MAX_TRIES = 5
IMG_SIZE = 224
ZOOM = 17
N_REQUESTS = 0

base_path = 'extracted_data/%s' % COUNTRY
locations_path = '%s/all_data/*'%base_path
# many of the images you download are blurry. If you don't have a list to keep track of the blurry images then
# every time you re-run the code it will attempt to download the blurry images again.  So to prevent this we create
# a list of image names that are blurry.
blurry_fn = base_path + '/blurries.csv'
out_images = base_path + '/images'
out_points = base_path + '/points'
if not os.path.exists(out_images):
    os.makedirs(out_images)
if not os.path.exists(out_points):
    os.makedirs(out_points)
if os.path.exists(blurry_fn):
    blurry_img_list = pd.read_csv(blurry_fn)['file_name'].tolist()
else:
    blurry_img_list = []


def load_locations(path):
    grid_locations_df = pd.read_csv(path)
    columns = ["lon", "lat", "class", 'pop']
    location = pd.concat([grid_locations_df[columns]])
    location= location.reset_index().drop("index", 1)
    return location


def download_images(file_name, locations,curr_city_name, out_path="./"):
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    global N_REQUESTS
    global blurry_img_list

    datapoints_file = open(file_name, "a")
    #,filename,land_type_pred,pop_pred,land_name,pop,filesize
    header = "%s,%s,%s,%s,%s,%s,%s,%s" % ('lat', 'lon', 'pop', 'land_name',
                                          'class1', 'land_dist', 'filename', 'filesize')
    datapoints_file.write("%s\n" % header)

    for i, r in locations.iterrows():
        land_type, pop, lat, lon = r['class'], r['pop'], r['lat'], r['lon']
        land_type_str = ''
        for elm in land_type.split(','):
            land_type_str += str(elm) + ' '
        basename = "%f_%f" % (lat, lon)
        base_jpg = basename + '.jpg'
        cur_filename = '%s/%s/%s.jpg' % (out_images, curr_city_name, basename)

        # Here we grab the index value and the land name from the index value
        lt_list = land_type.strip('[]').split(',')
        introw = pd.to_numeric(lt_list)
        index = np.argmax(introw)
        land_name = all_classes[index]
        value = str(introw[index])

        # Conditional to make sure that 99% of the image is only one class
        if float(value) >= 0.99:
            # If the current image is in the blurry list skip it
            if cur_filename in blurry_img_list:
                continue
            # If the image has already been downloaded
            if os.path.exists(cur_filename):
                fs = str(os.path.getsize(cur_filename))
                file_line = "%f,%f,%s,%s,%s,%s,%s,%s" % (lat, lon, pop, land_name, index, land_type_str, base_jpg, fs)
                # Add it to the datapoints csv
                datapoints_file.write("%s\n" % file_line)
                continue
            # Try catch is a simple fix to detect if the blurry image warning comes up.
            # If the warning comes up we don't save the image and add the name to our blurry_list
            try:
                print("Pulling image %d/%d... (# API requests = %d)" % (i, len(locations), N_REQUESTS))
                req = gmClient.construct_static_url((lat, lon), maptype="satellite", zoom=ZOOM,
                                                    imgsize=(int(IMG_SIZE * 1.18), int(IMG_SIZE * 1.18)))
                img = gmClient.get_static_map_image(req, filename=cur_filename,
                                                    max_tries=MAX_TRIES,
                                                    crop=True)
                if img is None:

                    print("API requests quota exceeded!")
                    continue

            except UserWarning:
                blurry_img_list.append(cur_filename)
                print('Image %s is blurry' % cur_filename)
                continue

            if N_REQUESTS >= MAX_REQUESTS:
                print('Max requests set')
                break

            # File size (fs) is used to help exclude images that have too low size and appear blurry
            fs = str(os.path.getsize(cur_filename))
            file_line = "%f,%f,%s,%s,%s,%s,%s,%s" % (lat, lon, pop, land_name, index, land_type_str, base_jpg, fs)
            datapoints_file.write("%s\n" % file_line)
            print('successfully wrote image %s' % cur_filename)
            N_REQUESTS += 1

    datapoints_file.close()


for city in glob.glob(locations_path):
    print('downloading city: %s\n' % city)

    # The 4 comes from removing file extension
    city_name = city.split('/')[-1][:-4]
    extraction_path = '%s/%s' % (out_images, city_name)
    locations = load_locations(city)
    locations.groupby("class").apply(len)

    filename = '%s/%s.csv' % (out_points, city_name)

    # Quick fix to delete any information if the previous file existed
    filehandle = open(filename, "w").close()
    download_images(filename, locations, city_name, out_path=extraction_path)

    blurry_img_list_df = pd.DataFrame(blurry_img_list, columns=['file_name'])
    blurry_img_list_df.to_csv(blurry_fn)
