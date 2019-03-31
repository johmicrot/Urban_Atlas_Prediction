#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Creator: John Rothman"""

import glob
import os
import numpy as np
from simpledbf import Dbf5
import geopandas as gpd
COUNTRY = 'PUT COUNTRY NAME HERE'
COUNTRY_ABBREV = 'PUT COUNTRY ABBREVIATION HERE'

all_population_files = 'Cities/AllPopulationEstimates/pop_%s/*.dbf' % COUNTRY_ABBREV
all_shapesfiles = 'Cities/%s/Shapefiles_Original/*/Shapefiles/*_UA2012.shp' % COUNTRY
shp_folder = COUNTRY + '/Shapefiles_pop_test/'

if not os.path.exists(shp_folder):
    os.makedirs(shp_folder)
for file in glob.glob(all_shapesfiles):
    print('\n' + file)
    fl = file.split('/')
    shape_file_name = fl[-1]
    file_base = shape_file_name[:-11]  # -11 is to remove the _UA2012.shp
    current_city = shape_file_name.split('_')[0]

    new_shp = shp_folder + file_base + '.shp'
    print(new_shp)

    # If the file already exists, skip it
    if not os.path.isfile(new_shp):
        pop_file = ''
        for tmp_pop_file in glob.glob(all_population_files):

            if current_city in tmp_pop_file:
                pop_file = tmp_pop_file
        if pop_file == '':
            print('No population file found for %s'%file)
            continue

        print('Population file being used: ', pop_file)

        pop_df = Dbf5(pop_file).to_dataframe()

        pop_list = []
        scaled_pop = []
        df = gpd.read_file(file)
        for i,row in df.iterrows():
            pop_value = int(pop_df[pop_df.UATL_ID == row['IDENT']].Pop_tot)
            scaled_pop.append(pop_value/row['Shape_Area'])
            pop_list.append(pop_value)
        df['Pop_tot'] = np.array(pop_list)
        df['Pop_tot_scaled'] = np.array(scaled_pop)
        print('completed writing %s'%new_shp)
    else:
        print('Already ran this file')
