import numpy as np pandas as pd ,os, glob, re
from pysatapi.googlemaps.GoogleMaps import GoogleMaps
import classes

def remove_junk_str(input):
    out = re.sub(' +','_', re.sub(r'([^\s\w]|_)+', '', input.lower()))
    return out

STATE = 'ENTER COUNTRY NAME HERE'

all_classes = classes.get_classes()
GMKEY= 'google key goes here' #my key
gmClient = GoogleMaps(key=GMKEY)

MAX_REQUESTS = 50000
MAX_TRIES    = 5
img_size     = 224
ZOOM         = 17
n_requests = 0
# global_counter = 0

base_path = 'extracted_data/%s' % STATE
locations_path = '%s/all_data/*'%base_path
blurry_fn = base_path + '/blurries.csv'
out_images = base_path + '/images'
out_points = base_path + '/points'
if not os.path.exists(out_images):
    os.makedirs(out_images)
if not os.path.exists(out_points):
    os.makedirs(out_points)

# many of the images you download are blurry. If you don't have a list to keep track of the blurry images then
# every time you re-run the code it will attempt to download the blurry images again.  So to prevent this we create
# a list of image names that are blurry.
if os.path.exists(blurry_fn):
    blurry_list = pd.read_csv(blurry_fn)['file_name'].tolist()
else:
    blurry_list = []

def load_locations(path):
    grid_locations_df = pd.read_csv(path)
    columns = ["lon", "lat", "class", 'pop']
    locations = pd.concat([grid_locations_df[columns]])
    locations = locations.reset_index().drop("index", 1)
    return locations
#

def download_images(file_name, locations,city_name, out_path="./"):
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    global n_requests
    global blurry_list

    datapoints_file = open(file_name, "a")
    header = "%s,%s,%s,%s,%s,%s,%s,%s" % ( 'lat', 'lon','pop','land_name', 'class1','land_dist','filename','filesize')
    datapoints_file.write("%s\n" % header)

    for i,r in locations.iterrows():
        land_type, pop, lat, lon = r['class'], r['pop'], r['lat'], r['lon']
        land_type_str = ''
        for elm in land_type.split(','):
            land_type_str += str(elm) + ' '
        basename = "%f_%f" % (lat, lon)
        base_jpg = basename + '.jpg'
        cur_filename = '%s/%s/%s.jpg' % (out_images, city_name, basename)

        lt_list = land_type.strip('[]').split(',')
        introw = pd.to_numeric(lt_list)
        index = np.argmax(introw)
        land_name = all_classes[index]
        value = str(introw[index])
        #if statement to make sure that 99% of the image is only one class
        if float(value) >= 0.99:
            #if the current image is in the blurry list skip it
            if cur_filename in blurry_list:
                continue
            #if the image has already been downloaded
            if os.path.exists(cur_filename):
                fs = str(os.path.getsize(cur_filename))
                file_line = "%f,%f,%s,%s,%s,%s,%s,%s" % ( lat, lon, pop,land_name, index,land_type_str, base_jpg,fs)
                #add it to the datapoints csv
                datapoints_file.write("%s\n" % file_line)
                continue
            # try catch is a simple fix to detect if the blurry image warning comes up.
            # if the warning comes up we don't save the image and add the name to our blurry_list
            try:
                print "Pulling image %d/%d... (# API requests = %d)" % (i, len(locations), n_requests)
                req = gmClient.construct_static_url((lat,lon), maptype="satellite", zoom=ZOOM, \
                                                    imgsize=(int(img_size*1.18), int(img_size*1.18)))
                img = gmClient.get_static_map_image(req, filename=cur_filename, \
                                                    max_tries=MAX_TRIES,\
                                                    crop=True)
                if img is None:

                    print "API requests quota exceeded!"
                    continue

            except UserWarning:
                blurry_list.append(cur_filename)
                print('Image %s is blurry'%cur_filename)
                continue

            if n_requests >= MAX_REQUESTS:
                print 'Max requests set'
                break

            #file size (fs) is used to help exclude images that have too low size and appear blurry
            fs = str(os.path.getsize(cur_filename))
            file_line = "%f,%f,%s,%s,%s,%s,%s,%s" % (lat, lon, pop, land_name, index, land_type_str, base_jpg,fs)
            datapoints_file.write("%s\n" % file_line)
            print('successfully wote image %s'%cur_filename)
            n_requests += 1

    datapoints_file.close()

for city in glob.glob(locations_path):
    print('downloading city: %s\n'%city)
    # the 4 comes from removing file extension
    city_name = city.split('/')[-1][:-4]
    extraction_path = '%s/%s'%(out_images, city_name)
    locations = load_locations(city)
    locations.groupby("class").apply(len)

    filename = '%s/%s.csv'%(out_points,city_name)

    # quick fix to delete any information if the previous file existed
    filehandle = open(filename, "w").close()
    download_images(filename, locations, city_name, out_path=extraction_path)

    blurries = pd.DataFrame(blurry_list,columns=['file_name'])
    blurries.to_csv(blurry_fn)
