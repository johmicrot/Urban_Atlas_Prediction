import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
import numpy as np , pandas as pd ,os, sys, subprocess, glob, re
from pysatapi.googlemaps.GoogleMaps import GoogleMaps
import seaborn as sns
import classes


warnings.filterwarnings("error")
def remove_junk_str(input):
    out = re.sub(' +','_', re.sub(r'([^\s\w]|_)+', '', input.lower()))
    return out
STATE = 'UK'



all_classes = classes.get_classes()
# label2class = {i: c for i, c in enumerate(all_classes)}
# class2label = {c: i for i, c in enumerate(all_classes)}


sns.set_style("whitegrid", {'axes.grid' : False})
# johns.roth.german key
# AIzaSyDWwDgtiqyU6FCf8Lt5EzZO_t1gcXzJrrI
# AIzaSyDjdCecsFSug2VO2q9LUuBPqUkhKh1_AFI

#johns.roth key
# AIzaSyCXcywTj0meLdmQ3eqaHKVQ_j2IodUinAY
GMKEY= 'AIzaSyDjdCecsFSug2VO2q9LUuBPqUkhKh1_AFI' #my key
# FILENAME = 'train_sample_locations_1_thresh_25kimg.csv'
gmClient = GoogleMaps(key=GMKEY)

MAX_REQUESTS = 50000
MAX_TRIES    = 5
img_size     = 224
ZOOM         = 17
n_requests = 0
# global_counter = 0

base_path = 'extracted_data/%s' % STATE
locations_path = '%s/all_data_15x/*'%base_path
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


# try:
#     blurry_list = pd.read_csv(blurry_fn)['file_name'].tolist()
# except:
#     blurry_list= []


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
    # global global_counter
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
                print('global counter: %d' % global_counter)
                # global_counter += 1
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
                # blank image
                if img is None:
                    # noncounter += 1
                    # if noncounter > 10:
                    #     break #exit out of the current shapefile
                    print "API requests quota exceeded!"
                    continue

            except UserWarning:
                blurry_list.append(cur_filename)
                print('Image %s is blurry'%cur_filename)
                continue

            if n_requests >= MAX_REQUESTS:
                print 'Max requests set'
                break #uncomment the return false when i run it for the first time
                # return False

            #file size (fs) is used to help exclude images that have too low size and appear blurry
            fs = str(os.path.getsize(cur_filename))
            file_line = "%f,%f,%s,%s,%s,%s,%s,%s" % (lat, lon, pop, land_name, index, land_type_str, base_jpg,fs)
            # file_line = "%d,%s,%s,%f,%f,%s,%s" % (i, pop, land_type_str, lat, lon, land_name, value)
            datapoints_file.write("%s\n" % file_line)
            print('successfully wote image %s'%cur_filename)
            # global_counter += 1
            # noncounter = 0
            n_requests += 1

    datapoints_file.close()

# for city in ['extracted_data/Germany/train/DE011L1_DUSSELDORF_train.csv']:
for city in glob.glob(locations_path):
    print('downloading city: %s\n'%city)
    # the 4 comes from the name typically is called
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
# print('global counter : ', global_counter)
# os.spawnve("P_WAIT", "/bin/ls", ["/bin/ls"], {})
# os.execve("/bin/ls", ["/bin/ls"], os.environ)
