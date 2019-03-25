import os, re, warnings, glob
# warnings.filterwarnings("always")
# warnings.filterwarnings("ignore", message="numpy.dtype size changed")
# warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
warnings.filterwarnings('ignore')
#warning suppresions have to be before urbanatlas
import urbanatlas as ua
import classes


Country = 'UK'
window_km = 50
# image_size = 224
img_area = (224 * 1.19 / 1000) ** 2  # in km^2, at zoom level 17
# this represents which polygons we will select
# if we say thresh_frac = 0.5, that means the polygons have to have atleast 50% of the area as the
# google earth image downloaded
thresh_frac = 1
thresh_area = img_area * thresh_frac
all_classes = classes.get_classes()

N_SAMPLES_PER_CITY = 20000
N_SAMPLES_PER_CLASS = N_SAMPLES_PER_CITY / len(all_classes)
MAX_SAMPLES_PER_POLY = 15

# def fn_process_path(s):
#     b = os.path.basename(s).split(".")[0]
#     country = b.split("_")[0]
#     city = " ".join(b.split("_")[1:])
#     country = re.findall("[a-zA-Z]+", country)[0]
#     return (city, country)

def generate_datapoints(city, OUT_PATH_CSV, all_classes):
    shapefn = city.split('/')[-1]
    city_name = shapefn[:-4]
    print "\nProcessing %s"%city


    mycity = ua.UAShapeFile(city, name=city_name)
    # if mycity._gdf is None:
    #     return "Error reading shapefile %s"%shapefn
    lonmin, latmin, lonmax, latmax = mycity._bounds
    city_center = ((latmin+latmax)/2.0, (lonmin+lonmax)/2.0)
    window = (window_km, window_km)
    mycity_crop = mycity.crop_centered_window(city_center, window)
    locations_train = mycity_crop.generate_sampling_locations(all_classes = all_classes,thresh_area=thresh_area, \
                                                         n_samples_per_class=N_SAMPLES_PER_CLASS,\
                                                         max_samples=MAX_SAMPLES_PER_POLY)
    locations_train.to_csv("%s/%s.csv"%(OUT_PATH_CSV,city_name),encoding = 'utf-8')
    print('Sucessfully saved %s file'%city_name)

OUT_PATH_CSV = "extracted_data/%s/data_points/"%(Country)
if not os.path.exists(OUT_PATH_CSV):
    os.makedirs(OUT_PATH_CSV)
all_shapefiles =  'Cities/%s/Shapefiles_with_pop/*.shp'%(Country)

for city in glob.glob(all_shapefiles):
    shapefn = city.split('/')[-1]
    city_name = shapefn[:-4]
    savename = "%s/%s_train.csv"%(OUT_PATH_CSV,city_name)
    if not os.path.isfile(savename):
        generate_datapoints(city, OUT_PATH_CSV, all_classes)
    else:
        print('already processed %s'%savename)
