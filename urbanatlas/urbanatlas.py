# Original code taken from https://github.com/adrianalbert/urban-environments/tree/master/dataset-collection
#This is just a modification of their urbanatlas.py file. It include working code for the UA 2012 dataset, as well as some
import numpy as np, pandas as pd , re , copy
import geopandas as gpd
from shapely.geometry import Polygon

from pysatml.utils import gis_utils as gu
from pysatml.utils import vector_utils as vu
from pysatml import satimage as satimg

CLASS_COL = 'ITEM2012'
POP_COL = 'POP_TOT'
AREA_COL = 'Shape_Area'
LEN_COL = 'Shape_Leng'
N_SAMPLES_PER_CITY  = 50000
N_SAMPLES_PER_CLASS = 3000
MAX_SAMPLES_PER_POLY= 100

class UAShapeFile():
	'''
	Class that encapsulates functionality for analyzing GIS vector data from the Urban Atlas dataset, published as shapefiles. The Urban Atlas is a land use dataset of 300 cities in Europe. 
	'''

	def __init__(self, shapefile, prjfile=None, class_col=CLASS_COL, pop_col= POP_COL, **kwargs):
		'''
		Initialize a GeoPandas GeoDataFrame with settings specific to the published Urban Atlas shape files.
		'''
		self._shapefile = shapefile
		self._class_col = class_col
		self._pop_col = pop_col

		for k,v in kwargs.iteritems():
			setattr(self, "_%s"%k, v)

		# read in shape file
		self._gdf = load_shapefile(self._shapefile)
		if self._gdf is None:
			return 

		self._classes = self._gdf[self._class_col].unique()
		print "%d polygons | %d land use classes" % (len(self._gdf), len(self._classes))

		self._prjfile = shapefile.replace(".shp", ".prj") if prjfile is None else prjfile
		try:
			self._prj = gu.read_prj_file(self._prjfile)  
		except:
			print "Error: cannot find projection file %s" % self._prjfile
			self._prj = ""

		self.compute_bounds()


	def compute_bounds(self):
		# compute bounds for current shapefile for easier access later
		lonmin, latmin, lonmax, latmax = vu.compute_gdf_bounds(self._gdf)
		self._bounds = (lonmin, latmin, lonmax, latmax)
		xmin, ymin = gu.lonlat2xy((lonmin, latmin), prj=self._prj)
		xmax, ymax = gu.lonlat2xy((lonmax, latmax), prj=self._prj)
		self._bounds_meters = (xmin, ymin, xmax, ymax)


	def compute_spatial_extent(self):
		xmin, ymin, xmax, ymax = self._bounds_meters
		L = np.sqrt((xmax-xmin)**2 + (ymax-ymin)**2) / 1.0e3 / np.sqrt(2)
		return L

	def compute_classified_area(self):
		xmin, ymin, xmax, ymax = self._bounds_meters
		box_area =  (xmax-xmin) / 1.0e3 * (ymax-ymin) / 1.0e3
		classified_area = self._gdf\
							.groupby(self._class_col)\
							.apply(lambda x: x[AREA_COL].sum())
		pop = self._gdf[POP_COL].sum()

		frac_classified = classified_area/box_area
		frac_pop = pop /box_area
		return frac_classified,frac_pop, pop

	def filter_by_polygon(self, poly):
		return vu.filter_gdf_by_polygon(self._gdf, poly)

	def crop_centered_window(self, center, window):
		'''
		Returns a UAShapeFile object obtained from original one by cropping a window of (W, H) (in kilometers) around a center (lon, lat).
		'''
		new_self = copy.deepcopy(self)

		new_self._gdf = vu.filter_gdf_by_centered_window(new_self._gdf, center, window)
		new_self.compute_bounds()

		return new_self

	def extract_class_raster(self,all_classes, center=None,window=None,grid_size=(100,100)):
		if center is None:
			lonmin, latmin, lonmax, latmax = self._bounds
			center = ((latmin+latmax)/2.0, (lonmin+lonmax)/2.0)
		if window is not None:
			bbox = gu.bounding_box_at_location(center, window)
		else:
			bbox = self._bounds
		return construct_class_raster(self._gdf, bbox,all_classes, class_col=self._class_col,pop_col=self._pop_col, grid_size=grid_size)


	def generate_sampling_locations(self,all_classes, n_samples_per_class=N_SAMPLES_PER_CLASS,thresh_area=0.25,max_samples=MAX_SAMPLES_PER_POLY):
		gdf_sel = self._gdf[self._gdf.Shape_Area>=thresh_area]
		return generate_sampling_locations(gdf_sel,all_classes = all_classes, n_samples_per_class=n_samples_per_class, class_col=self._class_col, pop_col=self._pop_col, max_samples=max_samples)

	def fix_sampling_locations(self, t_lat, t_lon,old_pop):
		return fix_sampling_locations(self._gdf, t_lat = t_lat, t_lon= t_lon, index=old_pop)

def remove_junk_str(input):
    out = re.sub(' +','_', re.sub(r'([^\s\w]|_)+', '', input.lower()))
    return out

def load_shapefile(shapefile, class_col=CLASS_COL):
	# read in shapefile
	try:
		gdf = gpd.GeoDataFrame.from_file(shapefile)
	except Exception as e:
		print('error is')
		print(e)
		print "--> %s: error reading file!"%shapefile
		return None

	gdf.columns = [c.upper() if c != "geometry" else c for c in gdf.columns]
	if AREA_COL not in gdf.columns:
		gdf[AREA_COL] = gdf['geometry'].apply(lambda p: p.area)
	if LEN_COL not in gdf.columns:
		gdf[LEN_COL] = gdf['geometry'].apply(lambda p: p.length)
		
	# convert area & length to km
	gdf[AREA_COL] = gdf[AREA_COL] / 1.0e6 # convert to km^2
	gdf[LEN_COL]  = gdf[LEN_COL] / 1.0e3 # convert to km
	
	# change coordinate system from northing/easting to lonlat
	targetcrs = {u'ellps': u'WGS84', u'datum': u'WGS84', u'proj': u'longlat'}
	gdf.to_crs(crs=targetcrs, epsg = 4326, inplace=True)

	return gdf

def construct_class_raster(gdf, bbox, all_classes,class_col=CLASS_COL,pop_col= POP_COL, label2class=None, grid_size=(100,100)):
	grid_size_lon, grid_size_lat = grid_size
	lonmin_grid, latmin_grid, lonmax_grid, latmax_grid = bbox
	latv = np.linspace(latmin_grid, latmax_grid, grid_size_lat+1)
	lonv = np.linspace(lonmin_grid, lonmax_grid, grid_size_lon+1)

	label2class = {i: c for i, c in enumerate(all_classes)}
	class2label = {c: i for i, c in enumerate(all_classes)}

	raster = np.zeros((grid_size_lon, grid_size_lat, len(all_classes)))
	locations = []
	for i in range(len(lonv)-1):
		for j in range(len(latv)-1):
			cell_poly = Polygon([(lonv[i],latv[j]), (lonv[i+1],latv[j]), \
								 (lonv[i+1],latv[j+1]), (lonv[i],latv[j+1])])

			gdf_frame = vu.filter_gdf_by_polygon(gdf, cell_poly)

			if len(gdf_frame) == 0:
				continue # exit current loop but continue to next loop
			areas_per_class = gdf_frame.groupby(class_col)\
								.apply(lambda x: x.intersection(cell_poly)\
									   .apply(lambda y:y.area*(6400**2)).sum())
			# test_full_per_class = gdf_frame.groupby(class_col).apply(lambda y: y.area * (6400 ** 2)).sum()
			classified_area = areas_per_class.sum()
			population = gdf_frame[POP_COL].sum()
			if classified_area > 0:
				areas_per_class = areas_per_class / float(classified_area)
				class_percents = [0] * len(all_classes)
				for class_name, value in areas_per_class.iteritems():
					class_name_cleaned = remove_junk_str(class_name)
					label = class2label[class_name_cleaned]
					class_percents[label] = value
				areas_per_class = areas_per_class / float(classified_area)

				raster[i,j,:] = [areas_per_class[label2class[k]] if label2class[k] in areas_per_class else 0 for k in range(len(all_classes))]

				li = [i, j, cell_poly.centroid.xy[0][0], cell_poly.centroid.xy[1][0]]
				li.append(class_percents)
				li.append(population)
				loc = tuple(li)
				locations.append(loc)
			else:
				print('classified area was zero')
	
	locations = pd.DataFrame(locations, columns=["grid-i", "grid-j", "lon", "lat", "class",'pop'])
	return raster, locations

def generate_sampling_locations(gdf_sel,all_classes, n_samples_per_class=N_SAMPLES_PER_CLASS,class_col=CLASS_COL,pop_col= POP_COL, max_samples=MAX_SAMPLES_PER_POLY):
	out_of_shape = 0
	counter = 0
	# select polygons to sample
	select_polygons = gdf_sel.groupby(class_col)\
						.apply(lambda x: sample_polygons(x, 
									n_samples=n_samples_per_class, 
									max_samples=max_samples))

	if class_col not in select_polygons.columns:
		select_polygons.reset_index(inplace=True)

	# make sure all polygons are ok
	select_polygons['geometry'] = select_polygons['geometry'].apply(lambda p: p.buffer(0) if not p.is_valid else p)
	
	# sample locations from each polygon
	locations = select_polygons.groupby(class_col)\
				.apply(lambda x: sample_locations_from_polygon(x,
					sample_on_boundary='road' in x[class_col].iloc[0].lower() or 'railway' in x[class_col].iloc[0].lower()))
	t_lat = locations['lat']
	t_lon = locations['lon']


	class2label = {c: i for i, c in enumerate(all_classes)}

	#calculated from 	https://gis.stackexchange.com/questions/115237/how-to-calculate-degrees-per-pixel-in-a-given-image-or-nth-of-degree
	# estimated image length is 170m -> 545.428 ft
	# img width is 224 pixels
	# ft/pixel = 545.428/224 = 2.434946429
		# decimal degree per pixel = 2.434946429/  363699 = 0,000006695
	#degree delta per whole image = 0.000006695 * 224 = 0.001499669

	out_locations = []
	deg_delt = 0.001499669
	for i in range(len(t_lat)):
		counter +=1

		lon = t_lon[i] - deg_delt / 2  # set the starting location to the bottom left corner
		lat = t_lat[i] - deg_delt / 2  # set the starting location to the bottom left corner
		cell_poly = Polygon([(lon, lat), (lon+deg_delt, lat),(lon+deg_delt, lat+deg_delt), (lon, lat+deg_delt)])
		gdf_window = vu.filter_gdf_by_polygon(gdf_sel, cell_poly)
		try:
			#intersection returns another polygon
			#https://gis.stackexchange.com/questions/230494/intersecting-two-shape-problem-using-geopandas
			areas_per_class = gdf_window.groupby(class_col) \
				.apply(lambda x: x.intersection(cell_poly) \
					   .apply(lambda y: y.area * (6400 ** 2) ).sum())
		except:
			out_of_shape +=1
			continue
		classified_area = areas_per_class.sum()

		percent_per_class= areas_per_class/classified_area

		gdf_w_area = gdf_window[AREA_COL]
		gdf_w_pop = gdf_window[POP_COL]
		gdf_pop_unit = gdf_w_pop/gdf_w_area
		gdf_window[POP_COL] = gdf_pop_unit
		poly_groups = gdf_window.groupby('IDENT')
		area_per_poly = poly_groups  \
			.apply(lambda x: x.intersection(cell_poly) \
				   .apply(lambda y: y.area * (6400 ** 2)).sum())

		total_classified_area= area_per_poly.sum()
		area_percent_per_polygon_l = list(area_per_poly/total_classified_area)

		pop_col_l = list(poly_groups[POP_COL].sum()) #sum just converts it to pandas series
		scaled_pop = []
		# box_area = ((deg_delt) / 1.0e3) **2
		area_pop = 0
		for i in range(len(pop_col_l)):
			pop_of_polygon = pop_col_l[i]
			percent_of_polygon = area_percent_per_polygon_l[i]
			accurate_pop = pop_of_polygon*percent_of_polygon
			scaled_pop.append(accurate_pop)
			area_pop += accurate_pop

		if type(classified_area) == np.float64: #ensures
			if classified_area > 0:
				class_percents = [0] * len(all_classes)
				for class_name, value in percent_per_class.iteritems():
					class_name_cleaned = remove_junk_str(class_name)
					#weird coditions because some landtypes were slightly off
					if class_name_cleaned == 'open_spaces_with_little_or_no_vegetations_beaches_dunes_bare_rocks_glaciers':
						continue
					if class_name_cleaned == 'water_bodies':
						class_name_cleaned = 'water'
					if class_name_cleaned == 'wetland':
						class_name_cleaned = 'wetlands'


					label = class2label[class_name_cleaned]
					class_percents[label] = value

				one_data_point = [lon + deg_delt/2, lat+ deg_delt/2]
				one_data_point.append(class_percents)
				one_data_point.append(area_pop)
				out_locations.append(tuple(one_data_point))
			else:
				print('classified area was zero')
		else:
			out_of_shape +=1
	print('there were %d points out of the shapefile'%out_of_shape)
	new_locations_pd = pd.DataFrame(out_locations, columns=["lon", "lat", "class", 'pop'])
	return new_locations_pd

def fix_sampling_locations(gdf_sel, t_lat, t_lon, index, class_col=CLASS_COL, pop_col=POP_COL):
	deg_delt = 0.001499669

	lon = t_lon - deg_delt / 2  # set the starting location to the bottom left corner
	lat = t_lat - deg_delt / 2  # set the starting location to the bottom left corner
	cell_poly = Polygon(
		[(lon, lat), (lon + deg_delt, lat), (lon + deg_delt, lat + deg_delt), (lon, lat + deg_delt)])

	gdf_window = vu.filter_gdf_by_polygon(gdf_sel, cell_poly)

	# areas_per_class = gdf_window.groupby(class_col) \
	# 	.apply(lambda x: x.intersection(cell_poly) \
	# 		   .apply(lambda y: y.area * (6400 ** 2)).sum())

	gdf_w_area = gdf_window[AREA_COL]
	gdf_w_pop = gdf_window[POP_COL]
	gdf_pop_unit = gdf_w_pop / gdf_w_area
	gdf_window[POP_COL] = gdf_pop_unit

	area_per_poly = gdf_window.groupby('IDENT') \
		.apply(lambda x: x.intersection(cell_poly) \
			   .apply(lambda y: y.area * (6400 ** 2)).sum())
	# area_per_poly_l = list(area_per_poly)
	total_classified_area = area_per_poly.sum()
	print('total_classified_area')
	print(total_classified_area)

	percent_per_polygon = area_per_poly / total_classified_area
	percent_per_polygon_l = list(percent_per_polygon)

	gdf_by_ident = gdf_window.groupby('IDENT')
	pop_col = gdf_by_ident[POP_COL].sum()  # sum just converts it to pandas series
	pop_col_l = list(pop_col)

	scaled_pop = []

	area_pop = 0
	for i in range(len(pop_col_l)):
		pop_of_polygon = pop_col_l[i]
		PPP = percent_per_polygon_l[i]
		accurate_pop = pop_of_polygon * PPP
		# this part is messed up,  what i should do is somehow grab the correct item.  the sizes might vary here
		pop_per_area_in_window = accurate_pop
		scaled_pop.append(pop_per_area_in_window)
		area_pop += pop_per_area_in_window
	return area_pop


AL = remove_junk_str('Arable land (annual crops)')
FR = remove_junk_str('Forests')

#function only sees one polygon type at a time
def sample_polygons(df, n_samples=500, max_samples=None):
	global AL
	global FR
	if AL in set(df['ITEM2012']) or FR in set(df['ITEM2012']):
		max_samples = 5
	samples_per_poly = (df.Shape_Area / float(df.Shape_Area.min())).astype(int)
	# samples_per_poly = 1+ 4* np.log((df.Shape_Area / float(df.Shape_Area.min()))).astype(int)

	# Bump up the samples per poly for the smaller classes that we were seeing only a few of.
	# It is just a quick fix for our project needs
	if 'Airports' in set(df['ITEM2012']) \
			or 'Railways and associated land' in set(df['ITEM2012']) \
			or 'Wetlands' in set(df['ITEM2012']) \
			or 'Continuous urban fabric (S.L. : > 80%)' in set(df['ITEM2012']) :
		samples_per_poly = samples_per_poly * 15
		n_samples == samples_per_poly +1

	if 'Other roads and associated land' in set(df['ITEM2012']):
		samples_per_poly = samples_per_poly * 2
		n_samples == samples_per_poly + 1

	if samples_per_poly.sum() > n_samples:
		pvec = np.array([0.0, 0.2, 0.5, 0.7, 0.9, 0.95, 1])
		bins = np.percentile(samples_per_poly, pvec*100)
		cnts, _ = np.histogram(samples_per_poly, bins)

		ret = []
		x = samples_per_poly
		for i in range(len(bins)-1):
			if cnts[i] == 0:
				continue
			y = x[(x>=bins[i]) & (x<bins[i+1])] if i<len(bins)-2 \
					else x[(x>=bins[i]) & (x<=bins[i+1])]
			y = y.sample(frac=pvec[i+1],random_state=1)
			ret.append(y)
		ret = pd.concat(ret)
		ret_scaled = (ret.astype(float) / ret.sum() * n_samples)\
						.apply(np.ceil).astype(int)
		ret_df = df.ix[ret_scaled.index]
		ret_df['samples'] = ret_scaled.values
	else:
		ret_df = df
		ret_df['samples'] = samples_per_poly.values

	if max_samples is not None:
		ret_df['samples'] = ret_df['samples'].apply(\
									lambda x: min([x, max_samples]))
	ret_df['samples'] = ret_df['samples'].astype(int)
	return ret_df

#function only sees one polygon type (can't determine nearby points here
def sample_locations_from_polygon(df, sample_on_boundary=False):

	polygons = df['geometry']
	nsamples = df['samples']

	if not sample_on_boundary:
		centroids = np.array([(p.centroid.coords.xy[0][0], p.centroid.coords.xy[1][0]) \
					  for p in polygons])    
		idx = nsamples > 1
		if idx.sum()>0:
			polygons = polygons[idx]
			nsamples = nsamples[idx]
			locs = [satimg.generate_locations_within_polygon(p, nSamples=m - 1, strict=True) \
					for p,m in zip(polygons, nsamples)]
			locs = np.vstack(locs).squeeze()
			locs = np.vstack([locs, centroids])
		else:
			locs = centroids
	else:
		boundaries= [zip(p.exterior.coords.xy[0], p.exterior.coords.xy[1]) for p in polygons]
		np.random.seed(10)
		locs = np.array([b[l] for b,m in zip(boundaries,nsamples) \
						 for l in np.random.choice(np.arange(0,len(b)), min([len(b),m]))])
	ret = pd.DataFrame(locs, columns=["lon", "lat"])
	return ret

