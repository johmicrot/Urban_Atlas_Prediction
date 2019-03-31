# This guide overviews how to generate the Urban Atlas Dataset, and how to train it in both a multi-task and single-task enviornments with Keras.






## Dataset creation - add a population to UA shapefiles, generate data points, and download images


### 1) Download specific city shapefiles from urban atlas 
	https://land.copernicus.eu/local/urban-atlas/urban-atlas-2012?tab=download
	
### 2) Unzip and put each city folder into the directory format shown below
	~/Cities/{country name}/Shapefiles_Original/{city folder}

### 3) Run file Modify_Country_Shapefiles.py. It will create new shapefiles with population information in the directory below 
	~/Cities/{country name}/Shapefiles_with_pop/
	
### 4) Unzip AllPopulationEstimates.zip and AllPopulationEstimates_part2.zip into the directory below
	~/Cities/AllPopulationEstimates

### 5) Run Country_Data_Generation.py. Generates the datapoints with targets.

### 6) Run Country_Static_Maps_download.py.  Downloads the satelite images.



pysatml taken from https://github.com/adrianalbert/pysatml <br/>
unmodified urbanatlas taken from https://github.com/adrianalbert/urban-environments/tree/master/dataset-collection


## Training with Keras

