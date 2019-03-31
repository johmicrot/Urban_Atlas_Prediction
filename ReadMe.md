# This guide overviews how to generate the Urban Atlas Dataset, and how to train it in both a multi-task and single-task enviornments with Keras.






### Dataset creation - add a population to UA shapefiles, generate data points, and download images


#### 1) Download specific city shapefiles from urban atlas 
	https://land.copernicus.eu/local/urban-atlas/urban-atlas-2012?tab=download
	
#### 2) Unzip and put each city folder into the directory format shown below
	~/Cities/{country name}/Shapefiles_Original/{city folder}

#### 3) Run file Modify_Country_Shapefiles.py. It will create new shapefiles with population information in the directory below 
	~/Cities/{country name}/Shapefiles_with_pop/
	
#### 4) Download AllPopulationEstimates.zip from https://goo.gl/n4w5gS and unzip into the directory below
	~/Cities/AllPopulationEstimates

#### 5) Run Country_Data_Generation.py. Generates the datapoints with targets.

#### 6) Run Country_Static_Maps_download.py to Downloads the satelite images.  We have provided a pre made dataet which can be downloaded from https://goo.gl/n4w5gS.  Unzip Urban_Atlas_Cities.7z to  
	~/Dataset/




### Training with Keras

#### 1) modify Train_Model.py to select the dataset you want to train and select if you are doing a multi-task or single-task




pysatml taken from https://github.com/adrianalbert/pysatml <br/>