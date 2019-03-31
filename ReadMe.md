
## Predicting Societal Indicators from Space
### Contributors: "Daniel Pototzky, and John Rothman"

This project makes three major contributions:
1. Create a high-quality dataset by matching satellite images from Google Maps Static API with auxiliary data from Urban Atlas. Every image has two labels, one for the land type, another for population count. Overall, the dataset contains 225k images from 183 European cities.

2. Explore the potential of Multi-task architectures in this domain. It is shown that Multi-task learning is particularly beneficial if the number of labeled images is relatively small.

3. Apply Data Distillation to make use of unlabeled images. Thereby, two new ideas are suggested: First, only images are included for which the Data Distillation Ensemble is sure about, i.e. the prediction certainty is above a certain threshold. Second, the class distribution of the training set is taken into account which improves performance on small classes. Overall, it is shown that Data Distillation leads to an increase in accuracy on the test set by extracting information from unlabeled images.


### Dataset creation - add a population to UA shapefiles, generate data points, and download images
    This can be skipped by downloading our pre made dataet, which can be downloaded from https://goo.gl/n4w5gS. 
    Make sure to  Unzip Urban_Atlas_Cities.7z to ~/Dataset/

#### 1) Download specific city shapefiles from urban atlas 
	https://land.copernicus.eu/local/urban-atlas/urban-atlas-2012?tab=download
	
#### 2) Unzip and put each city folder into the directory format shown below
	~/Cities/{country name}/Shapefiles_Original/{city folder}

#### 3) Run file Modify_Country_Shapefiles.py. It will create new shapefiles with population information in the directory below 
	~/Cities/{country name}/Shapefiles_with_pop/
	
#### 4) Download AllPopulationEstimates.zip from https://goo.gl/n4w5gS and unzip into the directory below
	~/Cities/AllPopulationEstimates

#### 5) Execute Country_Data_Generation.py. Generates the datapoints with targets.

#### 6) Execute Country_Static_Maps_download.py to Downloads the satelite images. Will save to the directory ~/Dataset/


### Exploring the potential of Multi-task architectures in this domain

#### 1) modify Train_Model.py to select the dataset you want to train and select if you are doing a multi-task or single-task


### Apply Data Distillation to make use of unlabeled images

#### 1) Execute Generate_Ensamble_Predictions.py.  It creates "labels" for unlabeled images given a model trained on a small labeled part of the data.


pysatml taken from https://github.com/adrianalbert/pysatml <br/>
