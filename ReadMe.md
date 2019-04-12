
## Predicting Societal Indicators from Space
### Contributors: "Daniel Pototzky, and John Rothman"

This project makes three major contributions:
1. Create a high-quality dataset by matching satellite images from Google Maps Static API with auxiliary data from Urban Atlas. Every image has two labels, one for the land type, another for population count. Overall, the dataset contains 225k images from 183 European cities.

2. Explore the potential of Multi-task architectures in this domain. It is shown that Multi-task learning is particularly beneficial if the number of labeled images is relatively small.

3. Apply Data Distillation to make use of unlabeled images. Thereby, two new ideas are suggested: First, only images are included for which the Data Distillation Ensemble is sure about, i.e. the prediction certainty is above a certain threshold. Second, the class distribution of the training set is taken into account which improves performance on small classes. Overall, it is shown that Data Distillation leads to an increase in accuracy on the test set by extracting information from unlabeled images.


### Dataset creation
    This can be skipped by downloading our pre made dataset, which can be downloaded from https://goo.gl/n4w5gS. 
    Make sure to  Unzip Urban_Atlas_Cities.7z to ~/Dataset/

 1) Download specific city shapefiles from urban atlas 
	https://land.copernicus.eu/local/urban-atlas/urban-atlas-2012?tab=download
	
2) Unzip and put each city folder into the directory format shown below
	~/Cities/{country name}/Shapefiles_Original/{city folder}

 3) Run file Modify_Shapefiles.py. It will create new shapefiles with population information in the directory below 
	~/Cities/{country name}/Shapefiles_with_pop/
	
4) Download AllPopulationEstimates.zip from https://goo.gl/n4w5gS and unzip into the directory below
	~/Cities/AllPopulationEstimates

 5) Execute Dataset_points_Generation.py. Generates the datapoints with targets.

6) Execute Static_Maps_download.py to Downloads the satelite images. Will save to the directory ~/Dataset/

Here are some examples of a few classes.<br><br>

Airport

<div class="column">
  <img src=Dataset/Germany/images/DE011L1_DUSSELDORF/51.284446_6.771030.jpg width="200" height="200">
  <img src=Dataset/Germany/images/DE011L1_DUSSELDORF/51.276106_6.751596.jpg width="200" height="200">
  <img src=Dataset/Germany/images/DE011L1_DUSSELDORF/51.285025_6.748615.jpg width="200" height="200">
</div>


<br>
Water
<div class="column">
  <img src=Dataset/Germany/images/DE011L1_DUSSELDORF/51.229189_6.730385.jpg width="200" height="200">
  <img src=Dataset/Germany/images/DE011L1_DUSSELDORF/51.202616_6.730954.jpg width="200" height="200">
</div>

<br>
Forests
<div class="column">
  <img src=Dataset/Germany/images/DE011L1_DUSSELDORF/51.105523_6.738932.jpg width="200" height="200">
  <img src=Dataset/Germany/images/DE011L1_DUSSELDORF/51.235477_7.034913.jpg width="200" height="200">
  <img src=Dataset/Netherlands/images/NL020L1_ROOSENDAAL/51.478826_4.389469.jpg width="200" height="200">
</div>
<br>
Arable Land (Annual Crops)
<div class="column">
  <img src=Dataset/Netherlands/images/NL020L1_ROOSENDAAL/51.495747_4.443052.jpg width="200" height="200">
  <img src=Dataset/Netherlands/images/NL020L1_ROOSENDAAL/51.477002_4.370888.jpg width="200" height="200">
  <img src=Dataset/Netherlands/images/NL020L1_ROOSENDAAL/51.566406_4.458441.jpg width="200" height="200">
</div>



### Explore the potential of Multi-task architectures in this domain

1) You can run both single-task and multi-task settings using Train_Model.py. Make sure to select the part of the dataset you want to train on. Also make sure to specify if you want to train a single-task or multi-task setup.

### Apply Data Distillation to make use of unlabeled images

 1) Train a base model on a labeled subpart of the data using Train_Model.py as described in the previous section
 2) Execute Generate_Ensamble_Predictions.py.  It generates Ensemble predictions for unlabeled images given the model that was trained on a small labeled part of the data.
 3) Generate labels for unlabeled images by running Create_new_labels.py. You can specify if you want to apply 'naive' Data Distillation, apply a threshold for inclusion of images, or combine a threshold with information about class distribution from the training set.
 4) Rerun Train_Model.py using the small number of labeled images as well as images 'labeled' by the Data Distillation ensemble. 


pysatml taken from https://github.com/adrianalbert/pysatml <br/>
