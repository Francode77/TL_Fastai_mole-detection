# FastAI for mole detection 

## Scope
We aim to identify malign and bening moles with a user-friendly prediction application.

## Project Description
 
We create a model from a pretrained model by re-training it with our image data. We then use a streamlit application to show predictions in a user-friendly interface.

## Prerequisites

This repo makes use of the HAM10000 dataset to train a model.

## Includes
 
### Model 

 - make_binary_model.ipynb 
 
### App
 
 - mole_detection_app.py
 
## Requirements
  
  - Python 3.7.9
  - CUDA 11.7 and CudNN  

## Installation

 - Create and activate a virtual env
 - Install the necessary libraries from the requirements.txt file in the main folder 
`
## Usage 

1) To create a model, run all the files in the notebook 
2) To launch the app, type in the base folder

`streamlit run mole_detection_app.py`
<br><br>
A browser window will open which launches the app

## Method

- We will use the FastAI API to preprocess the data and build our model.
- We use the full HAM_10000 dataset containing 10057 images for training and validation of which 15% is reserved for the validation set

### Preprocessing : 
#### Upsampling
- Visual augmentation<br>
  We create 8 variations of any image (8 dihedral transformations) and add this to the dataset to increase the diversity of the image set
- Data augmentation<br>
  We upsample with random crop and random resize 
  
### Training:

- We make use of the xceptionnet pretrained model
- We determine a good learning rate by choosing the valley in the learning rate curve
- We unfreeze the weights of the model and train it on the HAM10000 dataset in 24 epochs

### Validation
- We validate the model by plotting the confusion matrix and the loss on both training and validation set.

# Results

The model shows the following results
| Metric | Outcome |
| :--------- | ----------: |
| True Positives | 12.23% |
| False Positives | 3.89% |
| True Negatives | 76.78% |
| False Negatives | 7.09% |
|   |  |
| **Precision** | **75.8%** |
| **Recall** | **63.3%** | 

# Improvements
 
We worked on data preprocessing methods such as contrast, rebalancing and hair removal, which are not included nor used for the scope of this project.<br>


## Contributors 
- [Frank Trioen](https://github.com/Francode77) 

## Acknowledgements

 - [BeCode](https://becode.org/) coaches (Louis & Chrysanthi) 
