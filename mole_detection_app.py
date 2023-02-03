from fastai.data.all import *
from fastai.vision.all import *
from fastai.metrics import *
from fastai.vision.widgets import show_image

import pandas as pd 
import os
import streamlit as st
import random
 
# Load a stylesheet
from load_css import local_css
local_css("style.css")

## STREAMLIT BUTTONS ##
# Create the selection buttons 
st.sidebar.title("Check me") 
if st.sidebar.button(label='Random_image'):
    buttoninput='all'
if st.sidebar.button(label='No cancer'):
    buttoninput='no_cancer'
if st.sidebar.button(label='Skin cancer'):
    buttoninput='cancer'

with st.empty():
    # Check if button is pressed on last run
    try:
        buttoninput
        st.text('')
    except NameError:
        buttoninput='all' 
 
## RANDOM IMAGE FUNCTION ## 
# Returns an image from labeled directories
 
def chooseRandomImage(buttoninput):
 
    # Valid extensions
    imgExtension = ["png", "jpeg", "jpg"] 
    allImages = list()
    directories={   "all":"./data/HAM10000_images/",    # All images
                    "no_cancer":"./data/tf_data/0",     # No cancer images
                    "cancer":"./data/tf_data/1"         # Cancer images
                    } 
    directory=directories[buttoninput]

    # Get a list of files from button chosen directory
    for img in os.listdir(directory): 
        ext = img.split(".")[len(img.split(".")) - 1]
        if (ext in imgExtension):
            allImages.append(img)
    
    # Choose a random image from this list
    choice = random.randint(0, len(allImages) - 1)
    chosenImage = allImages[choice] 

    return chosenImage
 
# Make binary label dictionary
@st.cache(show_spinner=False)
def make_dict(df):

    global img_to_label_dict

    img_to_label_dict = df[["image_id", "label"]].to_dict(orient='list')  
    img_to_label_dict = {img_id : label for img_id,label in zip(img_to_label_dict['image_id'], img_to_label_dict['label']) }

    return img_to_label_dict 

# Get binary label of a file
def get_cnc_label_from_dict(path):
    return img_to_label_dict[path.stem]

# Load a dataframe from metadata
@st.cache(show_spinner=False)
class acquire_df:
    def __init__(self) -> None: 
        self.csv_path = "./data/HAM10000_metadata.csv"
        skin_df = pd.read_csv(self.csv_path)

        # Create binary classification labels
        skin_df['label']=0
        skin_df.loc[skin_df['dx'].isin(['akiec','bcc','mel']),'label']=int(1)
        
        # Make the dataframe callable
        self.df=skin_df 
        self.img_to_label_dict=make_dict(self.df) 

# Create readable labels from df.labels for label text
short_to_full_name_dict = {
    "akiec" : "Bowen's disease", # very early form of skin cancer 
    "bcc" : "basal cell carcinoma" , # basal-cell cancer or white skin cancer
    "bkl" : "benign keratosis-like lesions", # non-cancerous skin tumour
    "df" : "dermatofibroma", # non-cancerous rounded bumps 
    "mel" : "melanoma", # black skin cancer
    "nv" : "melanocytic nevi", # mole non-cancerous
    "vasc" : "vascular lesions", # skin condition
}
# Create explanation from df.labels for validation text
short_to_full_name_dict_explained = {
    "akiec" : "Bowen's disease is a very early form of skin cancer",
    "bcc" : "basal cell carcinoma is also called white skin cancer",
    "bkl" : "benign keratosis-like lesions is a non-cancerous skin tumour",
    "df" : "dermatofibroma is a non-cancerous rounded bumps", 
    "mel" : "melanoma is black skin cancer",
    "nv" : "melanocytic nevi is a non-cancerous mole",
    "vasc" : "vascular lesions is a skin condition"
}

# Create the dataframe
df=acquire_df()

# Multiclass label dictionary
img_to_class_dict = df.df.loc[:, ["image_id", "dx"]] 
img_to_class_dict_explained = df.df.loc[:, ["image_id", "dx"]] 
img_to_class_dict = img_to_class_dict.to_dict('list')  
img_to_class_dict_explained = img_to_class_dict_explained.to_dict('list')  

img_to_class_dict = {img_id : short_to_full_name_dict[disease] for img_id,disease in zip(img_to_class_dict['image_id'], img_to_class_dict['dx']) } 
img_to_class_dict_explained = {img_id : short_to_full_name_dict_explained[disease] for img_id,disease in zip(img_to_class_dict_explained['image_id'], img_to_class_dict_explained['dx']) } 
 

## IMAGE SELECTION ##
# Select a random image from button inputs
image=chooseRandomImage(buttoninput=buttoninput)
image_url=f'./data/HAM10000_images/{image}'

## STREAMLIT WEBPAGE ##
st.title("What the mole is that!?")
st.image(image_url, width=224)

# PREDICTION OUTPUT ##
with st.spinner('Predicting..'): 
    # Load model for binary classification  
    learn_binary     = load_learner(os.path.join('models/unfreezed_xception','xception_cnc_binarylabels.pkl'))

    # Make a prediction from the model and prepare the text output
    binary_prediction     = learn_binary.predict(image_url)[0]
    if binary_prediction=='1':
        binary_prediction_msg="<span class='highlight red'><span class='bold'>Skin cancer</span></span>"
    elif binary_prediction=='0':
        binary_prediction_msg="<span class='highlight green'><span class='bold'>Safe</span></span>"

    #st.header('Predicted class')
    st.markdown('**'+binary_prediction_msg+'**',unsafe_allow_html=True) 
    st.markdown('Labeled as: **'+img_to_class_dict[image[:-4]]+'**')

    # For binary classification validation
    cancer_list=["Bowen's disease","basal cell carcinoma","melanoma"]
    with st.spinner('Validating..'): 
        if binary_prediction=='0' and img_to_class_dict[image[:-4]] in cancer_list :
            st.markdown("### <span class='highlight red'><span class='bold'>False Negative!</span></span>",unsafe_allow_html=True)
        elif binary_prediction=='1' and img_to_class_dict[image[:-4]] not in cancer_list :
            st.markdown("### <span class='highlight green'><span class='bold'>False Positive!</span></span>",unsafe_allow_html=True)
        else:
            st.markdown('### ✔️ Correct!')

        st.markdown("<I>"+img_to_class_dict_explained[image[:-4]]+"</I>",unsafe_allow_html=True)
 