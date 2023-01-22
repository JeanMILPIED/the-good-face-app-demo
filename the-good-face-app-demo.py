import streamlit as st
from utils_theGoodFace import *
from PIL import Image
from io import BytesIO

def portraitImage_optimisation(image_name):

    total_proba_initial, total_proba_final = auto_portraitImage_optimisation(image_name)

    if total_proba_initial != 0 and total_proba_final != 0:
        img_opti = Image.open(image_name + '_ALLcorrected.jpg', mode='r')
        return img_opti
    else:
        return 'error'

st.title('The-good-face-app')
st.header('demo code')
portrait_img=None

col1, col2=st.columns(2)
source = st.radio(
    "Where is your portrait picture source ?",
    ('File','Webcam'))

if source=='File':
    portrait_img=st.file_uploader('Chose a portrait picture')
    if portrait_img!=None:
        bytes_data = portrait_img.getvalue()
        # Show the image filename and image.
        st.write(f'filename: {portrait_img.name}')
        st.image(bytes_data)
        with open('portrait.jpg', "wb") as f:
            f.write(bytes_data)

if source=='Webcam':
    portrait_img = st.camera_input('Take a new webcam picture')
    if portrait_img!=None:
        bytes_data = portrait_img.getvalue()
        # Show the image filename and image.
        st.write(f'filename: {portrait_img.name}')
        st.image(bytes_data)
        with open('portrait.jpg', "wb") as f:
            f.write(bytes_data)

if portrait_img !=None:
    total_proba_initial, total_proba_final = auto_portraitImage_optimisation('portrait')
    st.header('YOUR best portrait with quality probability')
    st.image('portrait_ALLcorrected.jpg')




