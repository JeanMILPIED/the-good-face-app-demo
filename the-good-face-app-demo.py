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

st.title('The-good-face-app 📷😎 (demo)')
st.header("Make a Linkedin portrait from your best shots")
st.subheader('Our AI will rate it and propose a better rated one')

portrait_img=None

source = st.radio(
    "Where is your portrait picture source ?",
    ('File','Webcam'))


if source=='File':
    portrait_img=st.file_uploader('Chose a portrait picture')
    col1, col2 = st.columns(2)
    if portrait_img!=None:
        bytes_data = portrait_img.getvalue()
        # Show the image filename and image.
        with open('portrait.jpg', "wb") as f:
            f.write(bytes_data)
        col1.write('your initial portrait')
        col1.image(bytes_data)

if source=='Webcam':
    portrait_img = st.camera_input('Take a new webcam picture')
    col1, col2 = st.columns(2)
    if portrait_img!=None:
        bytes_data = portrait_img.getvalue()
        # Show the image filename and image.
        with open('portrait.jpg', "wb") as f:
            f.write(bytes_data)
        col1.write('Your initial portrait')
        col1.image(bytes_data)



if portrait_img !=None:
    try:
        total_proba_initial, total_proba_final = auto_portraitImage_optimisation('portrait')
        col2.write('Your best portrait with quality probability')
        col2.image('portrait_ALLcorrected.jpg')

        #write probas
        col1.write('We rate it at {}'.format(total_proba_initial))
        col2.write('Our best work is now rated at {}'.format(total_proba_final))
    except:
        st.write("Oups! Something went wrong")




