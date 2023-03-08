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

st.title('The-good-face 📷😎 (app demo)')
st.header("Make a Lkdn portrait from your best shots")
st.subheader('Our AI rates your portrait and proposes a better one')

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
        col1.subheader('Your initial portrait')
        col1.image(bytes_data)

if source=='Webcam':
    portrait_img = st.camera_input('Take a new webcam picture')
    col1, col2 = st.columns(2)
    if portrait_img!=None:
        bytes_data = portrait_img.getvalue()
        # Show the image filename and image.
        with open('portrait.jpg', "wb") as f:
            f.write(bytes_data)
        col1.subheader('Your initial portrait')
        col1.image(bytes_data)



if portrait_img !=None:

    total_proba_initial, total_proba_final = auto_portraitImage_optimisation('portrait')
    #write probas
    col1.write('Portrait is rated at {}'.format(total_proba_initial))
    if total_proba_initial > 0.5:
        col1.write("Your pic is good as it is 👍")
    else:
        col1.write("Your pic is bad 😌")
    col2.subheader('AI best portrait 🤩')
    col2.image('portrait_ALLcorrected.jpg')
    col2.write('Our portrait is rated at {}'.format(total_proba_final))
    if total_proba_final > 0.5:
        col2.write("You can use this AI pic on Lkdn 👍")
    else:
        col2.write("We did not succeed in optimizing it 😱")
    # except:
    #      st.write("Oups! Something went wrong")




