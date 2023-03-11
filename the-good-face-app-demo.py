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
st.text('Our smart AI is capable to rate your portrait and then to propose a better one.'
        '\nIt has been trained on thousands of portrait pictures from all ages, genders \n'
        'and ethnics to avoid any bias.'
        '\nEnjoy! 🔥')

st.subheader('Step1 - Rate your current profile pic')
portrait_img=st.file_uploader('.jpg, .jpeg, .tiff format accepted')
col1, col2 = st.columns(2)

if portrait_img!=None:
    bytes_data = portrait_img.getvalue()
    # Show the image filename and image.
    with open('portrait.jpg', "wb") as f:
        f.write(bytes_data)
    col1.subheader('Your initial portrait')
    col1.image(bytes_data)
    total_proba_initial, total_proba_final = auto_portraitImage_optimisation('portrait')
    #write probas
    col1.write('AI rate {}  (0: bad - 1: great)'.format(total_proba_initial))
    if total_proba_initial > 0.7:
        col1.write("Your pic is good as it is 👍")
    else:
        col1.write("Your pic is bad 😌")
    col2.subheader('AI best portrait 🤩')
    col2.write('Sorry if not excellent, we are not wizzards yet 🎇')
    col2.image('portrait_ALLcorrected.jpg')
    col2.write('AI rate {}  (0: bad - 1: great)'.format(total_proba_final))
    if total_proba_final > 0.7:
        col2.write("You could use this new AI pic on Lkdn 👍")
    else:
        col2.write("We did not succeed in optimizing it 😱")
    portrait_img=None

st.subheader('Step2 - Take a new one with Webcam !')
st.subheader('Some (professional) tips')
st.text('1. Dress professionally \n2. Ensure light is bright \n3. Have a uniform background \n4. Look at camera and smile')
st.subheader("Let's take a nice webcam pic")
new_portrait_img = st.camera_input('')
col1, col2 = st.columns(2)

if new_portrait_img!=None:
    bytes_data = new_portrait_img.getvalue()
    # Show the image filename and image.
    with open('new_portrait.jpg', "wb") as f:
        f.write(bytes_data)
    col1.subheader('Your new portrait')
    col1.image(bytes_data)

    total_proba_initial, total_proba_final = auto_portraitImage_optimisation('new_portrait')
    # write probas
    col1.write('AI rate {}  (0: bad - 1: great)'.format(total_proba_initial))
    if total_proba_initial > 0.7:
        col1.write("Your pic is good as it is 👍")
    else:
        col1.write("Your pic is bad 😌")
    col2.subheader('AI best portrait 🤩')
    col2.write('Sorry if not excellent, we are not wizzards yet 🎇')
    col2.image('new_portrait_ALLcorrected.jpg')
    col2.write('AI rate {}  (0: bad - 1: great)'.format(total_proba_final))
    if total_proba_final > 0.7:
        col2.write("You could use this new AI pic on Lkdn 👍")
    else:
        col2.write("We did not succeed in optimizing it 😱")

    # except:
    #      st.write("Oups! Something went wrong")




