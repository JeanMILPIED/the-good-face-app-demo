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
st.write('Our smart AI is capable to rate your portrait and then to propose a better one.'
        '\nIt has been trained on thousands of portrait pictures from all ages, genders \n'
        'and ethnics to avoid any bias.'
        '\nEnjoy! 🔥')

with st.expander("The-Good-Face Core Values"):
    st.write('''
    1. We do not save any of your pictures 
    2. Our AI is trained on unbiased and public portrait dataset: in gender, ethnics and age
    3. We don't store your data
    4. We know we are not the best in the world, but our tool is SAFE
    5. We open source our code here: https://github.com/JeanMILPIED/the-good-face-app-demo
    ''')


st.subheader('Step1 - Rate your current profile pic')

portrait_img=st.file_uploader('.jpg, .jpeg, .jfif format accepted')

if portrait_img!=None:
    bytes_data = portrait_img.getvalue()
    # Show the image filename and image.
    with open('portrait.jpg', "wb") as f:
        f.write(bytes_data)
    st.subheader('Your initial portrait')
    col1, col2= st.columns(2)
    initial_feat_dict, total_proba_initial = portraitImage_score('portrait', my_folder=depotdir)
    col1.image(bytes_data)
    if initial_feat_dict["type"]!=1:

        #write probas
        col1.write('AI rate {}  (0: bad - 1: great)'.format(total_proba_initial))
        col1.write(message_proba(total_proba_initial))

        #write basic features extraction of the image
        col2.write(features_message(initial_feat_dict))

        st.subheader('Step2 - Launch our AI to optimize it 🤖')
        col_choice = st.radio("Choose Optimized background color", ('pink', 'blue', 'white', 'green'), horizontal=True)
        if col_choice == 'pink':
            bckgd_col = (195, 195, 255)
        elif col_choice == 'blue':
            bckgd_col = (255, 237, 159)
        elif col_choice == 'white':
            bckgd_col = (255, 255, 255)
        elif col_choice == 'green':
            bckgd_col = (154, 255, 167)
        else:
            st.write("Missing colour choice")
        val = st.button('Launch our AI 🚀')
        if val:
            final_feat_dict, total_proba_final = auto_portraitImage_optimisation('portrait',bckgd_col,initial_feat_dict)

            st.subheader('AI best portrait 🤩')
            st.write('Sorry if not excellent, we are not wizzards yet 🎇')
            col1, col2=st.columns(2)
            col1.image('portrait_ALLcorrected.jpg')
            col1.write('AI rate {}  (0: bad - 1: great)'.format(total_proba_final))
            col1.write(message_proba(total_proba_final))

            # write basic features extraction of the image
            col2.write(features_message(final_feat_dict))
            portrait_img=None
    else:
        st.write('OOOOUPS ! There is no portrait face detected in this image. Load a new one 🚀')


# st.subheader('Step3 - Take a new one with Webcam !')
# st.subheader('Some (professional) tips')
# st.text('1. Dress professionally \n2. Ensure light is bright \n3. Have a uniform background \n4. Look at camera and smile')
# st.subheader("Let's take a nice webcam pic")
# new_portrait_img = st.camera_input('')
# col1, col2 = st.columns(2)
#
# if new_portrait_img!=None:
#     bytes_data = new_portrait_img.getvalue()
#     # Show the image filename and image.
#     with open('new_portrait.jpg', "wb") as f:
#         f.write(bytes_data)
#     col1.subheader('Your new portrait')
#     col1.image(bytes_data)
#
#     total_proba_initial, total_proba_final = auto_portraitImage_optimisation('new_portrait')
#     # write probas
#     col1.write('AI rate {}  (0: bad - 1: great)'.format(total_proba_initial))
#     if total_proba_initial > 0.7:
#         col1.write("Your pic is good as it is 👍")
#     else:
#         col1.write("Your pic is bad 😌")
#     col2.subheader('AI best portrait 🤩')
#     col2.write('Sorry if not excellent, we are not wizzards yet 🎇')
#     col2.image('new_portrait_ALLcorrected.jpg')
#     col2.write('AI rate {}  (0: bad - 1: great)'.format(total_proba_final))
#     if total_proba_final > 0.7:
#         col2.write("You could use this new AI pic on Lkdn 👍")
#     else:
#         col2.write("We did not succeed in optimizing it 😱")
#
#     # except:
#     #      st.write("Oups! Something went wrong")




