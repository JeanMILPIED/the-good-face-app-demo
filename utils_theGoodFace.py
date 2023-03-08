import math
import os

import numpy as np

import cv2
from cv2 import CascadeClassifier, rectangle, cvtColor, Laplacian, GaussianBlur, Canny, LUT, calcHist, convertScaleAbs, detailEnhance, resize, adaptiveThreshold, medianBlur, imwrite
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont

import operator

import joblib

import time

from rembg import remove


basedir = '.'
depotdir = '.'
print(basedir, depotdir)

# function to extract features from an image

def get_image(my_image_name, my_folder=depotdir):
    """
    Converts an image number into the file path where the image is located,
    opens the image, and returns the image.
    """
    list_image_extensions=['jpg','jpeg','JPG','JPEG','png','PNG','bmp','BMP']
    for my_ext in list_image_extensions:
        filename = "{0}.{1}".format(my_image_name, my_ext)
        file_path = os.path.join(my_folder, filename)
        try:
            img = Image.open(file_path)
            break
        except:
            pass
    return img

def get_contrasted(image, mean_I, level=0.9, my_phi=1.1, my_theta=1):
    """
    increases the contrast of the image input to better get the face
    first converts to b&w and then to array
    """
    maxIntensity = 255.0  # depends on dtype of image data
    phi = my_phi
    theta = my_theta
    my_img_bw_array = np.array(image.convert('L'))
    if mean_I >= 127:
        newImage0 = (maxIntensity / phi) * (my_img_bw_array / (maxIntensity / theta)) ** 0.7
        newImage0 = newImage0.astype(np.uint8)
    elif mean_I < 127:
        newImage0 = (maxIntensity / phi) * (my_img_bw_array / (maxIntensity / theta)) ** level
        newImage0 = newImage0.astype(np.uint8)
    return newImage0


def get_basic_features(my_img):
    '''

    :param my_img:
    :return: basic features list of the image
    '''
    # create_features_list
    feat_list = []

    my_pixel_X, my_pixel_Y = my_img.size[0], my_img.size[1]

    # get the intensity features in black and white
    my_img_bw = my_img.convert('L')

    max_I, min_I, mean_I, std_I = np.max(my_img_bw), np.min(my_img_bw), np.round(np.mean(my_img_bw), 1), np.round(np.std(my_img_bw), 1)

    # return first features
    feat_list = [my_pixel_X, my_pixel_Y, max_I, min_I, mean_I, std_I]

    return feat_list


def load_haarcascade_objects():
    cascade_face = CascadeClassifier('haarcascade_frontalface_default.xml')
    cascade_profileface = CascadeClassifier('haarcascade_profileface.xml')
    cascade_eye = CascadeClassifier('haarcascade_eye.xml')
    cascade_smile = CascadeClassifier('haarcascade_smile.xml')
    cascade_glasses = CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
    return cascade_face, cascade_profileface, cascade_eye, cascade_smile, cascade_glasses

def face_detect(my_image_object, my_cascade_face, my_cascade_profileface, my_cascade_eye, my_cascade_smile, my_cascade_glasses):
    '''

    :param my_image_name:
    :param my_folder:
    :return: application of haarcascade methods to extract eye/smile etc positions adn features
    '''
    face_features = []
    cascade_face = my_cascade_face
    cascade_profileface = my_cascade_profileface
    cascade_eye = my_cascade_eye
    cascade_smile = my_cascade_smile
    cascade_glasses = my_cascade_glasses
    count_eye = 0
    count_smile = 0
    count_glasses = 0
    eye_center_list = []

    min_size_glasses = (0, 0)

    # Read the input image
    #img_other = get_image(my_image_name, my_folder=my_folder)
    img_other=my_image_object
    img = np.array(my_image_object.convert('RGB'))
    # Convert into grayscale
    my_img_features = get_basic_features(img_other)
    gray = get_contrasted(img_other, my_img_features[4])
    # Detect faces
    face = cascade_face.detectMultiScale(gray, 1.1, 5)
    profile_face = cascade_profileface.detectMultiScale(gray, 1.1, 5)
    if (len(face)) == 1:
        face_features.append(0)
        for (x_face, y_face, w_face, h_face) in face:
            ri_grayscale = gray[y_face:y_face + h_face, x_face:x_face + w_face]
            face_features.append(x_face + w_face / 2)
            face_features.append(y_face + h_face / 2)
            face_area = w_face * h_face
            face_features.append(face_area)

            min_size_smile = (int(w_face * 0.5), int(h_face * 0.2))
            max_size_smile = (int(w_face * 0.9), int(h_face * 0.4))
            min_size_eye = (int(w_face / 6), int(h_face / 6))
            max_size_eye = (int(w_face / 4), int(h_face / 4))
            max_size_glasses = (int(w_face), int(h_face))
            # divide faces into 2 zones (coordinates ranges)
            # top face: where eyes and glasses should be
            top_face_y_min = h_face / 2
            # bottom face: where smile should be
    elif (len(profile_face)) == 1:
        face_features.append(2)
        for (x_pface, y_pface, w_pface, h_pface) in profile_face:
            ri_grayscale = gray[y_pface:y_pface + h_pface, x_pface:x_pface + w_pface]
            face_features.append(x_pface + w_pface / 2)
            face_features.append(y_pface + h_pface / 2)
            pface_area = w_pface * h_pface
            face_features.append(pface_area)
            min_size_smile = (int(w_pface * 0.5), int(h_pface * 0.2))
            max_size_smile = (int(w_pface), int(h_pface * 0.6))
            min_size_eye = (int(w_pface / 6), int(h_pface / 6))
            max_size_eye = (int(w_pface / 4), int(h_pface / 4))
            max_size_glasses = (int(w_pface), int(h_pface))
            # top face: where eyes and glasses should be
            top_face_y_min = h_pface / 2
            # bottom face: where smile should be
    else:
        face_features.append(1)
        face_area = 0
        ri_grayscale = gray[0:1, 0:1]
        face_features.append(0)
        face_features.append(0)
        face_features.append(face_area)
        min_size_smile = (0, 1)
        max_size_smile = (0, 1)
        min_size_eye = (0, 1)
        max_size_eye = (0, 1)
        max_size_glasses = (0, 1)
        top_face_y_min = 0

    # detect eye
    eye = cascade_eye.detectMultiScale(ri_grayscale, 1.1, 5, minSize=min_size_eye, maxSize=max_size_eye)
    eye_area = 0
    for (x_eye, y_eye, w_eye, h_eye) in eye:
        if (y_eye) < top_face_y_min:
            count_eye += 1
            eye_area = eye_area + w_eye * h_eye
            min_size_glasses = (int(w_eye), int(h_eye))
            eye_center_list = eye_center_list + [x_eye + w_eye / 2, y_eye + h_eye / 2]
        else:
            pass
    if count_eye == 2:
        face_features.append(1)
        face_features.append(eye_area)
        angle_face = math.degrees(math.atan(
            np.abs(eye_center_list[3] - eye_center_list[1]) / np.abs(eye_center_list[2] - eye_center_list[0])))
        face_features.append(np.round(angle_face, 2))
    else:
        face_features.append(0)
        face_features.append(eye_area)
        face_features.append(0)

    # detect smile
    smile = cascade_smile.detectMultiScale(ri_grayscale, 1.1, 20, minSize=min_size_smile, maxSize=max_size_smile)
    for (x_smile, y_smile, w_smile, h_smile) in smile:
        if y_smile > top_face_y_min:
            smile_area = w_smile * h_smile
            count_smile += 1
        else:
            pass
    if (count_smile == 1):
        face_features.append(1)
        face_features.append(smile_area)
    else:
        face_features.append(0)
        face_features.append(0)
    # detect glasses
    glasses = cascade_glasses.detectMultiScale(ri_grayscale, 1.1, 10, minSize=min_size_glasses, maxSize=max_size_glasses)

    for (x_glasses, y_glasses, w_glasses, h_glasses) in glasses:
        if x_glasses > top_face_y_min:
            glasses_area = w_glasses * h_glasses
            count_glasses += 1
        else:
            pass
    if count_glasses == 2:
        face_features.append(1)
        face_features.append(glasses_area)
    else:
        face_features.append(0)
        face_features.append(0)

    img_cv = cvtColor(img, cv2.COLOR_BGR2RGB)

    return img_cv, face_features


def variance_of_laplacian(image):
    '''

    :param image:
    :return:  compute the Laplacian of the image and then return the focus
     measure, which is simply the variance of the Laplacian
    '''
    return Laplacian(image, cv2.CV_64F).var()


def is_it_blurry(my_image_object):
    '''

    :param my_image_name:
    :param my_folder:
    :param fm_threshold:
    :return: list of the blurr content of the image
    '''
    #img = get_image_cv(my_image_name, my_folder=my_folder)
    #img = get_image(my_image_name, my_folder=my_folder)
    img = np.array(my_image_object.convert('RGB'))
    gray = cvtColor(img, cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(gray)
    text = "Not Blurry"
    # if the focus measure is less than the supplied threshold,
    # then the image should be considered "blurry"
    return [round(fm, 2)]


def color_analysis(my_image_object):
    '''
        :param my_image_name:
    :param my_folder:
    :return: color palette of the image as a list (major col, light pc and dark pc)
    '''
    # obtain the color palette of the image
    img = my_image_object
    feature_col = 0
    palette = defaultdict(int)
    for pixel in img.getdata():
        palette[pixel] += 1
    # sort the colors present in the image
    sorted_x = sorted(palette.items(), key=operator.itemgetter(1), reverse=True)
    light_shade, dark_shade, shade_count, pixel_limit = 0, 0, 0, 200
    try:
        for i, x in enumerate(sorted_x[:pixel_limit]):
            if all(xx <= 50 for xx in x[0][:3]):  ## dull : too much darkness
                dark_shade += x[1]
            elif all(xx >= 200 for xx in x[0][:3]):  ## bright : too much whiteness
                light_shade += x[1]
            shade_count += x[1]
        feature_col = 1
    except:
        for i, x in enumerate(sorted_x[:pixel_limit]):
            if x[0] <= 50:  ## dull : too much darkness
                dark_shade += x[1]
            elif x[0] >= 200:  ## bright : too much whiteness
                light_shade += x[1]
            shade_count += x[1]
        feature_col = 0

    light_percent = round((float(light_shade) / shade_count) * 100, 2)
    dark_percent = round((float(dark_shade) / shade_count) * 100, 2)
    return [feature_col, light_percent, dark_percent]


def average_pixel_width(my_image_object):
    '''
    :param my_image_name:
    :param my_folder:
    :return: the average pixel width on the image
    '''
    img = my_image_object
    im_array = np.array(img.convert('RGB'))
    blurred = GaussianBlur(im_array, (3, 3), 0)
    # compute the median of the single channel pixel intensities
    sigma = 0.3
    v = np.median(blurred)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edges_sigma1 = Canny(blurred, lower, upper) / 255
    apw = (float(np.sum(edges_sigma1)) / (img.size[0] * img.size[1]))
    return [round(apw * 100, 2)]

def remove_background(my_image_object):
     output = remove(my_image_object)
     return output


def all_portrait_features(my_image_object,my_cascade_face, my_cascade_profileface, my_cascade_eye, my_cascade_smile, my_cascade_glasses):
    '''
    :param my_image_name:
    :param my_directory:
    :return: the concatenated list of features extracted form top listed functions
    '''
    my_all_feat_list = []
    my_all_feat_list = my_all_feat_list + get_basic_features(my_image_object)
    _, my_face_feat = face_detect(my_image_object,my_cascade_face, my_cascade_profileface, my_cascade_eye, my_cascade_smile, my_cascade_glasses)
    my_all_feat_list = my_all_feat_list + my_face_feat
    my_all_feat_list = my_all_feat_list + is_it_blurry(my_image_object)
    my_all_feat_list = my_all_feat_list + color_analysis(my_image_object)
    my_all_feat_list = my_all_feat_list + average_pixel_width(my_image_object)
    return my_all_feat_list


def clean_feat_vector(my_vector_dict):
    '''
    :param my_vector_df:
    :return: categorized features are transformed to int
    '''
    if my_vector_dict["smile_%"]!= my_vector_dict["smile_%"]:
        my_vector_dict["smile_%"]=0
    for i in my_vector_dict:
        if type(my_vector_dict[i]) == 'O':
            my_vector_dict[i] = int(my_vector_dict[i])
    return my_vector_dict


def get_features_from_new_image(image_object,my_cascade_face, my_cascade_profileface, my_cascade_eye, my_cascade_smile, my_cascade_glasses):
    '''
    :param image_name:
    :param my_folder:
    :return: get the features vector from an image as a dataframe
    '''
    features_vector = all_portrait_features(image_object,my_cascade_face, my_cascade_profileface, my_cascade_eye, my_cascade_smile, my_cascade_glasses)
    features_vector = ['tGFA_image'] + features_vector
    features_keys=['filename', 'pixel_X', 'pixel_Y', 'max_I', 'min_I', 'mean_I', 'std_I', 'type', 'central_face_x',
                 'central_face_y', 'face_area', 'eye_status', 'tot_eye_area', 'face_angle', 'smile_status',
                 'smile_area', 'glasses_status', 'glasses_area', 'blurry_rate', 'black_or_white', 'light_percent',
                 'dark_percent', 'APW_rate']
    features_dict = {features_keys[i]: features_vector[i] for i in range(len(features_keys))}
    my_pc_face = features_dict["face_area"] / (features_dict["pixel_X"] * features_dict["pixel_Y"]) * 100
    my_pc_face = np.round(my_pc_face, 1)
    features_dict['face_%'] = my_pc_face
    if int(features_dict["face_area"]) > 0:
        my_pc_smile = features_dict["smile_area"] / (features_dict["face_area"]) * 100
        my_pc_smile = np.round(my_pc_smile, 1)
    else:
        my_pc_smile = 0
    features_dict['smile_%'] = my_pc_smile
    features_dict_ok={}
    for my_feat in ['filename', 'pixel_X', 'pixel_Y', 'max_I', 'min_I',
       'mean_I', 'std_I', 'type', 'face_area', 'eye_status', 'tot_eye_area',
       'face_angle', 'smile_status', 'smile_area', 'glasses_status',
       'glasses_area', 'blurry_rate', 'light_percent', 'dark_percent',
       'APW_rate', 'face_%', 'smile_%']:
        features_dict_ok[my_feat]=features_dict[my_feat]
    features_dict_ok = clean_feat_vector(features_dict_ok)
    features_dict = clean_feat_vector(features_dict)
    return features_dict_ok, features_dict


### functions to predict score from models

def load_scale_models(my_model_folder=basedir):
    '''
    :param my_model_folder:
    :return: trained scaler and models
    '''
    # we load the scaler
    my_scaler = joblib.load(my_model_folder + '/models2/theScaler.gz')
    # we load the models
    my_RFmodel = joblib.load(my_model_folder + '/models2/theRFmodel.sav')
    #my_NNmodel = joblib.load(my_model_folder + '/models2/theNNmodel.sav')
    my_NNmodel = joblib.load(my_model_folder + '/models2/theRFmodel.sav')
    my_SVCmodel = joblib.load(my_model_folder + '/models2/theSVCmodel.sav')
    return my_scaler, my_RFmodel, my_NNmodel, my_SVCmodel


def predict_proba_from_image_feat(my_image_feat, my_scaler, my_rfmodel, my_nnmodel, my_svcmodel):
    '''

    :param my_image_feat:
    :param my_scaler: scaler applied to image features
    :param my_rfmodel: RandomForest classifier model loaded
    :param my_nnmodel: NeuralNetwork classifier model loaded
    :param my_svcmodel: Standard Vector Classifier model loaded
    :return: list of probability of class1 from each classifier
    '''
    X = np.array(my_image_feat[1:]).reshape(1,-1)
    print(X.shape)
    X_scale = my_scaler.transform(X)
    Y_pred_rf = my_rfmodel.predict_proba(X_scale)
    Y_pred_nn = my_nnmodel.predict_proba(X_scale)
    Y_pred_SVC = my_svcmodel.predict_proba(X_scale)
    my_result = [Y_pred_rf[0], Y_pred_nn[0], Y_pred_SVC[0]]
    return my_result


### functions to get old score, modify an image and get new score

def auto_portraitImage_optimisation(my_image, my_folder=depotdir,
                                    my_gamma=1.0, my_clip_hist_percent=5, my_kernel_size=(5, 5), my_sigma=1.0,
                                    my_amount=1.0, my_threshold=0, my_crop_factor=2, my_param_1=51, my_param_2=10,
                                    my_k_size=20, my_param_3=4):
    '''
    :param my_image: name of the input image
    :param my_folder: folder where image is loaded
    :param my_gamma: gamma parameter
    :param my_clip_hist_percent: parameter
    :param my_kernel_size: parameter
    :param my_sigma: parameter
    :param my_amount: paramater
    :param my_threshold: parameter
    :param my_crop_factor: parameter to crop input image around face only
    :param my_param_1: parameter
    :param my_param_2: parameter
    :param my_k_size: parameter
    :param my_param_3: parameter
    :return: cumulated probability score from original image and final transformed image. transformed image is saved in folder
    '''
    #load objects
    my_cascade_face, my_cascade_profileface, my_cascade_eye, my_cascade_smile, my_cascade_glasses=load_haarcascade_objects()
    #try:
    start_time = time.time()
    initial_img = get_image(my_image, my_folder=my_folder)
    img = np.array(initial_img.convert('RGB'))

    # 1. adjust_gamma
    invGamma = 1.0 / my_gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    new_gamma_image = LUT(img, table)
    new_gamma_image = cvtColor(new_gamma_image, cv2.COLOR_RGB2BGR)
    # 2. auto_brightness and contrast
    gray = cvtColor(new_gamma_image, cv2.COLOR_BGR2GRAY)

    # Calculate grayscale histogram
    hist = calcHist([gray], [0], None, [256], [0, 256])
    hist_size = len(hist)

    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index - 1] + float(hist[index]))

    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent = (maximum / 100.0) * my_clip_hist_percent
    clip_hist_percent = clip_hist_percent / 2

    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    # Locate right cut
    maximum_gray = hist_size - 1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha
    BandC_image = convertScaleAbs(new_gamma_image, alpha=alpha, beta=beta)

    # 3. sharpen the image:Return a sharpened version of the image, using an unsharp mask
    blurred = GaussianBlur(BandC_image, my_kernel_size, my_sigma)
    sharpened_image = float(my_amount + 1) * BandC_image - float(my_amount) * blurred
    sharpened_image = np.maximum(sharpened_image, np.zeros(sharpened_image.shape))
    sharpened_image = np.minimum(sharpened_image, 255 * np.ones(sharpened_image.shape))
    sharpened_image = sharpened_image.round().astype(np.uint8)
    #sharpened_image = detailEnhance(sharpened_image, 10, 0.15)
    if my_threshold > 0:
        low_contrast_mask = np.absolute(BandC_image - blurred) < my_threshold
        np.copyto(sharpened_image, BandC_image, where=low_contrast_mask)

    # 4. crop the image
    img = sharpened_image
    feat_dict, feat_dict_raw = get_features_from_new_image(initial_img,my_cascade_face, my_cascade_profileface, my_cascade_eye, my_cascade_smile, my_cascade_glasses)
    central_x = int(feat_dict_raw["central_face_x"])
    central_y = int(feat_dict_raw["central_face_y"])
    crop_amplitude = int(np.sqrt(feat_dict_raw["face_area"]) / 2 * my_crop_factor)
    x_min = central_x - crop_amplitude
    x_max = central_x + crop_amplitude
    y_min = central_y - crop_amplitude
    y_max = central_y + crop_amplitude
    if x_min < 0:
        x_min = 0
    if x_max > int(feat_dict_raw["pixel_X"]):
        x_max = int(feat_dict_raw["pixel_X"])
    if y_min < 0:
        y_min = 0
    if y_max > int(feat_dict_raw["pixel_Y"]):
        y_max = int(feat_dict_raw["pixel_Y"])
    crop_img = img[y_min:y_max, x_min:x_max]
    target_x_size = 600
    y_size = int(crop_img.shape[1] * 600 / (crop_img.shape[0]+1))
    crop_img_resized = resize(crop_img, (y_size, 600), interpolation=cv2.INTER_LINEAR)

    # 5. blur the background
    # we convert back the image from BGR to RGB
    img_cv = crop_img_resized.copy()
    gray_img = cvtColor(crop_img_resized, cv2.COLOR_BGR2GRAY)
    thresh = adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, my_param_1,
                                   my_param_2)
    thresh = cv2.bitwise_not(thresh)
    element = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(my_k_size, my_k_size))
    morph_img = thresh.copy()
    cv2.morphologyEx(src=thresh, op=cv2.MORPH_CLOSE, kernel=element, dst=morph_img)
    contours, _ = cv2.findContours(morph_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    sorted_areas = np.sort(areas)
    number_of_areas = sorted_areas.shape[0]

    cnt = contours[areas.index(sorted_areas[-1])]  # the biggest contour

    mask = np.zeros(gray_img.shape, np.uint8)
    cv2.drawContours(mask, [cnt], 0, 255, -1)
    blur = GaussianBlur(img_cv, (5, 5), my_param_3)
    out = img_cv.copy()
    out[mask == 0] = blur[mask == 0]
    out = medianBlur(out, 3)
    imwrite(my_folder + '/' + my_image + '_ALLcorrected.jpg', out)
    #
    out_nobgd=remove_background(out)
    imwrite(my_folder + '/' + my_image + '_ALLcorrected.jpg', out_nobgd)

    # return out features before and after modifications
    # we load scaler and models (3 models and we aggregate the score)
    my_scaler, my_RFmodel, my_NNmodel, my_SVCmodel = load_scale_models()

    #we extract features from initial image
    feat_df_initial = list(feat_dict.values())
    print(len(feat_df_initial))
    result_initial = predict_proba_from_image_feat(feat_df_initial, my_scaler, my_RFmodel, my_NNmodel, my_SVCmodel)
    total_proba_initial = np.round(np.sum([result_initial[0][1], result_initial[1][1], result_initial[2][1]])/2,1)

    feat_df_dict_final, _ = get_features_from_new_image(get_image(my_image + '_ALLcorrected', my_folder), my_cascade_face, my_cascade_profileface, my_cascade_eye, my_cascade_smile, my_cascade_glasses)
    feat_df_final=list(feat_df_dict_final.values())
    print(len(feat_df_final))
    result_final = predict_proba_from_image_feat(feat_df_final, my_scaler, my_RFmodel, my_NNmodel, my_SVCmodel)
    total_proba_final = np.round(np.sum([result_final[0][1], result_final[1][1], result_final[2][1]])/2,1)
    #score_text1='Your previous score was '+str(total_proba_initial)
    score_text1='generated by TheGoodFace'
    score_text2='Your new score is '+str(total_proba_final)
    write_results_on_image(my_image+'_ALLcorrected', score_text1, score_text2, my_folder)

    #cv2.imwrite(my_folder + '/' + my_image + '_ALLcorrected.jpg', my_scored_image)

    print("initial_proba= ", total_proba_initial)
    print("final_proba= ", total_proba_final)
    end_time = time.time()
    print(f"Runtime of the program is {end_time - start_time}")
    return total_proba_initial, total_proba_final
    #except:
    #    return 0, 0

def write_results_on_image( my_image_name, my_text1, my_text2, my_folder=depotdir):
    my_size1=50
    font1 = ImageFont.truetype("Lobster-Regular.ttf", my_size1)
    font2 = ImageFont.truetype("Lobster-Regular.ttf", my_size1*2)
    my_image = get_image(my_image_name, my_folder)
    width, height = my_image.size
    draw = ImageDraw.Draw(my_image)
    text1 = my_text1
    text2 = my_text2
    textwidth, textheight = draw.textsize(text1)
    margin = 50
    x = textwidth + margin - 150
    y = height - textheight - margin - 50
    draw.text((x, y), text1, font = font1, fill=(250,250,100))
    #draw.text((x,y-50), text2, font=font2, fill=(250,250,100))
    my_image.save(my_folder + '/' + my_image_name +'.jpg', "JPEG")
