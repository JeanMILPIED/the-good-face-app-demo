# the-good-face-app (demo) 📷😎
Streamlit demonstrator of the good face app  

1. load a portrait picture (File or Webcam)  
2. the custom portrait evaluation AI model rates the picture  
3. the custom portrait optimisation function boosts the picture  
4. the new pic is rated by portrait evaluation AI model  

#### What is the back end ?
a ML model (ensemble model) to predict "good portrait" probability score,  
The model has been trained (see notebook) on a custom dataset of unbiased 1000+ portrait images.  
This is not a deeplearning model: features are extracted from the image and used for classification prediction.  
Amongst key features, there are : face size, smile or not, colour ratio, eyes open or not, etc...   
