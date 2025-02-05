import streamlit as st
import tensorflow as tf
import numpy as np

def model_prediction(test_image):
     model=tf.keras.models.load_model("train_plant_disease_model.keras")
     image=tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
     input_arr=tf.keras.preprocessing.image.img_to_array(image)
     input_arr=np.array([input_arr])
     prediction=model.predict(input_arr)
     return np.argmax(prediction)
st.logo("Syngenta_Logo.svg.png")
st.sidebar.title("PLANT DISEASE SYSTEM FOR SUSTAINABLE AGRICULTURE")
app_mode=st.sidebar.selectbox("select page",["Home","Disease Recognition"])

from PIL import Image
img=Image.open("p2.jpg")
st.image(img)

if(app_mode=="Home"):
    st.markdown("<h1 style='text-align: center;>PLANT DISEASE SYSTEM FOR SUSTAINABLE AGRICULTURE",unsafe_allow_html=True)

elif(app_mode=="Disease Recognition"):
    st.header("POTATO LEAF DISEASE DETECTION")
st.subheader("choose an image :")
test_img=st.file_uploader("")
if(st.button("show image")): 
    st.image(test_img,width=4,use_column_width=True)

if(st.button("predict")):
    st.snow()
    st.write("our prediction")
    result_index = model_prediction(test_img)
    class_name=["Potato_Early_blight","Potato_Late_blight","Potato_Healthy"]
    st.success("MODEL IS PREDICTED ITS A {}".format(class_name[result_index])) 
        