import streamlit as st
import tensorflow as tf
import numpy as np



def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size = (128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    predictions = model.predict(input_arr)
    return np.argmax(predictions)


st.header('Potato Disease Detection System ')
    
test_image = st.file_uploader("")

print(test_image)
if(test_image):
    if(st.button('Show Image')):
        st.image(test_image,width=4,use_container_width=True)
    
if(test_image):
    if(st.button('Predict')): 
        st.snow()
        st.write("Our Predictions")
        result_index = model_prediction(test_image)
        class_name = ['potato___Early_blight','Potato___Late_blight','Potato___healthy']
        st.success('Model is predicting its a {}'.format(class_name[result_index]))
    
    