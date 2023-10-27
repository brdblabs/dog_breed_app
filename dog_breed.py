import tensorflow as tf
import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from helper_functions.py import get_pred_label
from helper_functions.py import create_data_batches

def main():
  st.title('Dog Breed Image Classifier')
  st.write("Upload a photo of your favourite pooch")
 
  file = st.file_uploader('Please upload an image', type=['jpg', 'png'])
  if file:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    
    # Progress bar while model is running
    st.write('Please standby while your image is analyzed...')

    # Load saved model
    model = tf.keras.models.load_model('dog_breed_model_pretrained.h5')
    
    # Turn the uploaded image into a batch
    batch_image = create_data_batches(image)
        
    # Make a prediction on the uploaded image
    custom_preds = model.predict(batch_image)
    
    
    # Get image prediction labels
    custom_pred_labels = [get_pred_label(custom_preds[i]) for i in range(len(custom_preds))]
  
    
    # Get custom images (our unbatchify() function wont work since there are not labels)
    custom_images = []
    # Loop through unbatched data
    for image in custom_data.unbatch().as_numpy_iterator():
        custom_images.append(image)
 
    # Check custom image predictions
    plt.figure(figsize=(6,6))
    for i, image in enumerate(custom_images):
        plt.subplot()
        plt.xticks([])
        plt.yticks([])
        plt.title(custom_pred_labels[i])
        plt.imshow(image)

  else:
    st.text('You have not uploaded an image yet.')

if __name__ == '__main__':
  main()
