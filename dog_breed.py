import tensorflow as tf
import tensorflow_hub as hub
import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing import image

def main():
  st.title('Dog Breed Identifier')
  st.write("Upload a photo of your favourite pooch")
 
  file = st.file_uploader('Please upload an image', type=['jpg', 'png'])
  if file:
    uploaded_image = Image.open(file)
    st.image(uploaded_image, use_column_width=True)
    
    # Progress bar while model is running
    st.write('Please standby while your image is analyzed...')

    # Load breed labels
    labels_csv = pd.read_csv("labels.csv")
    labels = labels_csv["breed"].to_numpy()  # convert to Numpy array

    # Find unique breeds within labels
    unique_breeds = np.unique(labels)

    # Process the uploaded image
    new_image = uploaded_image.resize((224,224))
    # process_image = image.load_img(uploaded_image, target_size=(224, 224))
    new_image = image.img_to_array(new_image) / 255.0
    new_image = new_image.reshape((1, 224, 224, 3))
    
    # Load saved model
    model = tf.keras.models.load_model('dog_breed_model_pretrained.h5', custom_objects={'KerasLayer':hub.KerasLayer})
         
    # Make a prediction on the uploaded image
    predictions = model.predict(new_image, batch_size=1) 
        
    # Get image prediction labels
    labels_csv = pd.read_csv("labels.csv")
    labels = labels_csv["breed"].to_numpy()
    unique_breeds = np.unique(labels)
    predicted_dog = unique_breeds[np.argmax(predictions[0])]
    st.write(f"Predicted dog: {predicted_dog}")

    # Plot predicted dog breed
    num_rows = 1
    num_cols = 1
    num_images = 1

    fig = plt.figure(figsize=(5*2*num_cols, 5*num_rows))
    for i in range(num_images):
      plt.subplot(num_rows, 2*num_cols, 2*i+1)
      plt.xticks([])
      plt.yticks([])
      plt.title(custom_pred_labels[i])
      plt.imshow(image)

      # plt.subplot(num_rows, 2*num_cols, 2*i+2)
      # plot_pred_conf(prediction_probabilities=custom_preds,
                labels=custom_pred_labels)
      # plt.tight_layout(h_pad=1.0)
    st.pyplot(fig)
    
  else:
    st.text('You have not uploaded an image yet.')

if __name__ == '__main__':
  main()
