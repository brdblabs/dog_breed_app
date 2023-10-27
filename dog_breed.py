import tensorflow as tf
import tensorflow_hub as hub
import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from helper_functions import get_pred_label
# from helper_functions import create_data_batches

def main():
  st.title('Dog Breed Image Classifier')
  st.write("Upload a photo of your favourite pooch")
 
  file = st.file_uploader('Please upload an image', type=['jpg', 'png'])
  if file:
    uploaded_image = Image.open(file)
    st.image(uploaded_image, use_column_width=True)
    
    # Progress bar while model is running
    st.write('Please standby while your image is analyzed...')

    # Load saved model
    model = tf.keras.models.load_model('dog_breed_model_pretrained.h5', custom_objects={'KerasLayer':hub.KerasLayer})
    
    # Turn the uploaded image into a batch
    # batch_image = create_data_batches(image)

    resized_image = uploaded_image.resize((32, 32))
    img_array = np.array(resized_image) / 255
    img_array = img_array.reshape((1, 32, 32, 3))
    
    # Read in image file
    # process_image = tf.io.read_file(uploaded_image)
    # Turn the jpeg image into a numerical Tensor with 3 colours
    # process_image = tf.image.decode_jpeg(process_image, channels=3)
    # Convert the colour channel values from 0-225 values to 0-1 values
    # process_image = tf.image.convert_image_dtype(process_image, tf.float32)
    # Resize the image to our desired size (224, 224)
    # process_image = tf.image.resize(process_image, size=[224, 224])
        
    # Make a prediction on the uploaded image
    custom_preds = model.predict(img_array)
    
    
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
