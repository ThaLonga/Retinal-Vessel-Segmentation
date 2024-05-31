from PIL import Image
import os
import numpy as np
import cv2
import math

class DataLoader:
    
    @staticmethod
    def load_images_from_dir(image_dir):
      images_list = os.listdir(image_dir)
      images_list.sort()
      images = []
      for filename in images_list:
          # Construct the full file path
          filepath = os.path.join(image_dir, filename)
    
          # Open the image file
          if filepath.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp', '.gif')):
            with Image.open(filepath) as img:

                # Convert image to numpy array and normalize pixel values if necessary
                img_array = np.array(img) / 255.0  # Normalization to [0,1] range if needed

                # Resize to make divisible by 32
                new_size = math.floor(min(img_array.shape[0]/32, img_array.shape[1]/32))*32
                img_array = cv2.resize(img_array, (new_size, new_size), interpolation=cv2.INTER_AREA)

                # Append the array to the list of images
                images.append(img_array)

        # Convert list of arrays into a single 3D numpy array
      return np.array(images) 
      
    @staticmethod
    def load_images_from_dir_no_preprocessing(image_dir):
      images_list = os.listdir(image_dir)
      images_list.sort()
      images = []
      for filename in images_list:
          # Construct the full file path
          filepath = os.path.join(image_dir, filename)
    
          # Open the image file
          if filepath.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp', '.gif')):
            with Image.open(filepath) as img:

                # Convert image to numpy array and normalize pixel values if necessary
                img_array = np.array(img).astype('float32') / 255.0  # Normalization to [0,1] range if needed
                images.append(img_array)

        # Convert list of arrays into a single 3D numpy array
      return np.array(images) 
