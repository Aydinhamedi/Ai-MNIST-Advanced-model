from keras.models import load_model
from PIL import Image, ImageFilter
import numpy as np
import os
import matplotlib.pyplot as plt
import sys

try:
    model = load_model('MNIST_model.h5')
except (ImportError, IOError) as e:
    print(f'failed to load the model ERROR:\n{e}')
    sys.exit()
else:
    print('loading model done.')

# Load and preprocess image
image_dir = input('image dir(.png/.jpg):')
if image_dir == '':
    image_dir = 'image.jpg'
try:
    img = Image.open(image_dir).convert('L')
    if img.format == 'JPEG':
        img = img.convert('RGB')
except FileNotFoundError:
    try:
        img = Image.open(image_dir).convert('L')
        if img.format == 'JPEG':
            img = img.convert('RGB')
    except FileNotFoundError:
        print('cant find: image.(png/jpg)')
        sys.exit()
    
img = img.resize((28, 28))
img = img.filter(ImageFilter.GaussianBlur(0.5))
img_array = np.array(img)
img_array = img_array.reshape(1, 28, 28, 1).astype('float32') / 255

# Make predictions
predictions = model.predict(img_array)
predicted_digit = np.argmax(predictions)
confidence = predictions[0][predicted_digit]
print(f'The predicted digit is {predicted_digit} with a confidence of {confidence:.3f}')
print('other digits:')
for i in range(len(predictions[0])):
    print(f'{i} with a confidence of {predictions[0][i]:.3f}')
    

