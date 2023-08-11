from keras.models import load_model
from PIL import Image
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
img_array = np.array(img)
img_array = img_array.reshape(1, 28, 28, 1)

# Make predictions
predictions = model.predict(img_array)
predicted_digit = np.argmax(predictions)
confidence = predictions[0][predicted_digit]
print(f'The predicted digit is {predicted_digit} with a confidence of {confidence:.2f}')

# Create directories if they don't exist
if not os.path.exists('IMG_ITER'):
    os.mkdir('IMG_ITER')
if not os.path.exists('IMG_ITER/white'):
    os.mkdir('IMG_ITER/white')
if not os.path.exists('IMG_ITER/black'):
    os.mkdir('IMG_ITER/black')

# Apply threshold and save images
thresholds = range(20, 250, 5)
white_confidences = []
black_confidences = []
inverted_white_confidences = []
inverted_black_confidences = []
digit_counts = np.zeros(10)
digit_confidences = np.zeros(10)
for threshold in thresholds:
    W_threshold = threshold if threshold <= 195 else 195
    white_thresholded_img = img.point(lambda x: 255 if x < W_threshold else x)
    white_thresholded_img.save(f'IMG_ITER/white/threshold_{W_threshold}.png')
    white_thresholded_img_array = np.array(white_thresholded_img)
    white_thresholded_img_array = white_thresholded_img_array.reshape(1, 28, 28, 1)
    predictions = model.predict(white_thresholded_img_array)
    predicted_digit = np.argmax(predictions)
    confidence = predictions[0][predicted_digit]
    white_confidences.append(confidence)
    digit_counts[predicted_digit] += 1
    digit_confidences[predicted_digit] += confidence
    print(f'Threshold: {W_threshold}, Color: white, Predicted digit: {predicted_digit}, Confidence: {confidence:.2f}')
    
    black_thresholded_img = img.point(lambda x: 0 if x < threshold else x)
    black_thresholded_img.save(f'IMG_ITER/black/threshold_{threshold}.png')
    black_thresholded_img_array = np.array(black_thresholded_img)
    black_thresholded_img_array = black_thresholded_img_array.reshape(1, 28, 28, 1)
    predictions = model.predict(black_thresholded_img_array)
    predicted_digit = np.argmax(predictions)
    confidence = predictions[0][predicted_digit]
    black_confidences.append(confidence)
    digit_counts[predicted_digit] += 1
    digit_confidences[predicted_digit] += confidence
    print(f'Threshold: {threshold}, Color: black, Predicted digit: {predicted_digit}, Confidence: {confidence:.2f}')

    inverted_white_thresholded_img = img.point(lambda x: x if x < W_threshold else 255 - x)
    inverted_white_thresholded_img_array = np.array(inverted_white_thresholded_img)
    inverted_white_thresholded_img_array = inverted_white_thresholded_img_array.reshape(1, 28, 28, 1)
    predictions = model.predict(inverted_white_thresholded_img_array)
    predicted_digit = np.argmax(predictions)
    confidence = predictions[0][predicted_digit]
    inverted_white_confidences.append(confidence)
    digit_counts[predicted_digit] += 1
    digit_confidences[predicted_digit] += confidence

    inverted_black_thresholded_img = img.point(lambda x: x if x < threshold else 255 - x)
    inverted_black_thresholded_img_array = np.array(inverted_black_thresholded_img)
    inverted_black_thresholded_img_array = inverted_black_thresholded_img_array.reshape(1, 28, 28, 1)
    predictions = model.predict(inverted_black_thresholded_img_array)
    predicted_digit = np.argmax(predictions)
    confidence = predictions[0][predicted_digit]
    inverted_black_confidences.append(confidence)
    digit_counts[predicted_digit] += 1
    digit_confidences[predicted_digit] += confidence
# Create graph
plt.plot(thresholds, white_confidences, label='White')
plt.plot(thresholds, black_confidences, label='Black')
plt.plot(thresholds, inverted_white_confidences, label='Inverted White')
plt.plot(thresholds, inverted_black_confidences, label='Inverted Black')
plt.xlabel('Threshold')
plt.ylabel('Confidence')
plt.legend()
plt.show()

# Display most predicted digits and their average confidence
digit_counts = digit_counts / np.sum(digit_counts)
digit_scores = np.zeros(10)
for digit in range(10):
    if digit_counts[digit] > 0:
        avg_confidence = digit_confidences[digit] / ((digit_counts[digit] * len(thresholds) * 2) * 2)
        print(f'Digit: {digit}, Percentage: {digit_counts[digit]:.2f}, Average Confidence: {avg_confidence:.2f}')
        digit_scores[digit] = digit_counts[digit] + (avg_confidence / 3)

best_digit = np.argmax(digit_scores)
best_score = digit_scores[best_digit]
print(f'\nThe best result is for digit {best_digit} with a score of {best_score:.2f}')
