# Import necessary libraries
import cv2
import numpy as np
from keras.models import load_model

# Load the pre-trained model
model = load_model('MNIST_model.h5')

# Load the image
image = cv2.imread('image.png')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply thresholding to preprocess the image
_, thresh = cv2.threshold(blurred, 110, 255, cv2.THRESH_BINARY_INV)

# Erosion and dilation to remove noise and holes
kernel = np.ones((3,3),np.uint8)
thresh = cv2.erode(thresh, kernel, iterations = 1)
thresh = cv2.dilate(thresh, kernel, iterations = 1)

# Canny edge detection to find edges
edges = cv2.Canny(thresh, 30, 200)

# Find contours in the thresholded image
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Display the thresholded image
cv2.imshow("thresh", edges)

# Process each contour
for contour in contours:
    # Get the bounding box coordinates
    x, y, w, h = cv2.boundingRect(contour)

    # Make sure the expanded ROI doesn't exceed the image dimensions
    x_start = max(0, x-25)
    y_start = max(0, y-25)
    x_end = min(gray.shape[1], x+w+25)
    y_end = min(gray.shape[0], y+h+25)

    # Extract the digit ROI from grayscale image
    digit_roi = gray[y_start:y_end, x_start:x_end]

    # Resize the digit ROI to the input shape of the model
    digit_roi_scaled = cv2.resize(digit_roi, (28, 28))

    # Reshape and normalize the digit ROI
    digit_roi_processed = digit_roi_scaled.reshape(1, 28, 28, 1).astype('float32') / 255

    # Predict the digit using the loaded model
    prediction = model.predict(digit_roi_processed)

    # Get the predicted digit and confidence score
    digit = np.argmax(prediction)
    confidence = np.max(prediction)

    # Draw a green bounding box around the digit
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Draw a purple bounding box that is 15 pixels larger than the image that the model sees
    cv2.rectangle(image, (x_start, y_start), (x_end, y_end), (255, 0, 255), 2)

    # Display the predicted digit and confidence score
    text = f"{digit} ({confidence:.1f})"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 2
    (w_text, h_text), _ = cv2.getTextSize(text, font, font_scale, thickness)
    cv2.rectangle(image, (x, y - h_text - 10), (x + w_text, y), (0, 255, 0), cv2.FILLED)
    cv2.putText(image, text, (x, y - 5), font, font_scale, (0, 0, 0), thickness)

# Display the result
cv2.imshow("Result", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
