# Ai-MNIST-Advanced-model
<img src="https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue"/> <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"/> <img src="https://img.shields.io/badge/Keras-FF0000?style=for-the-badge&logo=keras&logoColor=white"/>

#### This project uses an AI model trained on the MNIST dataset to predict handwritten numbers with noise.

## Data Processing

The data processing pipeline is optimized to handle noisy images. It includes the following steps:
1. Noise reduction: The images are pre-processed to reduce the noise in the image.
2. Normalization: The pixel values are normalized to have zero mean and unit variance.
3. Data augmentation: The training data is augmented by applying random transformations to the images.

## Model

The AI model is a convolutional neural network (CNN) that is trained on the MNIST dataset. The model architecture and hyperparameters are chosen to achieve high accuracy on noisy images.

## Usage

To use the model, follow these steps:
1. Install the required dependencies by running `pip install -r requirements.txt`.
2. You can use the pre trained model `...\pre-trained model\MNIST_model.h5`<br />

If you just want to test it use this release: 
