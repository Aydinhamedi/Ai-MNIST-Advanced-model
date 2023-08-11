# Ai-MNIST-Advanced-model
<img src="https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue"/> <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"/> <img src="https://img.shields.io/badge/Keras-FF0000?style=for-the-badge&logo=keras&logoColor=white"/>

#### This project uses an AI model trained on the MNIST dataset to predict handwritten numbers with noise.
> # :warning: **WARNING**
>  Please note that this model is optimized for predicting very noisy images. As a result, it may not perform with very high accuracy on the standard MNIST validation data. Keep this in mind when evaluating the model’s
> performance.

> # :warning: **WARNING**
> This model was not generated with this code due to some version control problems. We are aware of this issue and it will be fixed soon. 
> However, please note that **this model is better than the model that the current code generates**. 
> Thank you for your patience.

## Release
> ### Newest release 📃
> https://github.com/Aydinhamedi/Ai-MNIST-Advanced-model/releases/tag/V0.3.5-beta
## Data Processing (training)

The data processing pipeline is optimized to handle noisy images. It includes the following steps:
1. Random zooming: The images are randomly zoomed in or out to create variations in the training.
2. Random cropping: The images are randomly cropped to create variations in the training data.
3. Adding noise: Random noise is added to the images to simulate real-world conditions.
4. Increasing the number of training records: The data augmentation techniques increase the number of training records from the original 60,000 to around 480,000.

## Model

The AI model is a convolutional neural network (CNN) that is trained on the MNIST dataset. The model architecture and hyperparameters are chosen to achieve high accuracy on noisy images.

## Usage

To use the model, follow these steps:
1. Install the required dependencies by running `pip install -r requirements.txt`.
2. You can use the pre trained model `...\pre-trained model\MNIST_model.h5`<br />

