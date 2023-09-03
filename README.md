# Ai-MNIST-Advanced-model
<img src="https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue"/> <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"/> <img src="https://img.shields.io/badge/Keras-FF0000?style=for-the-badge&logo=keras&logoColor=white"/>
#### This project uses an AI model trained on the MNIST dataset to predict handwritten numbers with noise(real world data).
> # :warning: **WARNING**
>  Please note that this model is optimized for predicting very noisy images. As a result, it may not perform with very high accuracy on the standard MNIST validation data. Keep this in mind when evaluating the model’s
> performance.
## Release
> ### Newest release 📃
> #### [Go to newest release](https://github.com/Aydinhamedi/Ai-MNIST-Advanced-model/releases/tag/V0.3.6)

> ### Examples 📚
> #### [Go to example](TR.md)
## Data Processing (training)

The data processing pipeline is optimized to handle noisy images. It includes the following steps:
1. Random zooming: The images are randomly zoomed in or out to create variations in the training.
2. Random cropping: The images are randomly cropped to create variations in the training data.
3. Adding noise: Random noise is added to the images to simulate real-world conditions.
4. generating data
5. Increasing the number of training records: The data augmentation techniques increase the number of training records from the original 60,000 to around 760,000.

## Model

The AI model is a convolutional neural network (CNN) that is trained on the MNIST dataset. The model architecture and hyperparameters are chosen to achieve high accuracy on noisy images(real world data).

## Usage

To use the model, follow these steps:
1. Install the required dependencies by running `pip install -r requirements.txt`.
2. You can use the pre trained model `...\pre-trained model\MNIST_model.h5`
3. download the data set with `download dataset.py`<br />


