# Brain Hemorrhage Detection using Convolutional Neural Networks
This repository contains a Python script for detecting brain hemorrhage in medical images using convolutional neural networks (CNNs). The script processes a dataset of head CT scan images, labels them based on the presence or absence of hemorrhage, and trains a CNN model to classify new images.

## Dataset
The dataset consists of head CT scan images stored as PNG files along with corresponding labels indicating the presence of hemorrhage. The dataset is divided into training, validation, and test sets. </br>

- head_ct folder contains the CT scan images. </br>
- labels.csv file contains the corresponding labels for each image.
## Preprocessing
The script preprocesses the images by resizing them to a uniform size of 128x128 pixels and normalizing their pixel values. Data augmentation techniques such as rotation, zooming, and flipping are applied to increase the robustness of the model.

## Model Architecture
The CNN model architecture consists of several convolutional layers followed by max-pooling layers to extract features from the images. Dropout layers are added to reduce overfitting, and dense layers with sigmoid activation are used for classification.

## Training
The model is trained using the training data and validated using the validation data. Model checkpoints are saved to track the best performing model based on validation accuracy.

## Evaluation
The trained model is evaluated using the test set to measure its performance in detecting brain hemorrhage. The accuracy and confusion matrix are calculated to assess the model's effectiveness.

## Results
The model achieved an accuracy of 85% on the test set.</br>
True positive: 8, True negative: 9, False positive: 1, False negative: 2</br>
Total accuracy: 85%
## Requirements
Python 3 </br>
TensorFlow</br>
Keras</br>
NumPy</br>
Pandas</br>
Matplotlib</br>
Seaborn</br>
OpenCV
