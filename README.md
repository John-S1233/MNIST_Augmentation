# AugmentMNIST-CNN
![Augmented Digits](https://github.com/user-attachments/assets/91da3f44-10b3-434c-8a70-a50dcb0f35a1)
## Overview

AugmentMNIST-CNN is a Python project that focuses on generating and augmenting the MNIST dataset using advanced image processing techniques and training a Convolutional Neural Network (CNN) on this augmented data. The augmentation includes adding noise, rotating the images, and applying multipliers to enhance the training dataset, leading to more robust and generalized models.

## Features

- **Data Augmentation**: Includes random rotations, noise addition, and multipliers to create a more diverse training set.
- **Customizable Augmentation**: Parameters for augmentation can be easily modified to experiment with different noise levels, rotation angles, and multipliers.
- **Efficient Dataset Handling**: Augmented datasets are generated, combined with the original MNIST dataset, and shuffled for optimal training.
- **CNN Model Training**: A robust Convolutional Neural Network architecture is provided to train on the augmented dataset, ensuring improved accuracy and generalization.
- **GPU Support**: GPU memory growth is enabled for efficient use of resources during training.

## Project Structure

- **`AugmentMNIST-CNN.py`**: Contains the core class `MNISTDataAugmentation`, which handles dataset augmentation, CNN model construction, and training. This script generates the augmented dataset and trains the model.
- **`MNIST_Image_Generator.py`**: Provides utility functions for applying specific augmentations like rotation, noise, and multiplier to MNIST images.
- **`load_dataset.py`**: Handles loading and visualizing the augmented dataset, allowing users to inspect images with specific augmentation parameters.
- **Dataset**: Generated dataset files are stored in `.npz` format and include both original and augmented images.
- **Model**: Trained models are saved in `.h5` format for easy loading and further evaluation.

## Requirements

- Python 3.9 (For Windows GPU Support)
- TensorFlow 2.x
- NumPy
- Matplotlib
- Scikit-learn

