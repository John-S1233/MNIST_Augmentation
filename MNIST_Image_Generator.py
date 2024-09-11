import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

class MNISTGenerator:
    def __init__(self):
        pass  # No initialization required as we are not using a pre-trained generator

    def apply_rotation(self, image, angle):
        image = tf.keras.preprocessing.image.apply_affine_transform(image, theta=angle, fill_mode='nearest')
        return image

    def apply_multiplier(self, image, multiplier):
        image = np.clip(image * multiplier, 0, 1)
        return image

    def add_pixel_noise(self, image, noise_range):
        noisy_image = np.copy(image)
        height, width = image.shape[:2]
        for y in range(height):
            for x in range(width):
                noisy_image[y, x] += np.random.uniform(noise_range[0], noise_range[1])
        noisy_image = np.clip(noisy_image, 0, 1)  # Ensure pixel values remain in valid range [0, 1]
        return noisy_image

    def get_augmented_image_batch(self, images, noise_range, rotation_range, multiplier_range):
        rotation_values = np.random.uniform(rotation_range[0], rotation_range[1], size=len(images))
        multiplier_values = np.random.uniform(multiplier_range[0], multiplier_range[1], size=len(images))
        augmented_images = []

        for i in range(len(images)):
            image = images[i]
            
            # Apply rotation first
            augmented_image = self.apply_rotation(image, rotation_values[i])
            
            # Apply multiplier second
            augmented_image = self.apply_multiplier(augmented_image, multiplier_values[i])

            # Apply noise last
            noisy_image = self.add_pixel_noise(augmented_image, noise_range)

            augmented_images.append(noisy_image)

        augmented_images = np.array(augmented_images)
        return augmented_images, rotation_values, multiplier_values

def get_training_pairs_batch(images, noise_range, rotation_range, multiplier_range):
    mnist_gen = MNISTGenerator()
    return mnist_gen.get_augmented_image_batch(images, noise_range, rotation_range, multiplier_range)
