import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

class MNISTGenerator:
    def __init__(self, model_path='mnist_generator_model.h5'):
        self.model_path = model_path
        if os.path.exists(self.model_path):
            self.generator = self.load_generator()
        else:
            self.generator = self.build_generator()
            self.train_generator()
            self.save_generator()

    def build_generator(self):
        model = models.Sequential([
            layers.Input(shape=(10,)),
            layers.Dense(128, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(512, activation='relu'),
            layers.Dense(784, activation='sigmoid')
        ])
        return model

    def train_generator(self):
        (x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()
        x_train = x_train.astype('float32') / 255.0
        x_train = x_train.reshape(-1, 784)
        y_train = tf.keras.utils.to_categorical(y_train, 10)

        self.generator.compile(optimizer='adam', loss='mse')
        self.generator.fit(y_train, x_train, epochs=50, batch_size=128)

    def save_generator(self):
        self.generator.save(self.model_path)
        print(f'Model saved to {self.model_path}')

    def load_generator(self):
        model = models.load_model(self.model_path)
        print(f'Model loaded from {self.model_path}')
        return model

    def apply_rotation(self, image, angle):
        # Add an extra dimension to make it multi-channel (e.g., convert from (28, 28) to (28, 28, 1))
        image = np.expand_dims(image, axis=-1)
        
        # Apply the rotation
        image = tf.keras.preprocessing.image.apply_affine_transform(image, theta=angle, fill_mode='nearest')
        
        # Remove the extra dimension to return it back to its original shape
        image = np.squeeze(image, axis=-1)
    
        return image
    
    def apply_multiplier(self, image, multiplier):
        image = np.clip(image * multiplier, 0, 1)
        return image

    def add_pixel_noise(self, image, noise_range):
        noisy_image = np.copy(image)
        height, width = image.shape
        for y in range(height):
            for x in range(width):
                noisy_image[y, x] += np.random.uniform(noise_range[0], noise_range[1])
        noisy_image = np.clip(noisy_image, 0, 1)  # Ensure pixel values remain in valid range [0, 1]
        return noisy_image

    def get_augmented_image_batch(self, digits, noise_range, rotation_range=(-10, 10), multiplier_range=(0.5, 1.5)):
        base_inputs = np.zeros((len(digits), 10))
        for i, digit in enumerate(digits):
            base_inputs[i, digit] = 1
        
        generated_digits = self.generator.predict(base_inputs)
        generated_digits = generated_digits.reshape(-1, 28, 28, 1)

        rotation_values = np.random.uniform(rotation_range[0], rotation_range[1], size=len(digits))
        multiplier_values = np.random.uniform(multiplier_range[0], multiplier_range[1], size=len(digits))

        noisy_digits = np.zeros_like(generated_digits)

        for i in range(len(noisy_digits)):
            noisy_image = self.add_pixel_noise(generated_digits[i].reshape(28, 28), noise_range)
            noisy_image = self.apply_rotation(noisy_image, rotation_values[i])
            noisy_image = self.apply_multiplier(noisy_image, multiplier_values[i])
            noisy_digits[i] = noisy_image.reshape(28, 28, 1)

        rotation_values = rotation_values.reshape(-1, 1, 1, 1)
        rotation_values = np.tile(rotation_values, (1, 28, 28, 1))

        multiplier_values = multiplier_values.reshape(-1, 1, 1, 1)
        multiplier_values = np.tile(multiplier_values, (1, 28, 28, 1))

        return noisy_digits, rotation_values, multiplier_values

def get_training_pairs_batch(digits, noise_range, model_path='mnist_generator_model.h5'):
    mnist_gen = MNISTGenerator(model_path=model_path)
    return mnist_gen.get_augmented_image_batch(digits, noise_range)
