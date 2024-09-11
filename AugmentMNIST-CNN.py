import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.utils import shuffle
from tensorflow.keras import backend as K
K.clear_session()

# Enable GPU growth
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print("GPU growth enabled.")
    except RuntimeError as e:
        print(f"Error enabling GPU growth: {e}")

class MNISTDataAugmentation:
    def __init__(self, augmentation_factor=1, noise_value_range=(0.1, 0.5), rotation_range=(-10, 10), multiplier_range=(0.5, 1.5), batch_size=128, dataset_path='augmented_mnist_dataset.npz', regenerate=False, train_model=True):
        self.augmentation_factor = augmentation_factor
        self.noise_value_range = noise_value_range
        self.rotation_range = rotation_range
        self.multiplier_range = multiplier_range
        self.batch_size = batch_size
        self.dataset_path = dataset_path
        self.regenerate = regenerate
        self.train_model = train_model

    def generate_augmented_data(self):
        # Check if we need to regenerate the dataset
        if not self.regenerate and os.path.exists(self.dataset_path):
            print(f"Loading dataset from {self.dataset_path}")
            data = np.load(self.dataset_path)
            combined_images = data['combined_images']
            combined_noise_values = data['combined_noise_values']
            combined_rotation_values = data['combined_rotation_values']
            combined_multiplier_values = data['combined_multiplier_values']
            combined_labels = data['combined_labels']
            x_test = data['x_test']
            y_test = data['y_test']
            return combined_images, combined_noise_values, combined_rotation_values, combined_multiplier_values, combined_labels, x_test, y_test

        # Load MNIST dataset
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0

        # Reshape for CNN input
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)

        # Generate augmented data
        new_images, new_noise_values, new_rotation_values, new_multiplier_values, new_labels = [], [], [], [], []

        for _ in range(self.augmentation_factor):
            for i in range(0, len(x_train), self.batch_size):
                batch_images = x_train[i:i+self.batch_size]
                batch_labels = y_train[i:i+self.batch_size]

                rotation_values = np.random.uniform(self.rotation_range[0], self.rotation_range[1], size=len(batch_images))
                multiplier_values = np.random.uniform(self.multiplier_range[0], self.multiplier_range[1], size=len(batch_images))

                for j in range(len(batch_images)):
                    image = batch_images[j]
                    
                    # Apply rotation first
                    augmented_image = self.apply_rotation(image, rotation_values[j])
                    
                    # Apply multiplier second
                    augmented_image = self.apply_multiplier(augmented_image, multiplier_values[j])

                    # Apply noise last
                    noisy_image = self.add_pixel_noise(augmented_image, self.noise_value_range)

                    new_images.append(noisy_image)
                    new_noise_values.append(noisy_image - augmented_image)
                    new_rotation_values.append(rotation_values[j])
                    new_multiplier_values.append(multiplier_values[j])
                    new_labels.append(batch_labels[j])

        # Convert lists to numpy arrays
        new_images = np.array(new_images)
        new_noise_values = np.array(new_noise_values).reshape(-1, 28, 28, 1)
        new_rotation_values = np.array(new_rotation_values).reshape(-1, 1, 1, 1)
        new_rotation_values = np.tile(new_rotation_values, (1, 28, 28, 1))  # Broadcast to match shape
        new_multiplier_values = np.array(new_multiplier_values).reshape(-1, 1, 1, 1)
        new_multiplier_values = np.tile(new_multiplier_values, (1, 28, 28, 1))  # Broadcast to match shape
        new_labels = np.array(new_labels)

        # Convert original MNIST dataset to new format ((image, 0, 0, 1), number)
        original_images = x_train
        original_noise_values = np.zeros_like(x_train)
        original_rotation_values = np.zeros_like(x_train)
        original_multiplier_values = np.ones_like(x_train)

        # Combine and shuffle datasets
        combined_images = np.concatenate((new_images, original_images), axis=0)
        combined_noise_values = np.concatenate((new_noise_values, original_noise_values), axis=0)
        combined_rotation_values = np.concatenate((new_rotation_values, original_rotation_values), axis=0)
        combined_multiplier_values = np.concatenate((new_multiplier_values, original_multiplier_values), axis=0)
        combined_labels = np.concatenate((new_labels, y_train), axis=0)

        combined_images, combined_noise_values, combined_rotation_values, combined_multiplier_values, combined_labels = shuffle(
            combined_images, combined_noise_values, combined_rotation_values, combined_multiplier_values, combined_labels
        )

        # Save the combined dataset
        np.savez(self.dataset_path, 
                 combined_images=combined_images, 
                 combined_noise_values=combined_noise_values, 
                 combined_rotation_values=combined_rotation_values, 
                 combined_multiplier_values=combined_multiplier_values, 
                 combined_labels=combined_labels, 
                 x_test=x_test, 
                 y_test=y_test)
        print(f"Dataset saved to {self.dataset_path}")

        return combined_images, combined_noise_values, combined_rotation_values, combined_multiplier_values, combined_labels, x_test, y_test

    def add_pixel_noise(self, image, noise_range):
        noisy_image = np.copy(image)
        height, width = image.shape[:2]
        for y in range(height):
            for x in range(width):
                noisy_image[y, x] += np.random.uniform(noise_range[0], noise_range[1])
        noisy_image = np.clip(noisy_image, 0, 1)  # Ensure pixel values remain in valid range [0, 1]
        return noisy_image

    def apply_rotation(self, image, angle):
        image = tf.keras.preprocessing.image.apply_affine_transform(image, theta=angle, fill_mode='nearest')
        return image

    def apply_multiplier(self, image, multiplier):
        image = np.clip(image * multiplier, 0, 1)
        return image

    def build_cnn_model(self):
        model = models.Sequential([
            layers.Input(shape=(28, 28, 4)),  # 4 channels: image + noise value + rotation + multiplier
            
            # First Convolutional Block
            layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            
            # Second Convolutional Block
            layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            
            # Third Convolutional Block
            layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),

            # Fourth Convolutional Block
            layers.Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),

            # Fully Connected Layer
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            # Second Fully Connected Layer
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            # Output Layer
            layers.Dense(10, activation='softmax')
        ])

        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def train_and_evaluate(self):
        combined_images, combined_noise_values, combined_rotation_values, combined_multiplier_values, combined_labels, x_test, y_test = self.generate_augmented_data()

        # Concatenate all channels for training
        training_data = np.concatenate((combined_images, combined_noise_values, combined_rotation_values, combined_multiplier_values), axis=-1)

        # Build and train the CNN model if train_model is True
        if self.train_model:
            model = self.build_cnn_model()
            model.fit(training_data, combined_labels, epochs=50, batch_size=1024, validation_split=0.05)

            # Prepare test set with noise value = 0, rotation = 0, multiplier = 1
            test_noise_values = np.zeros_like(x_test)
            test_rotation_values = np.zeros_like(x_test)
            test_multiplier_values = np.ones_like(x_test)
            test_data = np.concatenate((x_test, test_noise_values, test_rotation_values, test_multiplier_values), axis=-1)

            # Evaluate model on test set
            test_loss, test_acc = model.evaluate(test_data, y_test)
            print(f'Test accuracy: {test_acc}')

            # Save the trained model
            model_save_path = 'mnist_cnn_model_augmented.h5'
            model.save(model_save_path)
            print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    data_augmenter = MNISTDataAugmentation(augmentation_factor=1, noise_value_range=(0.02, 0.75), rotation_range=(-15, 15), multiplier_range=(0.8, 1.2), batch_size=128, regenerate=True, train_model=False)
    data_augmenter.train_and_evaluate()
