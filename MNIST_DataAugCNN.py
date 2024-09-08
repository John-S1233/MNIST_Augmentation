import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from sklearn.utils import shuffle
from MNIST_Generator import get_training_pairs_batch
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
    def __init__(self, generated_dataset_length, noise_value_range=(0.1, 0.5), rotation_range=(-10, 10), multiplier_range=(0.5, 1.5), batch_size=128, dataset_path='augmented_mnist_dataset.npz', regenerate=False, train_model=True):
        self.generated_dataset_length = generated_dataset_length
        self.noise_value_range = noise_value_range
        self.rotation_range = rotation_range
        self.multiplier_range = multiplier_range
        self.batch_size = batch_size
        self.dataset_path = dataset_path
        self.regenerate = regenerate
        self.train_model = train_model  # Add this line to allow control over training

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

        # Generate new dataset in batches
        new_images, new_noise_values, new_rotation_values, new_multiplier_values, new_labels = [], [], [], [], []
        num_batches = self.generated_dataset_length // self.batch_size

        for i in range(num_batches):
            digits = np.random.randint(0, 10, size=self.batch_size)
            noise_values = np.random.uniform(self.noise_value_range[0], self.noise_value_range[1], size=self.batch_size)
            batch_images, batch_rotation_values, batch_multiplier_values = get_training_pairs_batch(digits, noise_values)
            new_images.extend(batch_images)
            new_noise_values.extend(noise_values)
            new_rotation_values.extend(batch_rotation_values)
            new_multiplier_values.extend(batch_multiplier_values)
            new_labels.extend(digits)
            print(f"Generated batch {i+1}/{num_batches}")

        # Convert lists to numpy arrays
        new_images = np.array(new_images)
        new_noise_values = np.array(new_noise_values).reshape(-1, 1, 1, 1)
        new_noise_values = np.tile(new_noise_values, (1, 28, 28, 1))
        new_rotation_values = np.array(new_rotation_values).reshape(-1, 28, 28, 1)
        new_multiplier_values = np.array(new_multiplier_values).reshape(-1, 28, 28, 1)
        new_labels = np.array(new_labels)

        # Convert original MNIST dataset to new format ((image, 0, 0, 1), number)
        original_images = x_train
        original_noise_values = np.zeros_like(x_train)
        original_rotation_values = np.zeros_like(x_train)
        original_multiplier_values = np.ones_like(x_train)
        original_labels = y_train

        # Combine and shuffle datasets
        combined_images = np.concatenate((new_images, original_images), axis=0)
        combined_noise_values = np.concatenate((new_noise_values, original_noise_values), axis=0)
        combined_rotation_values = np.concatenate((new_rotation_values, original_rotation_values), axis=0)
        combined_multiplier_values = np.concatenate((new_multiplier_values, original_multiplier_values), axis=0)
        combined_labels = np.concatenate((new_labels, original_labels), axis=0)

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

        if self.train_model:  # Check if training should be performed
            # Build and train the CNN model
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
        else:
            print("Model training skipped.")

if __name__ == "__main__":
    data_augmenter = MNISTDataAugmentation(generated_dataset_length=500, noise_value_range=(0.02, 0.75), rotation_range=(-15, 15), multiplier_range=(0.8, 1.2), batch_size=128, regenerate=True, train_model=False)
    data_augmenter.train_and_evaluate()
