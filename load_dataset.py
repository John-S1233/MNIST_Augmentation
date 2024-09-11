import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
data = np.load('augmented_mnist_dataset.npz')

# Access the data arrays
combined_images = data['combined_images']
combined_noise_values = data['combined_noise_values']
combined_multiplier_values = data['combined_multiplier_values']
combined_rotation_values = data['combined_rotation_values']
combined_labels = data['combined_labels']
x_test = data['x_test']
y_test = data['y_test']

# Example: Print the shape of the loaded arrays
print("Combined Images Shape:", combined_images.shape)
print("Combined Noise Values Shape:", combined_noise_values.shape)
print("Combined Multiplier Values Shape:", combined_multiplier_values.shape)
print("Combined Rotation Values Shape:", combined_rotation_values.shape)
print("Combined Labels Shape:", combined_labels.shape)
print("Test Images Shape:", x_test.shape)
print("Test Labels Shape:", y_test.shape)

# Filter to keep only indices where multiplier != 1, noise != 0, and rotation != 0
valid_indices = [i for i in range(len(combined_images))
                 if combined_multiplier_values[i][0][0][0] != 1 
                 and combined_noise_values[i][0][0][0] != 0
                 and combined_rotation_values[i][0][0][0] != 0]

# Function to display an image from the dataset
def show_image(index):
    if index < 0 or index >= len(valid_indices):
        print("Index out of range.")
        return

    actual_index = valid_indices[index]
    image = combined_images[actual_index].reshape(28, 28)
    noise = combined_noise_values[actual_index][0][0][0]
    multiplier = combined_multiplier_values[actual_index][0][0][0]
    rotation = combined_rotation_values[actual_index][0][0][0]
    label = combined_labels[actual_index]

    ax.clear()
    ax.imshow(image, cmap='gray')
    ax.set_title(f"Label: {label}, Noise: {noise:.2f}, Multiplier: {multiplier:.2f}, Rotation: {rotation:.2f}")
    fig.canvas.draw()

# Set initial index
index = 0

# Setup plot
fig, ax = plt.subplots()
show_image(index)

# Function to handle key presses
def on_key(event):
    global index
    if event.key == 'right':
        index = (index + 1) % len(valid_indices)
    elif event.key == 'left':
        index = (index - 1) % len(valid_indices)
    elif event.key == 'up':
        index = (index + 10) % len(valid_indices)
    elif event.key == 'down':
        index = (index - 10) % len(valid_indices)
    
    show_image(index)

# Connect the key press event to the figure
fig.canvas.mpl_connect('key_press_event', on_key)
plt.show()
