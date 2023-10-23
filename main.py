import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import numpy as np

# Data preprocessing
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_generator = datagen.flow_from_directory(
    directory='C:/Users/thaku/OneDrive/Desktop/data/data',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    subset='training')

validation_generator = datagen.flow_from_directory(
    directory='C:/Users/thaku/OneDrive/Desktop/data/data',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    subset='validation')

# Building the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# Train the model
history = model.fit(train_generator, validation_data=validation_generator, epochs=1)

# Save the model in .h5 format
model.save("model.h5")

# Preprocess new images
def preprocess_new_image(img_path):
    # Load the image
    img = image.load_img(img_path, target_size=(150, 150))
    # Convert the image to a numpy array and scale the pixel values
    x = image.img_to_array(img) / 255.0
    # Expand the dimensions to match the shape your model expects
    x = np.expand_dims(x, axis=0)
    return x

# Load and predict on a new image
img_path = r'C:\Users\thaku\OneDrive\Desktop\data\no1.jpg'
img = preprocess_new_image(img_path)
prediction = model.predict(img)

# Get the class with the highest probability
class_names = ['glioma', 'meningioma', 'noutumor', 'pituitary']
predicted_class = class_names[np.argmax(prediction)]

print(f"The image belongs to the class: {predicted_class}.")
