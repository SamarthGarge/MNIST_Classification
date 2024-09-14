import tensorflow as tf
import numpy as np
from PIL import Image

mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images, test_images = train_images / 255.0, test_images / 255.0

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer='adam', loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)

img = Image.open(r"C:\Users\Samarth\Downloads\mnist_img2.png")
img = img.convert('L')
img = img.resize((28,28))
img

img_array = np.array(img)/225.0

img_array = np.expand_dims(img_array,axis=0)
predictions = model.predict(img_array)
predictions

probabilities = tf.nn.softmax(predictions[0])
predicted_class = np.argmax(probabilities)
predicted_class

model.save('mnist_model.h5')
print("Model saved to mnist_model.h5")