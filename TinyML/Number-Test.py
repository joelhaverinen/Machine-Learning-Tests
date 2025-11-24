import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# -----------------------------
# 1️⃣ Ladataan MNIST-data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalisoidaan pikseliarvot
x_train = x_train.reshape(-1, 28 * 28) / 255.0
x_test = x_test.reshape(-1, 28 * 28) / 255.0

# -----------------------------
# 2️⃣ Luodaan pieni neuroverkko
model = models.Sequential([
    layers.Dense(32, activation='relu', input_shape=(28 * 28,)),  # vain 32 neuronia
    layers.Dense(10, activation='softmax')  # 10 luokkaa (0-9)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# -----------------------------
# 3️⃣ Koulutetaan malli
model.fit(x_train, y_train, epochs=5, batch_size=32)

# -----------------------------
# 4️⃣ Arvioidaan malli testidatalla
loss, acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {acc:.2f}")

# -----------------------------


# -----------------------------
# 6️⃣ Ennustetaan oma kuva
# Korvaa "oma_kuva.png" haluamallasi kuvatiedostolla
image_path = "oma_kuva.png"

# Ladataan ja muokataan kuva sopivaksi
img = Image.open(image_path).convert('L')  # Harmaasävy
img = img.resize((28, 28))  # Muutetaan 28x28
img_array = np.array(img) / 255.0  # Normalisoidaan
img_array = img_array.reshape(1, 28 * 28)  # Muoto 1x784

# Ennustetaan
prediction = model.predict(img_array)
predicted_label = np.argmax(prediction)
print(f"Malli arvaa numeroksi: {predicted_label}")

# Piirretään oma kuva
plt.imshow(img_array.reshape(28, 28), cmap='gray')
plt.title(f"Ennuste: {predicted_label}")
plt.axis('off')
plt.show()
