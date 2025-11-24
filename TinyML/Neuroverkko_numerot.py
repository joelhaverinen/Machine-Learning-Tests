import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# -----------------------------
# 1️⃣ Ladataan MNIST-data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalisoidaan pikseliarvot 0-1 välille ja muutetaan muoto (784,)
x_train = x_train.reshape(-1, 28 * 28) / 255.0
x_test = x_test.reshape(-1, 28 * 28) / 255.0

# -----------------------------
# 2️⃣ Luodaan syvempi neuroverkko
model = models.Sequential([
    layers.Dense(256, activation='relu', input_shape=(28 * 28,)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),  # vähentää ylikoulutusta

    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),

    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),

    layers.Dense(10, activation='softmax')  # 10 luokkaa (0-9)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()  # Tulostaa mallin rakenteen

# -----------------------------
# 3️⃣ Koulutetaan malli
history = model.fit(
    x_train, y_train,
    epochs=15,          # enemmän epoceja
    batch_size=64,
    validation_split=0.1,
    verbose=2
)

# -----------------------------
# 4️⃣ Arvioidaan malli testidatalla
loss, acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {acc:.4f}")

# -----------------------------
# 5️⃣ Piirretään koulutushistoria
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# -----------------------------
# 6️⃣ Ennustetaan oma kuva
image_path = "oma_kuva.png"
img = Image.open(image_path).convert('L')
img = img.resize((28, 28))
img_array = np.array(img) / 255.0
img_array = img_array.reshape(1, 28 * 28)

prediction = model.predict(img_array)
predicted_label = np.argmax(prediction)
print(f"Malli arvaa numeroksi: {predicted_label}")

plt.imshow(img_array.reshape(28, 28), cmap='gray')
plt.title(f"Ennuste: {predicted_label}")
plt.axis('off')
plt.show()
