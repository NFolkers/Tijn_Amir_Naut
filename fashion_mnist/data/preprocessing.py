import mnist_reader
import numpy as np
from sklearn.utils import shuffle
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


X_train, y_train = mnist_reader.load_mnist('/Users/amir0/Documents/MachineLearningPractical/Tijn_Amir_Naut/data', kind='train')
X_test, y_test = mnist_reader.load_mnist('/Users/amir0/Documents/MachineLearningPractical/Tijn_Amir_Naut/data', kind='t10k')


# Shuffle the data
X_train, y_train = shuffle(X_train, y_train, random_state=42)


# Normalize the data
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,  # Rotate images in the range
    width_shift_range=0.1, #  Shift images horizontally
    height_shift_range=0.1, # Shift images vertically
    shear_range=0.1, # Shear images 
    zoom_range=0.1, # Zoom image 
    horizontal_flip=True, # Flip images horizontally
    fill_mode='nearest' # Fill the new pixels
)

# Fitting the ImageDataGenerator
datagen.fit(X_train.reshape(-1, 28, 28, 1))


X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


# Flatten
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_val_flat = X_val.reshape(X_val.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# SVM model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_flat, y_train)
y_val_pred = svm_model.predict(X_val_flat)
svm_val_accuracy = accuracy_score(y_val, y_val_pred)

print("SVM Accuracy: ", svm_val_accuracy)

# Test set
y_test_pred = svm_model.predict(X_test_flat)

svm_test_accuracy = accuracy_score(y_test, y_test_pred)
print("SVM Test Accuracy: ", svm_test_accuracy)

# CNN model
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model.summary()

# Compile the model
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

epochs = 10

# Train the model
history = model.fit(X_train.reshape(-1, 28, 28, 1), y_train, epochs=epochs, validation_data=(X_val.reshape(-1, 28, 28, 1), y_val))

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test.reshape(-1, 28, 28, 1), y_test)
print("CNN Test Accuracy: ", test_accuracy)

# Visualize training results
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()