import mnist_reader
import numpy as np
from sklearn.utils import shuffle
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import Adam, SGD

# check wether hardware optimization is working
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

author = "Naut"

if author == "Amir":
    X_train, y_train = mnist_reader.load_mnist('/Users/amir0/Documents/MachineLearningPractical/Tijn_Amir_Naut/data', kind='train')
    X_test, y_test = mnist_reader.load_mnist('/Users/amir0/Documents/MachineLearningPractical/Tijn_Amir_Naut/data', kind='t10k')

if author == "Naut":
    X_train, y_train = mnist_reader.load_mnist('D:\\studie\\machine learning practical\\Tijn_Amir_Naut\\data', kind='train')
    X_test, y_test = mnist_reader.load_mnist('D:\\studie\\machine learning practical\\Tijn_Amir_Naut\\data', kind='t10k')


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


X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


# Flatten
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)



# SVM model
print("training SVM")
def buildSVC(kernel_choice='linear', C=3):
    svm_model = SVC(kernel=kernel_choice, C=C)
    return svm_model

SVC_params = {
    'kernel': ['linear', 'rbf', 'poly'],
    'C': [-3, -1, 1, 3, 5, 7]
}

function_svc = buildSVC()

kfold_svc = StratifiedKFold(n_splits=6)
svc_random_search = RandomizedSearchCV(estimator=function_svc, param_distributions=SVC_params, n_iter=15, cv=kfold_svc, scoring='accuracy', n_jobs=-1)

svc_results = svc_random_search.fit(X_train_flat, y_train)
# svm_model.fit(X_train_flat, y_train)

y_test_pred = svc_results.best_estimator_.predict(X_test_flat)
svm_test_accuracy = accuracy_score(y_test, y_test_pred)

print("SVM Accuracy: ", svm_test_accuracy)

# Test set
y_test_pred = svc_results.best_estimator_.predict(X_test_flat)

svm_test_accuracy = accuracy_score(y_test, y_test_pred)
print("SVM Test Accuracy: ", svm_test_accuracy)
print("svm best params: ", svc_results.best_params_)
print(svc_results)
print()

# CNN model
def buildModel(learning_rate = 0.001, layer_nodes=32, activation_function='relu', optimization_algorithm='adam'):
    model = Sequential([
        Conv2D(layer_nodes, kernel_size=(3, 3), activation=activation_function, input_shape=(28, 28, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Conv2D(2 * layer_nodes, (3, 3), activation=activation_function),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(4 * layer_nodes, activation=activation_function),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])

    model.summary()

    # allow learning rate to be set for the optimizer
    if optimization_algorithm == 'adam':
        optimizer_algo = Adam(learning_rate=learning_rate)
    elif optimization_algorithm == 'sgd':
        optimizer_algo = SGD(learning_rate=learning_rate)

    # Compile the model
    model.compile(loss='sparse_categorical_crossentropy',
                optimizer=optimizer_algo,
                metrics=['accuracy'])
    
    return model

epochs = 10
    
keras_classifier = KerasClassifier(build_fn=buildModel, epochs=10, batch_size=32, verbose=0)

params_cnn = {
    'learning_rate': [0.001, 0.005, 0.0001, 0.0005, 0.002],
    'layer_nodes': [8, 16, 32, 64],
    'activation_function': ['relu', 'tanh'],
    'optimization_algorithm': ['adam', 'sgd']
}

kfold_cnn = StratifiedKFold(n_splits=6)

print("training CNN")
random_search = RandomizedSearchCV(estimator=keras_classifier, param_distributions=params_cnn, n_iter=15, cv=kfold_cnn, scoring='accuracy', n_jobs=-1)

# Train the model with hyperparameter tuning
cnn_results = random_search.fit(X_train.reshape, y_train)



print(cnn_results)
print("cnn best params: ", cnn_results.best_params_)


# history = model.fit(X_train.reshape(-1, 28, 28, 1), y_train, epochs=epochs, testidation_data=(X_test.reshape(-1, 28, 28, 1), y_test))

# Etestuate the model
test_loss, test_accuracy = cnn_results.best_estimator_.etestuate(X_test.reshape(-1, 28, 28, 1), y_test)
print("CNN Test Accuracy: ", test_accuracy)

# Visualize training results
acc = cnn_results.best_estimator_.history['accuracy']
test_acc = cnn_results.best_estimator_.history['test_accuracy']

loss = cnn_results.best_estimator_.history['loss']
test_loss = cnn_results.best_estimator_.history['test_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, test_acc, label='testidation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and testidation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, test_loss, label='test Loss')
plt.legend(loc='upper right')
plt.title('Training and test Loss')
plt.show()