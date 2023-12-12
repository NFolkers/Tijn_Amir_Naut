import mnist_reader
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

X_train, y_train = mnist_reader.load_mnist('/Users/amir0/Documents/MachineLearningPractical/Tijn_Amir_Naut/data', kind='train')
X_test, y_test = mnist_reader.load_mnist('/Users/amir0/Documents/MachineLearningPractical/Tijn_Amir_Naut/data', kind='t10k')

# pca = PCA(n_components=100)
# X_train_pca = pca.fit_transform(X_train)

# np.random.seed(0)  
# indices = np.random.choice(X_train_pca.shape[0], 10000, replace=False)
# X_subset = X_train_pca[indices]
# y_subset = y_train[indices]

# # Apply t-SNE to the data
# tsne = TSNE(n_components=2, random_state=0)
# X_tsne = tsne.fit_transform(X_subset)


# plt.figure(figsize=(12, 8))
# for i in range(10):
#     indices = y_subset == i
#     plt.scatter(X_tsne[indices, 0], X_tsne[indices, 1], label=str(i))
# plt.legend()
# plt.title('t-SNE visualization of Fashion MNIST data')
# plt.xlabel('t-SNE feature 1')
# plt.ylabel('t-SNE feature 2')
# plt.show()


# Plot 1: Histogram of Pixel Intensities
# Flatten the pixel values of all images
pixels = X_train.flatten()
plt.figure(figsize=(10, 6))
plt.hist(pixels, bins=10, color='gray', alpha=0.7)
plt.title('Histogram of Pixel Intensities')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.show()

# Plot 2: Class Distribution Bar Chart
classes, counts = np.unique(y_train, return_counts=True)
plt.figure(figsize=(10, 6))
plt.bar(classes, counts, tick_label=classes, color='skyblue', alpha=0.7)
plt.title('Class Distribution in Fashion MNIST Dataset')
plt.xlabel('Class')
plt.ylabel('Number of samples')
plt.show()