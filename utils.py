import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

def load_mnist_data():
    mnist = fetch_openml('mnist_784', version=1)
    X = mnist.data.values / 255.0  
    y = mnist.target.astype(int)  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)


def visualize_image(image_data):
    image = image_data.reshape(28, 28)
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()
