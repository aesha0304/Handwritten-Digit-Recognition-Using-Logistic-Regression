from utils import load_mnist_data, visualize_image
from model import LogisticRegressionOvA
import numpy as np
import os
import pickle
import random
from PIL import Image  

X_train, y_train, X_test, y_test = load_mnist_data()

print("Visualizing a sample image from the training set:")
visualize_image(X_train[0])

model_filename = "logistic_regression_model.pkl"

def calculate_accuracy(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = np.mean(predictions == y_test) * 100  
    print(f"Model Accuracy: {accuracy:.2f}%")
    return accuracy

#if os.path.exists(model_filename):
 #   print("Loading saved model parameters...")
  #  with open(model_filename, "rb") as file:
   #     model = pickle.load(file)
#else:
model = LogisticRegressionOvA(num_classes=10, learning_rate=0.01, n_iterations=500) 
print("Training the multi-class logistic regression model...")
model.train(X_train, y_train)
print("Training complete.\n")

with open(model_filename, "wb") as file:
    pickle.dump(model, file)

accuracy = calculate_accuracy(model, X_test, y_test)

def evaluate_random_samples(model, X_test, y_test, n_samples=5):
    print("Evaluating model on random test samples...")
    indices = random.sample(range(len(X_test)), n_samples)  
    for idx in indices:
        predicted_label = model.predict(X_test[idx].reshape(1, -1))[0]
        true_label = y_test[idx]
        print(f"Test sample index: {idx}, Predicted label: {predicted_label}, True label: {true_label}")
        visualize_image(X_test[idx]) 

evaluate_random_samples(model, X_test, y_test, n_samples=5)


def predict_custom_image(image_path):
    image = Image.open(image_path).convert("L")  
    image = image.resize((28, 28))  
    image_data = np.array(image) / 255.0 
    image_data = image_data.reshape(1, -1)

    predicted_label = model.predict(image_data)[0]
    print(f"Predicted label for the image '{image_path}': {predicted_label}")

    visualize_image(image_data.reshape(28, 28))

predict_custom_image('image.png')
predict_custom_image('image2.png')
predict_custom_image('image3.png')
predict_custom_image('image4.jpg')
predict_custom_image('image5.png')
