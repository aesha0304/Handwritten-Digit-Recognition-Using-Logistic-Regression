
# Handwritten Digit Recognition using Logistic Regression

## Project Overview

This project implements a multi-class logistic regression model for handwritten digit recognition using a one-vs-all (OvA) approach. Built from scratch, the model is trained on the MNIST dataset, allowing for the classification of digits (0–9) based on pixel intensity values. The project includes model training, evaluation on test samples, and predictions on custom input images.

## Table of Contents
- [Project Features](#project-features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Future Enhancements](#future-enhancements)
- [Contributors](#contributors)

## Project Features
- Logistic regression model for multi-class classification using one-vs-all strategy.
- Custom image prediction functionality for real-world applications.
- Evaluation on the MNIST dataset with saved model parameters for reuse.
- Visualizations of sample predictions for interactive analysis.

## Dataset
This project uses the [MNIST dataset](https://www.openml.org/d/554), a well-known database of handwritten digits often used as a benchmark in image recognition tasks. The dataset contains thousands of labeled images of digits 0–9, each sized 28x28 pixels.

## Installation

1. Clone this repository:
   git clone https://github.com/yourusername/handwritten-digit-recognition.git
2. Navigate into the project directory:
   cd handwritten-digit-recognition
3. Install the required dependencies:
   pip install -r requirements.txt

## Usage

### Training the Model
To train the model from scratch:
   python src/main.py --train
   This will load the MNIST data, preprocess it, and train the model with logistic regression.

### Visualizing Sample Predictions
To visualize predictions on random test samples:
   python src/main.py --evaluate


### Predicting Custom Images
You can input custom images for prediction. Place your image files in the `/images` directory and run:

   python src/main.py --predict --image_path images/your_image.png


## Project Structure

```
handwritten-digit-recognition/
│
├── data/                      # Directory for MNIST dataset
│
├── models/                    # Directory for saved models
│   └── logistic_regression_model.pkl  # Serialized logistic regression model
│
├── src/                       # Source code
│   ├── main.py                # Main script for training and predictions
│   ├── utils.py               # Data loading and visualization functions
│   └── model.py               # Logistic regression model classes
│
├── images/                    # Directory for custom images
│
├── results/                   # Directory for evaluation results
│   └── evaluation_metrics.txt # Text file for accuracy and metrics
│
└── README.md                  # Project overview and instructions
```

## Future Enhancements
- Extend the model to recognize letters and sentences.
- Integrate with more advanced machine learning models, like CNNs, to improve accuracy.
- Deploy the model in real-time applications for live handwritten character recognition.


---

**License**
MIT License
```

This README provides a comprehensive project description, installation instructions, usage examples, and a clear structure overview for GitHub. Let me know if you need any additional adjustments!
