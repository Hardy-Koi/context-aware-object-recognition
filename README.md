# Context-Aware Object Recognition

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)
![License](https://img.shields.io/badge/License-MIT-green)

A simple **computer vision project** that classifies common objects found in a student's room using a **ResNet18 convolutional neural network**.

The project demonstrates how transfer learning can be applied to recognize everyday objects such as beds, books, computers, cars, and bikes.

------

# Project Overview

This project explores whether machine learning models can recognize objects commonly seen in a student's room.

Inspired by the development of modern computer vision technologies, the project trains a CNN-based classifier to distinguish between several daily objects.

The system performs image classification on the following categories:

- Bed
- Book
- Computer
- Car
- Bike

A **pretrained ResNet18 model** is used and fine-tuned for this task.

------

# Dataset

The dataset used in this project is:

**Daily Objects Around The World Dataset**

To keep the dataset balanced, the experiment uses:

```
130 images per category
```

Data augmentation techniques were applied to increase dataset diversity, including:

- Random flipping
- Rotation
- Brightness adjustment
- Cropping

These transformations help reduce **overfitting** and improve the model's ability to generalize to unseen images.

The dataset is split into:

```
Training set: 80%
Validation set: 20%
```

------

# Model

The classification model is based on **ResNet18**, a convolutional neural network architecture widely used in image recognition tasks.

Transfer learning is applied by replacing the final fully connected layer:

```
model.fc = nn.Linear(model.fc.in_features, num_classes)
```

In this project:

```
num_classes = 5
```

The model is trained to map extracted image features to the five object categories.

------

# Installation

First clone the repository:

```bash
git clone https://github.com/yourusername/student-room-object-classification.git
cd student-room-object-classification
```

Create a virtual environment (recommended):

```bash
python -m venv venv
```

Activate the environment:

Mac/Linux

```bash
source venv/bin/activate
```

Windows

```bash
venv\Scripts\activate
```

Install required packages:

```bash
pip install torch torchvision matplotlib numpy scikit-learn
```

------

# Usage

### Train the model

Run the training script:

```bash
python train.py
```

The script will:

1. Load the dataset
2. Apply data augmentation
3. Train the ResNet18 model
4. Evaluate on the validation set

------

### Predict a new image

```bash
python predict.py --image example.jpg
```

Example output:

```
Prediction: computer
Confidence: 0.94
```

------

# Project Structure

```
student-room-object-classification
│
├── train.py
├── predict.py
├── formatted_dataset
│   ├── bed
│   ├── book
│   ├── computer
│   ├── car
│   └── bike
│
├── model
│   └── resnet18_model.pth
│
├── notebook
│   └── main.ipynb
│
└── README.md
```

------

# Results

The trained model achieved strong classification performance.

According to the confusion matrix in the experiment, most predictions fall on the diagonal, meaning the model correctly identifies most samples.

Example observations:

- Book, car, and computer images were classified correctly.
- Only a small number of misclassifications occurred between **bed** and **bike**.

The validation accuracy reached approximately:

```
97%
```

The training and validation curves show that the model quickly learns the dataset features and stabilizes after several epochs.

------

# Future Work

Several improvements can be made to enhance this project:

- Increase dataset size
- Use more advanced architectures (ResNet50 / EfficientNet)
- Evaluate using additional metrics (F1 score, recall)
- Improve generalization to new scenes

------

# References

Daily Objects Around The World Dataset
ResNet: Deep Residual Learning for Image Recognition



