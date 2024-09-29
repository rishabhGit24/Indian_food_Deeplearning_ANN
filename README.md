Indian Food Classification Using Deep Learning (ANN)
This project focuses on building a deep learning model based on an Artificial Neural Network (ANN) to classify various types of Indian food. The model uses an ANN architecture to learn features from the food images and predict the food category.

Table of Contents
Project Overview
Installation
Dataset
Model Architecture
Indian cuisine consists of a rich variety of dishes, and this project aims to classify different Indian food items using a deep learning-based ANN model. While CNNs are often used for image recognition tasks, this project explores how well an ANN can perform on a similar task by flattening image data and training the network on fully connected layers.

Installation
To set up and run the project locally, follow these steps:

Clone the repository:
git clone https://github.com/rishabhGit24/Indian_food_Deeplearning_ANN.git
cd Indian_food_Deeplearning_ANN

Dataset:
The dataset used for this project comprises images of different Indian food items. You can either download a pre-existing Indian food image dataset from sources like Kaggle or build your own dataset by scraping images from the web.

Model Architecture
The ANN model in this project is a fully connected network that processes flattened image data. The architecture includes several dense layers with activation functions, dropout for regularization, and a final output layer with softmax activation for classification.

Key Components:
Dense layers: Fully connected layers
Dropout: Prevents overfitting
Activation functions: ReLU for hidden layers, softmax for the output layer
