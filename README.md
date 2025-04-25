# Hate Speech Detection Model

This project performs hate speech detection on social media texts, specifically on tweets. The model is trained to classify tweets as either **hate speech** or **non-hate speech**. It utilizes **NLP** techniques and a **deep learning model** built using Keras.

## Features
- **Text Preprocessing**: Cleaning, tokenization, and stopword removal.
- **Vectorization**: Using **TF-IDF** for converting text into numerical format.
- **Classification Model**: A **Neural Network** with multiple layers:
  - Input layer with the shape of the input data.
  - Two hidden layers with ReLU activation for learning complex features.
  - Output layer with **Sigmoid** activation for binary classification (hate speech vs non-hate speech).

## Model Architecture
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

model = Sequential()
model.add(Input(shape=(X_train.shape[1],)))
model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


