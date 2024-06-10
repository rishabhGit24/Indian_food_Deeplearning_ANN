import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense

# Load the dataset
cancer = pd.read_csv("R:/DP/ANN/indian_food.csv")
cancer.head()

# Separate features and label
X = cancer.drop('state', axis=1)
y = cancer['state']

# Identify categorical features
categorical_features = X.select_dtypes(include=['object']).columns

# Encode categorical features in X
label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Encode the target variable y if it's categorical
y_encoder = LabelEncoder()
y = y_encoder.fit_transform(y)

# Feature scaling for values in CSV - MinMax scaling
min_max_scaler = MinMaxScaler()
X_scale = min_max_scaler.fit_transform(X)

# Split the data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X_scale, y, test_size=0.20, random_state=0)

# Deep Learning
model = Sequential()

# Hidden layers
model.add(Dense(48, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(48, activation='relu'))
model.add(Dense(88, activation='relu'))
model.add(Dense(88, activation='relu'))
model.add(Dense(102, activation='relu'))
model.add(Dense(102, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dense(2048, activation='relu'))
model.add(Dense(2048, activation='relu'))


# Output layer
model.add(Dense(1, activation='sigmoid'))

# Compilation
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=100)

# SGD with momentum
opt = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=32, epochs=10)

# Predict
y_pred = model.predict(X_test)

# Convert the predicted real numbers to 0 and 1
y_pred = np.where(y_pred > 0.5, 1, 0)

# Stack predictions with true labels
np.column_stack((y_pred, y_test))

# Evaluate the model
from sklearn.metrics import confusion_matrix, accuracy_score
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
