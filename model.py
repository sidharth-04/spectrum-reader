import numpy as np
from keras import models
from keras import layers
import random
import matplotlib.pyplot as plt

train_features = np.load("processed/spectradata.npy")
train_labels = np.load("processed/labeldata.npy")
temp = list(zip(train_features, train_labels))
random.shuffle(temp)
train_features, train_labels = zip(*temp)

def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(train_features.shape[1])))
    model.add(layers.Dense(64,activation='relu'))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

model = build_model()
history = model.fit(train_features, train_labels, validation_split=0.2, epochs=10, batch_size=64)
mae_history = history.history["val_mean_absolute_error"]
plt.plot(range(1, len(mae_history)+1), mae_history)
plt.xlabel("Epochs")
plt.ylabel("Validation MAE")
plt.show()
