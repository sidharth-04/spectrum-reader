import numpy as np
from keras import models
from keras import layers
import random
import matplotlib.pyplot as plt

dir_loc = './processed/'

features = np.load(dir_loc+"spectradata.npy")
labels = np.load(dir_loc+"labeldata.npy")
temp = list(zip(features, labels))
random.shuffle(temp)
features, labels = zip(*temp)
features, labels = np.array(features), np.array(labels)

train_features = features[100:]
train_labels = labels[100:]

mean = train_features.mean()
train_features -= mean
std = train_features.std()
train_features /= std

test_features = features[:100]
test_labels = labels[:100]

test_features -= mean
test_features /= std

def build_model_overfitting():
    model = models.Sequential()
    model.add(layers.Dense(14, activation='relu', input_shape=(train_features.shape[1],)))
    model.add(layers.Dense(14,activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

model = build_model()
history = model.fit(
    train_features,
    train_labels,
    validation_split=0.1,
    epochs=60,
    batch_size=64)

val_mae_history = history.history["val_mae"]
mae_history = history.history["mae"]
plt.plot(range(10, len(mae_history)+1), mae_history[9:], 'b-', label="training error")
plt.plot(range(10, len(val_mae_history)+1), val_mae_history[9:], 'r-',  label="validation error")
leg = plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Error")
plt.show()

model.evaluate(test_features, test_labels)
