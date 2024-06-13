import pandas as pd
import numpy as np

from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split


# Load CSV data
df = pd.read_csv("training.csv")

# Convert the "Label" column to boolean values
y = np.array(df["Label"].apply(lambda x: 0 if x == "s" else 1))
X = np.array(df.drop(["EventId", "Label"], axis=1))

X_train, X_test, y_train, y_test = train_test_split(X[:10000], y[:10000], test_size=0.3)

model = Sequential()
model.add(Dense(600, input_shape=(31,), activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(400, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(100, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1, activation="sigmoid"))
sgd = SGD(lr=0.01, decay=1e-6)

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=64,
    validation_data=(X_test, y_test),
)
