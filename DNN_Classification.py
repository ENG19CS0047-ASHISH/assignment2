from keras.layers import Dense
from keras.models import Sequential
import pandas as pd
from google.colab import files
uploaded = files.upload()
data = pd.read_csv('diabetes.csv.csv')
print(data)
x = data.drop("Outcome", axis=1)
y = data["Outcome"]
model = Sequential()
model.add(Dense(12, input_dim=8, activation="relu"))
model.add(Dense(12, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss="binary_crossentropy",
              optimizer="adam", metrics=["accuracy"])
model.fit(x, y, epochs=150, batch_size=10)
_, accuracy = model.evaluate(x, y)
print("Model accuracy: %.2f" % (accuracy*100))
predictions = model.predict(x)
print([round(x[0]) for x in predictions])
