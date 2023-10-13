from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow import config
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import sys


print(config.list_physical_devices('GPU'))

# sudo cp /usr/lib/python3/dist-packages/tensorflow/libcudnn* /usr/lib/x86_64-linux-gnu/

if len(sys.argv) < 5:
    print('Too few args...')
    sys.exit(1)

X = np.load(sys.argv[1])
Y = np.load(sys.argv[2])

trainX, testX, trainY, testY = train_test_split(X, Y, test_size=.1)
print(trainX.shape, testX.shape, trainY.shape, testY.shape)

opt = Adam(.0001)
input_shape = trainX[0].shape

model = Sequential()
model.add(LSTM(514, input_shape=input_shape, activation="tanh", return_sequences=True, dropout=0.2)) 
model.add(LSTM(256, activation="tanh", return_sequences=True, dropout=0.2))
model.add(LSTM(128, activation="tanh", dropout=0.2))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(32, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(16, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(2, activation="softmax"))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

history = model.fit(trainX, trainY, batch_size=32, epochs=int(sys.argv[3]), validation_data=(testX, testY), verbose=1)

# train_preds = model.predict(trainX)
# test_preds = model.predict(testX)

hist_df = pd.DataFrame(history.history)

hist_df.to_csv(f'{sys.argv[4]}_model_history.csv', index=False)
model.save(f'{sys.argv[4]}_rnn_model.h5')

# pd.DataFrame(train_preds).to_csv(f'{os.getcwd()}//train_preds.csv', index=False)
# pd.DataFrame(test_preds).to_csv(f'{os.getcwd()}//test_preds.csv', index=False)

if len(sys.argv) >= 7:
    np.save(sys.argv[5], testX)
    np.save(sys.argv[6], testY)
