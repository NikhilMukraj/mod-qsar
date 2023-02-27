from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
# from tokenization import return_tokens
# from julia import Main
import h5py
import hdf5plugin

# pip install tensorflow-gpu==2.3
# pip install pandas==1.3.4
# just in case gpu crashes this appears to be correct env 

# X = np.load(f'{os.getcwd()}\\encoded_seqs.npy')
# Y = np.load(f'{os.getcwd()}\\activity.npy')

# n = 60_000
# X, Y = X[:n], Y[:n]

f = h5py.File(f'{os.getcwd()}//augmented_data.jld', 'r')
X = np.array(f['X'])
X = np.transpose(X)
Y = np.array(f['Y'])
Y = np.transpose(Y)

# vocab = pd.read_csv(f'{os.getcwd()}\\vocab.csv')
# tokenizer = {i : n for n, i in enumerate(vocab['tokens'].to_list())}
# reverse_tokenizer = {value: key for key, value in tokenizer.items()}
# convert_back = lambda x: ''.join(reverse_tokenizer.get(np.argmax(i)-1, '') for i in x)

trainX, testX, trainY, testY = train_test_split(X, Y, test_size=.1)
trainX.shape, testX.shape, trainY.shape, testY.shape

opt = Adam(.0001)
input_shape = trainX[0].shape

# potentially needs more lstm layers
# https://stackoverflow.com/questions/56575579/how-to-increase-accuracy-of-lstm-training

model = Sequential()
model.add(LSTM(128, input_shape=input_shape, activation="tanh", return_sequences=True, dropout=0.2)) 
model.add(LSTM(128, activation="tanh", return_sequences=True, dropout=0.2))
model.add(LSTM(64, activation="tanh", dropout=0.2))
model.add(Dense(32, activation="tanh"))
model.add(Dropout(0.2))
model.add(Dense(16, activation="tanh"))
model.add(Dropout(0.2))
model.add(Dense(2, activation="softmax"))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

history = model.fit(trainX, trainY, batch_size=32, epochs=100, validation_data=(testX, testY))
hist_df = pd.DataFrame(history.history) 
hist_df.to_csv(f'{os.getcwd()}//model_history.csv')
model.save(f'{os.getcwd()}//rnn_model')

# restructure file system using __init__.py before or while using aws

# should check accuracy of the model on one string and its equalivents 
# and see if it predicts the same thing since the structure is the same 
# but the string representation is different

# https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/AccessingInstancesLinux.html
# https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/connection-prereqs.html
