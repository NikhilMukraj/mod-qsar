import numpy as np
from c_wrapper import seqOneHot, seqsOneHot
import h5py
import hdf5plugin
import sys
import os


# df = pd.read_csv(f'{os.getcwd()}//filtered_dataset.csv')

# vocab = pd.read_csv(f'{os.getcwd()}//vocab.csv')['tokens'].to_list()

# tokenizer = {i : n for n, i in enumerate(vocab)}
# reverse_tokenizer = {value: key for key, value in tokenizer.items()}
# convert_back = lambda x: ''.join(reverse_tokenizer.get(np.argmax(i)-1, '') for i in x)

# clean up code so its not messy as hell

# Main.eval('using JLD; unencoded_data = load("unencoded_data.jld"); activity = load("augmented_activity.jld")')
# unencoded_data = Main.unencoded_data
# activity = Main.activity
# unencoded_data = np.array(unencoded_data['features'])
# activity = np.array(activity['activity'])

####################### needs to be implemented somewhere ########################
# unencoded_data = np.save(f'{os.getcwd()}//unencoded_data.npy', unencoded_data) #
# activity = np.save(f'{os.getcwd()}//activity.npy', activity)                   # 

# unencoded_data = np.load(f'{os.getcwd()}//unencoded_data.npy')
# activity = np.load(f'{os.getcwd()}//activity.npy')

# encoded_data = np.zeros((*unencoded_data.shape, np.max(unencoded_data)+1))

# for i in range(unencoded_data.shape[0]):
#     encoded_data[i] = to_categorical(unencoded_data[i], np.max(unencoded_data)+1)

#     if i % 100 == 0:
#         print(f'at {i}/{unencoded_data.shape[0]}')

# encoded_data = np.array(encoded_data)

'''read from jld'''

f = h5py.File(f'{os.getcwd()}//{sys.argv[1]}_aug_unencoded_data.jld')
unencoded_data = np.transpose(np.array(f['features']))

# check shape first and then run rest of program

seqs_shape = np.array([len(unencoded_data[0]), np.max(unencoded_data)+1], dtype=np.int32)
unencoded_data = np.array(unencoded_data, dtype=np.int32)

encoded_seqs = np.array([seqOneHot(i, seqs_shape) for i in unencoded_data])

# encoded_seqs = seqsOneHot(unencoded_data, seqs_shape) 
# need to ensure is of dtype=np.int32 otherwise data is mangled

np.save(f'{os.getcwd()}//{sys.argv[1]}_encoded_seqs.npy', encoded_seqs)

f = h5py.File(f'{os.getcwd()}//{sys.argv[1]}_aug_activity.jld')
activity = np.transpose(np.array(f['activity']))

np.save(f'{os.getcwd()}//{sys.argv[1]}_activity.npy', activity)
