import numpy as np
from c_wrapper import seqOneHot, seqsOneHot
import h5py
import hdf5plugin
import sys
import os


f = h5py.File(f'{sys.argv[1]}_aug_unencoded_data.jld')
unencoded_data = np.transpose(np.array(f['features']))

# check shape first and then run rest of program

seqs_shape = np.array([len(unencoded_data[0]), np.max(unencoded_data)+1], dtype=np.int32)
unencoded_data = np.array(unencoded_data, dtype=np.int32)

encoded_seqs = np.array([seqOneHot(i, seqs_shape) for i in unencoded_data])

# encoded_seqs = seqsOneHot(unencoded_data, seqs_shape) 
# need to ensure is of dtype=np.int32 otherwise data is mangled

np.save(f'{sys.argv[1]}_encoded_seqs.npy', encoded_seqs)

f = h5py.File(f'{sys.argv[1]}_aug_activity.jld')
activity = np.transpose(np.array(f['activity']))

np.save(f'{sys.argv[1]}_activity.npy', activity)
