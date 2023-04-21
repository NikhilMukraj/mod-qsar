import numpy as np
import ctypes
import subprocess
import os


# https://stackoverflow.com/questions/779495/access-data-in-package-subdirectory
# import os
# this_dir, this_filename = os.path.split(__file__)
# DATA_PATH = os.path.join(this_dir, "data", "data.txt")
# print(open(DATA_PATH).read())
# potentially outdated

# bash$ gcc -fPIC -shared -o ldist.so levenshtein.c
# cmd> gcc -shared -o ldist.dll levenshtein.o
# check with numpy arrays and modify to work with windows too

# needs docs

this_dir, this_filename = os.path.split(__file__)

if len([i for i in os.listdir(f'{this_dir}//src') if '.so' in i]) == 0:
    subprocess.run(['./funcs_compiler.sh'])

try:
    leven_func = ctypes.CDLL(f'{this_dir}//src//ldist.so')
except OSError:
    leven_func = ctypes.WinDLL(f'{this_dir}\\src\\ldist.dll')

levenshteinc = leven_func.levenshtein

def ldist(str1, str2):
    str1 = (ctypes.c_int * len(str1))(*str1)
    str2 = (ctypes.c_int * len(str2))(*str2)

    return levenshteinc(str1, len(str1), str2, len(str2))

try:
    onehot_func = ctypes.CDLL(f'{this_dir}//src//onehot.so')
except:
    onehot_func = ctypes.WinDLL(f'{this_dir}\\src\\onehot.dll')

onehotc = onehot_func.onehot
onehotc.restype = ctypes.POINTER(ctypes.c_int)

def onehot(n, length):
    arr = onehotc(n, length)   
    return np.ctypeslib.as_array(arr, shape=(length,))

flatSeqOneHotc = onehot_func.flatSeqOneHot
intpointer = np.ctypeslib.ndpointer(dtype=np.int32, ndim=1) 
flatSeqOneHotc.argtypes = [intpointer, intpointer]
flatSeqOneHotc.restype = ctypes.POINTER(ctypes.c_int)

def seqOneHot(seq, shape):
    seqp, shapep = seq.astype(np.int32), shape.astype(np.int32)
    arr = flatSeqOneHotc(seqp, shapep)
    arr = np.ctypeslib.as_array(arr, shape=(shape[0] * shape[1],))
    return arr.reshape(shape)

flatSeqsToOneHotc = onehot_func.flatSeqsToOneHot
flatSeqsToOneHotc.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
flatSeqsToOneHotc.restype = ctypes.POINTER(ctypes.c_int)

def seqsOneHot(seqs, shape):
    # seqs, shape = np.array(seqs, dtype=np.int32), np.array(shape, dtype=np.int32)
    seqsp = seqs.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    shapep = shape.ctypes.data_as(ctypes.POINTER(ctypes.c_int))    
    arr = flatSeqsToOneHotc(seqsp, shapep)
    arr = np.ctypeslib.as_array(arr, shape=(shape[0] * shape[1] * shape[2],))
    return arr.reshape(shape)
