import ctypes
import os


# bash$ gcc -fPIC -shared -o ldist.so levenshtein.c
# cmd> gcc -shared -o ldist.dll levenshtein.o
# check with numpy arrays and modify to work with windows too

try:
    cfunc = ctypes.CDLL(os.getcwd() + '//ldist.so')
except OSError:
    cfunc = ctypes.WinDLL(os.getcwd() + '\\ldist.dll')

levenshtein = cfunc.levenshtein

def ldist(str1, str2):
    str1 = (ctypes.c_int * len(string1))(*str1)
    str2 = (ctypes.c_int * len(string2))(*str2)

    return levenshtein(str1, len(str1), str2, len(str2))

string1 = [4, 5, 1, 2, 3]
string2 = [1, 2, 3]

print(ldist(string1, string2))