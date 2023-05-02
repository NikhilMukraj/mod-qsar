#!/bin/bash


gcc -fPIC -shared -o "$1/src/ldist.so" "$1/src/levenshtein.c"
gcc -fPIC -shared -o "$1/src/onehot.so" "$1/src/onehot.c"