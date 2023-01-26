#include <stdlib.h>
#include <stdio.h>


// take a few args, assume its already padd
// vocab_len : length of vocab (72)
// arr of arr of nums : [[0, 0, ..., 1, 4,], [0, 0, ... 2, 3]] 
//
// take this and then create arrays of size
// vocab_len and set arr[num] to 1 
//
// repeat for every arr in nums
                                           
int *onehot(int n, int length) {
    int *arr = calloc(length, sizeof(int));
    arr[n] = 1;
    return arr;
}

int **seqonehot(int *seq, int *shape) {
	int **arr = malloc(shape[0] * sizeof(int*));
	for (int i = 0; i < shape[0]; i++) {
		arr[i] = onehot(seq[i], shape[1]);
	}

	return arr;
}

int *flatSeqOneHot(int *seq, int *shape) {
    int *arr = calloc(shape[0] * shape[1], sizeof(int*));

    for (int i = 0; i < shape[0]; i++) {
        arr[shape[1] * i + seq[i]] = 1;
    }

    return arr;
}

int ***arrOfOneHots(int **seqs, int *shape) {
	int ***encodedArr = malloc(shape[0] * sizeof(int**));
    int inner_shape[] = {shape[1], shape[2]};

	for (int i = 0; i < shape[0]; i++) {
		encodedArr[i] = seqonehot(seqs[i], inner_shape);
	}

	return encodedArr;
}

int *flatArrOfOneHots(int **seqs, int *shape) {
    int *arr = calloc(shape[0] * shape[1] * shape[2], sizeof(int));

    for (int i = 0; i < shape[0]; i++) {
        for (int j = 0; j < shape[1]; j++) {
            int index = shape[2] * j + (shape[2] * shape[1]) * i + seqs[i][j];
            printf("%d\n", index);
            arr[index] = 1;
        }
    }

    return arr;
}

int get2dIndex(int* arr, int width, int row, int col) {
    return arr[row * width + col];
}

int *flatSeqsToOneHot(int *seqs, int *shape) {
    int *arr = calloc(shape[0] * shape[1] * shape[2], sizeof(int));

    for (int i = 0; i < shape[0]; i++) {
        for (int j = 0; j < shape[1]; j++) {
            int c = get2dIndex(seqs, shape[1], i, j);
            int index = shape[2] * j + (shape[2] * shape[1]) * i + c;
            arr[index] = 1;
        }
    }

    return arr;
}
