import pandas as pd
import numpy as np
from tokenization import return_tokens
import os


class SMILES:
    def __init__(self, string=None, vocab=None, tokenizer_dict=None, reverse_tokenizer=None):
        self.string = string

        if self.string is not None:
            self.tokens, invalidity = return_tokens(string)

            if invalidity:
                raise Exception("String contains tokens not within vocabulary")

        if vocab is None:
            self.vocab = pd.read_csv(f'{os.getcwd()}\\vocab.csv').iloc[:, 0].to_list()
        else:
            self.vocab = vocab

        # optimize this memory chokehold
        self.max_value = len(self.vocab) - 1 # !! important !! needs to be considered before onehot encoding
        self.bit_length = len(bin(self.max_value)[2:])
        self.max_bit_value = int('1' * self.bit_length, 2)
        self.buffer = self.max_bit_value - self.max_value

        if tokenizer_dict is None:
            self.tokenizer_dict = {i : n for n, i in enumerate(self.vocab)}
        else:
            self.tokenizer_dict = tokenizer_dict
        
        if reverse_tokenizer is None:
            self.reverse_tokenizer = {value : key for key, value in self.tokenizer_dict.items()}
        else:
            self.reverse_tokenizer = reverse_tokenizer

        self.bit_string = None

    def randomTokens(self, n):
        self.tokens = np.random.choice(list(self.tokenizer_dict), size=np.random.randint(3, n)).tolist()

        return self.tokens

    def encode(self, tokens=None):
        if tokens is None:
            tokens = self.tokens

        bit_string = ''

        # maybe check if bit string is valid or needs underflow editing

        for i in tokens:
            bit_string += str(bin(self.tokenizer_dict[i] + self.buffer)[2:]).zfill(self.bit_length)

        self.bit_string = bit_string

        return bit_string

    def decode(self, encoded_string=None):
        if encoded_string is None:
            encoded_string = self.bit_string

        tokens = []

        for i in range(0, len(encoded_string) - self.bit_length + 1, self.bit_length):
            tokens.append(self.reverse_tokenizer[int(encoded_string[i:i+self.bit_length], 2) - self.buffer])

        return tokens

        # make sure to check if mutation makes value fall below buffer
        # if below underflow back to top

    def underflowBitString(self, input_string=None):
        if input_string is None:
            input_string = self.bit_string

        strings = []

        for i in range(0, len(input_string) - self.bit_length + 1, self.bit_length):
            part = int(input_string[i:i+self.bit_length], 2)
            if part < self.buffer:
                part = self.max_bit_value - part
            
            strings.append(str(bin(part))[2:].zfill(self.bit_length))

        self.bit_string = ''.join(str(i) for i in strings)