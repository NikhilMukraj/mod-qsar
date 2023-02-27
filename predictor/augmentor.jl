# http://localhost:8888/tree?
# using IJulia
# notebook()

using PyCall


py"""
from smiles_tools import return_tokens, SmilesEnumerator
from c_wrapper import seqOneHot
import pandas as pd
import numpy as np
import os

vocab = pd.read_csv(f'{os.getcwd()}//vocab.csv')['tokens'].to_list()

tokenizer = {i : n for n, i in enumerate(vocab)}
reverse_tokenizer = {value: key for key, value in tokenizer.items()}
convert_back = lambda x: ''.join(reverse_tokenizer.get(np.argmax(i)-1, '') for i in x)

def augment_smiles(string, n):
    sme = SmilesEnumerator()
    output = []
    for i in range(n):
        output.append(sme.randomize_smiles(string))
    
    return output

testX = np.load(f'{os.getcwd()}//testx_sample.npy')
testY = np.load(f'{os.getcwd()}//testy_sample.npy')
strings = [convert_back(i) for i in testX[:1000]]
"""

tokenizer = py"tokenizer"
reverse_tokenizer = py"reverse_tokenizer"
convert_back = py"convert_back"
return_tokens = py"return_tokens"

augment_smiles(str, n) = py"augment_smiles"(str, n)

function standardizeCase(string::String)
    str = titlecase(string)
    str = replace(string, "h" => "H")
end

function return_augmented_list(string::String, n::Int64=5)
    current_augmentation = Matrix{String}(undef, 0, 1)
    counter = 0

    while length(current_augmentation) < n && counter < n * 2
        new_string = augment_smiles(string, 1)[begin]
        current_tokens = standardizeCase.(return_tokens(new_string)[begin])
        if issubset(Set(current_tokens), keys(tokenizer))
            current_augmentation = vcat(current_augmentation, join(current_tokens))
        end
        counter += 1
    end

    return current_augmentation
end

function return_augmented_tokens(string::String, n::Int64=5)
    current_augmentation = []
    counter = 0

    while length(current_augmentation) < n && counter < n * 2
        new_string = augment_smiles(string, 1)[begin]
        current_tokens = standardizeCase.(return_tokens(new_string)[begin])
        if issubset(Set(current_tokens), keys(tokenizer))
            push!(current_augmentation, current_tokens)
        end
        counter += 1
    end

    return current_augmentation
end

tokens = return_augmented_tokens.(strings)

# indicies = [i for i in 1:length(tokens) if length(tokens[i]) == 5]

# answers = [activity[i, :] for i in indicies]

function tokenize_and_pad(tokens_vector)
    len = 190-length(tokens_vector)
    return vcat(zeros(Int, len), [tokenizer[i]+1 for i in tokens_vector])
end

tokenize_and_pad.(tokens[begin])

# convert_back(py"seqOneHot"(tokenize_and_pad(tokens[begin][begin]), [190, 72]))

# convert_back.([py"seqOneHot"(tokenize_and_pad(i), [190, 72]) for i in tokens[begin]])

encoded_seqs = [[py"seqOneHot"(tokenize_and_pad(i), [190, 72]) for i in tokens[j]] for j in indicies]

# testX = py"testX"

original_seqs = [testX[i, :, :] for i in indicies]

py"""
model.compile()
"""

# needs to be reshaped to (batch_size, 190, 72)

py"""
preds = model.predict(np.array($(original_seqs)))
"""

initial_preds = py"preds"

py"""
aug_preds = [model.predict(np.array(i), verbose=0) for i in $(encoded_seqs)]
"""

"""
aug_preds = py"aug_preds"

initial_preds

acc(ŷ, y) = sum(argmax.(eachrow(ŷ)) .== argmax.(eachrow(y))) / length(answers)

acc(initial_preds, reduce(vcat,transpose.(answers)))

sum([argmax(sum(eachrow(i))) for i in aug_preds] .== argmax.(eachrow(reduce(vcat,transpose.(answers))))) / length(answers)
"""
