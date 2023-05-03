using PyCall
using JLD
using ProgressBars
include("df_parser.jl")

py"""
from smiles_tools import return_tokens
from smiles_tools import SmilesEnumerator
from SmilesPE.pretokenizer import atomwise_tokenizer
import pandas as pd
import os

def augment_smiles(string, n):
    sme = SmilesEnumerator()
    output = []
    for i in range(n):
        output.append(sme.randomize_smiles(string))
    
    return output
"""

augment_smiles(str, n) = py"augment_smiles"(str, n)
atomwise_tokenizer(str) = py"atomwise_tokenizer"(str)
return_tokens(str, vocab) = py"return_tokens"(str, vocab)

n = parse(Int, ARGS[1])
vocab_path = ARGS[2]
debug = parse(Bool, lowercase(ARGS[3]))

dfs = [df_parser.getdf("$(ARGS[i])_filtered_dataset.csv") for i in 4:length(ARGS)]

println("Generating augmentations...")

smiles = let smiles
    smiles = df_parser.dfToStringMatrix.(dfs)
    for df_num in 1:length(smiles)
        println("Augmentation set number: $df_num")
        for i in tqdm(1:length(smiles[df_num][:, begin]))
            for augmented in augment_smiles(smiles[df_num][:, begin][i], n)
                smiles[df_num] = vcat(smiles[df_num], String[augmented smiles[df_num][:, end][i]])
            end
        end
    end
    smiles
end

println("Generated augmented dataframe, now processing tokens...")

strings = [[] for df_num in 1:length(smiles)]
activity = [[] for df_num in 1:length(smiles)]
vocabs = []

for df_num in tqdm(1:length(smiles))
    for i in 1:length(smiles[df_num][:, begin])
        tokens = [j for j in atomwise_tokenizer(smiles[df_num][:, begin][i])]
        push!(strings[df_num], tokens)
        push!(activity[df_num], smiles[df_num][:, end][i] == "Active" ? [1, 0] : [0, 1])

        # if i % 100 == 0
        #     println("$df_num | $i | strings: $(length(strings[df_num])), activity: $(length(activity[df_num]))")
        # end
    end

    # create vocab df and convert to tokens
    push!(vocabs, Set(reduce(vcat, strings[df_num])))
end

vocab = union(vocabs...)

py"""
vocab = list($(vocab))
vocab.sort()
vocab_df = pd.DataFrame(vocab, columns=["tokens"])
vocab_df.to_csv($(vocab_path), index=None)
"""

vocab = py"vocab"

tokenizer = Dict(j => i for (i, j) in enumerate(vocab))
# reverse_tokenizer = Dict(value => key for (key, value) in tokenizer)

for section in 1:length(strings)
    strings[section] = [[tokenizer[j] for j in i] for i in strings[section]]
end

formatted_activity = []
for section in 1:length(activity)
    push!(formatted_activity, reduce(hcat, activity[section])')
end

for section in 1:length(strings)
    @assert length(strings[section]) == size(formatted_activity[section])[begin]
end

max_length = maximum(reduce(vcat, [length.(strings[section]) for section in 1:length(strings)]))

function pad_features(input_strings, length_max)
    features = []
    for i in input_strings
        dim = size(i)[1]
        pad_size = length_max - dim 
        if pad_size > 0
            pad_array = zeros(Int64, pad_size)
            result = append!(pad_array, i)
        else
            result = i[1:length_max]
        end
        push!(features, result)
    end
    return features
end

for i in 3:length(ARGS)
    padded_features = pad_features(strings[i-2], max_length)
    padded_features = reduce(hcat, padded_features)'

    # save to jld and then process rest in python
    save("$(ARGS[i])_aug_unencoded_data.jld", "features", Matrix(padded_features), compress=true)

    save("$(ARGS[i])_aug_activity.jld", "activity", Matrix(formatted_activity[i-2]), compress=true)
end