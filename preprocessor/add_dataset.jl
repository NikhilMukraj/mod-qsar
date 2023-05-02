using PyCall
using JLD
using ProgressBars
include("df_parser.jl")

py"""
from smiles_tools import return_tokens
from smiles_tools import SmilesEnumerator
from SmilesPE.pretokenizer import atomwise_tokenizer
"""

py"""
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

n = parse(Int, ARGS[2])
max_length = let max_length
    try
        max_length = parse(Int, ARGS[3])
    catch ArgumentError
        max_length = parse(Bool, lowercase(ARGS[3]))
    end
    max_length
end

override = parse(Bool, lowercase(ARGS[4]))

df = df_parser.getdf("$(ARGS[begin])_filtered_dataset.csv")

println("Generating augmentations...")

smiles = let temp_df
    temp_df = df_parser.dfToStringMatrix(df)
    for i in tqdm(1:length(temp_df[:, begin]))
        for augmented in augment_smiles(temp_df[:, begin][i], n)
            temp_df = vcat(temp_df, String[augmented temp_df[:, end][i]])
        end
    end
    temp_df
end

println("Generated augmented dataframe, now processing tokens...")

activity = reduce(hcat, [i == "Active" ? [1, 0] : [0, 1] for i in smiles[:, end]])'

strings = []
activity = []

# check for pre-existing tokens here

vocab_path = ARGS[5]
vocab = df_parser.dfToStringMatrix(df_parser.getdf(vocab_path))

tokenizer = Dict(j => i for (i, j) in enumerate(vocab))
reverse_tokenizer = Dict(value => key for (key, value) in tokenizer)

for i in tqdm(1:length(smiles[:, begin]))
    returned_tokens, validToken = return_tokens(smiles[:, begin][i], tokenizer)
    if validToken && override
        # println("$i | Overriding token")
        continue
    elseif validToken && !override
        throw("Not a valid token")
    end

    processed_tokens = [tokenizer[j] for j in returned_tokens]
    if typeof(max_length) != Bool && length(processed_tokens) <= max_length
        push!(strings, processed_tokens)
        push!(activity, smiles[:, end][i] == "Active" ? [1, 0] : [0, 1])
    end

    # if i % 100 == 0
    #     println("$i | strings: $(length(strings)), activity: $(length(activity))")
    # end
end

# strings = [[tokenizer[standardizeCase(j)] for j in return_tokens(i)[begin]] for i in smiles[:, begin]]
# activity = reduce(hcat, [i == "Active" ? [1, 0] : [0, 1] for i in smiles[:, end]])'

activity = reduce(hcat, activity)'

@assert length(strings) == size(activity)[begin]

# convert_back(x) = join([i in keys(reverse_tokenizer) ? reverse_tokenizer[i] : "" for i in x])

if typeof(max_length) == Bool
    max_length = maximum(length.(strings))
end

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

padded_features = pad_features(strings, max_length)
padded_features = reduce(hcat, padded_features)'

# save to jld and then process rest in python

save("$(ARGS[begin])_aug_unencoded_data.jld", "features", Matrix(padded_features), compress=true)

save("$(ARGS[begin])_aug_activity.jld", "activity", Matrix(activity), compress=true)
