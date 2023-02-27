using PyCall
using JLD


function getdf(path)
    py"""
    import pandas as pd

    def read_csv(path):
        return pd.read_csv(path)
    """
    data = py"read_csv"(path)
    return data
end

function dfToMatrix(df)
    data_matrix = Array{Float64}(undef, 0, length(df.columns))

    for i in df.index
        data_matrix = vcat(data_matrix, [convert(Float64, j) for j in df.loc[convert(Int64, i) + 1]]')
    end

    return data_matrix
end

function dfToStringMatrix(df)
    data_matrix = Array{String}(undef, 0, length(df.columns))

    for i in df.index
        data_matrix = vcat(data_matrix, reshape([j for j in df.loc[convert(Int64, i) + 1]], 1, length(df.columns)))
    end

    return data_matrix
end

df = getdf(joinpath(@__DIR__, "$(ARGS[begin])_filtered_dataset.csv"))

# function readPyFile(path)
#     py"""
#     exec(open($(path)).read(), globals(), locals())
#     """
# end

# readPyFile("tokenization.py")
# readPyFile("smilesenumeration.py")

py"""
from smiles_tools import return_tokens
from smiles_tools import SmilesEnumerator
"""

vocab = dfToStringMatrix(getdf(joinpath(@__DIR__, "vocab.csv")))

tokenizer = Dict(j => i for (i, j) in enumerate(vocab))
reverse_tokenizer = Dict(value => key for (key, value) in tokenizer)

py"""
def augment_smiles(string, n):
    sme = SmilesEnumerator()
    output = []
    for i in range(n):
        output.append(sme.randomize_smiles(string))
    
    return output
"""

augment_smiles(str, n) = py"augment_smiles"(str, n)
return_tokens(str) = py"return_tokens"(str)

n = parse(Int, ARGS[2])
max_length = let max_len
    try
        max_len = parse(Int, ARGS[3])
    catch ArgumentError
        max_len = parse(Bool, lowercase(ARGS[3]))
    end
    max_len
end

smiles = let temp_df
    temp_df = dfToStringMatrix(df)
    for i in 1:length(temp_df[:, begin])
        for augmented in augment_smiles(temp_df[:, begin][i], n)
            temp_df = vcat(temp_df, String[augmented temp_df[:, end][i]])
        end
    end
    temp_df
end

activity = reduce(hcat, [i == "Active" ? [1, 0] : [0, 1] for i in smiles[:, end]])'

function standardizeCase(str)
    str = titlecase(str)
    str = replace(str, "h" => "H")
end

strings = []
activity = []

for i in 1:length(smiles[:, begin])
    try
        processed_tokens = [tokenizer[standardizeCase(j)] for j in return_tokens(smiles[:, begin][i])[begin]]
        if typeof(max_length) != Bool && length(processed_tokens) <= max_length
            push!(strings, processed_tokens)
            push!(activity, smiles[:, end][i] == "Active" ? [1, 0] : [0, 1])
        end
        # https://discourse.julialang.org/t/using-push/30935/2
    catch
    end

    if i % 100 == 0
        println("$i | strings: $(length(strings)), activity: $(length(activity))")
    end
end

# strings = [[tokenizer[standardizeCase(j)] for j in return_tokens(i)[begin]] for i in smiles[:, begin]]
# activity = reduce(hcat, [i == "Active" ? [1, 0] : [0, 1] for i in smiles[:, end]])'

activity = reduce(hcat, activity)'

@assert length(strings) == size(activity)[begin]

convert_back(x) = join([i in keys(reverse_tokenizer) ? reverse_tokenizer[i] : "" for i in x])

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

println("Finished augmentations")

# function onehot(x)
#     onehot_array = zeros(length(keys(tokenizer)) + 1)
#     onehot_array[x + 1] = 1
#     return onehot_array
# end

# parsed = [onehot.(i) for i in padded_features]
# println(length(parsed))

# save("encoded_data.jld", "encoded_data", parsed, compress=true)

# matrix_parsed = [mapreduce(permutedims, vcat, i) for i in parsed]

# X = [matrix_parsed[i] for i in 1:length(strings)]
# Y = convert(Matrix{Float32}, activity)

# save("augmented_data.jld", "X", X, "Y", Y, compress=true)