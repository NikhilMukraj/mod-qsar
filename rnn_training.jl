using Flux
using Flux.Data: DataLoader
using PyCall
using TOML
using Pkg
using Random
using Plots
using PlotThemes
using BSON: @save

env_path = TOML.parse(read("mltools.toml", String))["env_path"]
Pkg.activate(env_path)

import MLTools as mlt

df = mlt.getdf(joinpath(@__DIR__, "filtered_dataset.csv"))

smiles = mlt.dfToStringMatrix(df)

vocab = mlt.dfToStringMatrix(mlt.getdf(joinpath(@__DIR__, "vocab.csv")))

tokenizer = Dict(j => i for (i, j) in enumerate(vocab))
reverse_tokenizer = Dict(value => key for (key, value) in tokenizer)

function readPyFile(path)
    py"""
    exec(open($(path)).read(), globals(), locals())
    """
end

readPyFile("tokenization.py")
return_tokens(str) = py"return_tokens"(str)

strings = [[tokenizer[j] for j in return_tokens(i)[begin]] for i in smiles[:, begin]]
activity = reduce(hcat, [i == "Active" ? [1, 0] : [0, 1] for i in smiles[:, end]])'

@assert length(strings) == size(activity)[begin]

# println("average len is $(sum(length.(strings))/length(strings))"

# test_tokens = [reverse_tokenizer[i] for i in strings[begin]]
# test_string = join(test_tokens)

# println("test reverse tokenization: $test_string")

# println("max length string: $(join([reverse_tokenizer[i] for i in strings[findmax(length.(strings))[end]]])))")

convert_back(x) = join([i in keys(reverse_tokenizer) ? reverse_tokenizer[i] : "" for i in x])

max_length = maximum(length.(strings))

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

# sample_padded = pad_features(strings, max_length)[begin]
# convert_back(sample_padded)

# return a json of relevant model metadata

padded_features = pad_features(strings, max_length)

function onehot(x)
    onehot_array = zeros(length(keys(tokenizer)) + 1)
    onehot_array[x + 1] = 1
    return onehot_array
end

parsed = [onehot.(i) for i in padded_features]
println(length(parsed))

matrix_parsed = [mapreduce(permutedims, vcat, i) for i in parsed]

X = [matrix_parsed[i] for i in 1:length(strings)]
Y = convert(Matrix{Float32}, activity)

trainX, trainY, testX, testY = mlt.trainTestSplit(X, Y, percent_train=.9)
size(trainX), size(trainY), size(testX), size(testY)

trainX = convert.(Matrix{Float32}, trainX)
trainY = convert(Matrix{Float32}, Float32.(trainY))
testX = convert.(Matrix{Float32}, testX)
testY = convert(Matrix{Float32}, Float32.(testY))

trainX = reshape(trainX, 1, size(trainX)[begin])
trainY = trainY'
testX = reshape(testX, 1, size(testX)[begin])
testY = testY'

# batch(arr, n) = [arr[i:min(i + n - 1, end)] for i in 1:n:length(arr)]

# trainX = batch(trainX, 32)
# trainY = batch(trainY, 32)
# testX = batch(testX, 32)
# testY = batch(testY, 32)

# maybe no batch and leave to dataloader?

N = length(keys(tokenizer)) + 1

model = Chain(
    LSTM(N => 128), relu,
    LSTM(128 => 16), relu,
    Dense(16 => 2), softmax
) # |> gpu

function eval_model(x)    
    out = model(x)[:, end]
    Flux.reset!(model)

    return out'
end

function vcatTranspose(x)
    temp_matrix = Matrix{Float32}(undef, 0, size(x[begin])[end])

    for i in x 
        temp_matrix = vcat(temp_matrix, i)
    end

    return temp_matrix
end

function loss(x, y)
    pred = eval_model.(x)
    pred = vcatTranspose(pred)

    return Flux.crossentropy(pred', y)
end

# model(convert(Matrix{Float32}, trainX[begin])[:, 1])
# eval_model(convert(Matrix{Float32}, trainX[begin]))
# eval_model(convert(Matrix{Float32}, trainX[begin]))
# loss(convert(Matrix{Float32}, trainX[begin]), convert(Matrix{Float32}, trainY[begin]))

function accuracy(ŷ, y)
    # check accuracy of prediction
    # ŷ
    # ŷ = eval_model(x)
    return sum(argmax.(ŷ) .== argmax.(y)) / length(y)
end

# https://spcman.github.io/getting-to-know-julia/deep-learning/vision/flux-cnn-zoo/
# maybe just directly input?

# losses = []

# for i in 1:length(trainX)
#     push!(losses, loss(convert(Matrix{Float32}, trainX[i]), convert(Matrix{Float32}, trainY[i])))
# end

# data = [(trainX, trainY)]
# train_data_len = length(data)
# test_data_len = length(testX)

# create vectorized loop to train

# function totaling(x, y)
#     total = 0
#     @simd for i in x
#         total += i * y
#     end
#
#     return total
# end

# shuffle!(data)

data = DataLoader((trainX, trainY), batchsize=32, shuffle=true)

opt = Adam(.001)

epochs = 1:100

training_losses = []
testing_losses = []
training_accs = []
testing_accs = []

for i in epochs
    println("Epoch: $i")
    Flux.train!(loss, Flux.params(model), data, opt)

    pred = eval_model.(trainX)
    test_pred = eval_model.(testX)
    pred = vcatTranspose(pred)
    test_pred = vcatTranspose(pred)

    push!(training_losses, Flux.crossentropy(pred', trainY))
    push!(testing_losses, Flux.crossentropy(test_pred', testY))

    push!(testing_accs, accuracy([pred[i, :] for i in 1:size(pred)[begin]], 
                                 [trainY[:, i] for i in 1:size(trainY)[end]]))
    push!(testing_accs, accuracy([test_pred[i, :] for i in 1:size(test_pred)[begin]], 
                                 [testY[:, i] for i in 1:size(testY)[end]]))

    println("Training Accuracy: $(training_accs[i])")
    println("Testing Accuracy: $(testing_accs[i])")

    println("Training Loss: $(training_losses[i])")
    println("Testing Loss: $(testing_losses[i])")
end