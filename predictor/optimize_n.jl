using Flux
include("../preprocessor/df_parser.jl")


accs = df_parser.dfToMatrix(df_parser.getdf(ARGS[begin]))

# sets up logistic model
a, b, c, d = rand(1), rand(1), rand(1), rand(1)

predict(x) = (a .* exp.(c .* x .+ d)) ./ (exp.(c .* x .+ d) .+ b)

loss(x, y) = Flux.Losses.mse(predict(x), y)

ps = Flux.params(a, b, c, d)

opt = Adam(.1)
data = Flux.DataLoader((accs[:, begin], accs[:, end]))

# trains logistic model on given accuracy data
for i in 1:100
    Flux.train!(loss, ps, data, opt)
end

test_range = Float32.(hcat(collect(minimum(accs[:, begin]):.01:maximum(accs[:, begin]))))

# plots logistic curve for visual analysis
# scatter(accs[:, begin], accs[:, end], color="blue")
# plot!(test_range, predict(test_range), color="green")

# finds where logistic curve starts to peak based on threshold (rounded maximum predicted value)
# (could be edited to take the floor)
preds = predict(test_range)
optimized_n_value = let optimized = 0
        for (n, i) in enumerate(preds)
            if round(i; digits=2) >= round(maximum(preds); digits=1)
                optimized = Int32(round(test_range[n]))
                break
            end
        end
        optimized
    end

GREEN = "\033[1;32m"
NC = "\033[0m"
println("Optimized amount of augmentations: $(GREEN)$(optimized_n_value)$(NC)")
