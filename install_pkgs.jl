using Pkg


open("jl_requirements.txt", "r") do f
    Pkg.add.([split(read(f, String))])
end