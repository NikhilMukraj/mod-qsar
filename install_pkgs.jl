using Pkg


open("jl_requirements.txt", "r") do f
    txt = read(f, String)
    pkgs = [split(i, "==") for i in split(txt)]

    [Pkg.add(Pkg.PackageSpec(;name=convert(String, i[begin]), version=convert(String, i[end]))) for i in pkgs]
end