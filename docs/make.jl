using Documenter

push!(LOAD_PATH, "../src")
using ApplicationDrivenLearning

makedocs(;
    modules=[ApplicationDrivenLearning],
    doctest=false,
    clean=true,
    sitename="ApplicationDrivenLearning.jl",
    authors="Giovanni Amorim, Joaquim Garcia",
    pages=[
        "Home" => "index.md",
        "API Reference" => "reference.md"
    ]
)