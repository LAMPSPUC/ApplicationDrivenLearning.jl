import Documenter

using ApplicationDrivenLearning

Documenter.makedocs(;
    clean = true,
    sitename = "ApplicationDrivenLearning.jl documentation",
    authors = "Giovanni Amorim, Joaquim Garcia",
)

Documenter.deploydocs(;
    repo = "github.com/LAMPSPUC/ApplicationDrivenLearning.jl.git",
    push_preview = true,
)
