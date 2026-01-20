import Documenter

using ApplicationDrivenLearning

Documenter.makedocs(;
    clean = true,
    sitename = "ApplicationDrivenLearning.jl documentation",
    authors = "Giovanni Amorim, Joaquim Garcia",
    pages = [
        "Home" => "index.md",
        "Tutorials" =>
            joinpath.(
                "tutorials",
                ["getting_started.md", "modes.md", "custom_forecast.md"],
            ),
        "Examples" => joinpath.("examples", ["scheduling.md", "newsvendor.md"]),
        "API Reference" => "reference.md",
    ],
)

Documenter.deploydocs(;
    repo = "github.com/LAMPSPUC/ApplicationDrivenLearning.jl.git",
    push_preview = true,
)
