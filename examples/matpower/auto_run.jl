import JobQueueMPI

JQM = JobQueueMPI

dotm_files = [
    # "pglib_opf_case24_ieee_rts",
    "pglib_opf_case118_ieee",
    "pglib_opf_case300_ieee",
]

for i=1:size(dotm_files, 1)
    println("Running case: ", dotm_files[i])
    if i > 1
        # change the case name in the config file
        open(
            "config.jl",
            "r+"
        ) do file
            content = read(file, String)
            new_content = replace(
                content, 
                dotm_files[i-1] => dotm_files[i]
            )
            seek(file, 0)
            write(file, new_content)
        end
    end

    ## pretrain
    open(
        joinpath(@__DIR__, "config.jl"),
        "r+"
    ) do file
        content = read(file, String)
        new_content = replace(content, r"run_mode = \d" => "run_mode = 1")
        seek(file, 0)
        write(file, new_content)
    end
    include("main.jl")

    ## gradient
    open(
        joinpath(@__DIR__, "config.jl"),
        "r+"
    ) do file
        content = read(file, String)
        new_content = replace(content, r"run_mode = \d" => "run_mode = 2")
        seek(file, 0)
        write(file, new_content)
    end
    JQM.mpiexec(exe -> run(`$exe -n 12 $(Base.julia_cmd()) --project main.jl`))

    ## nelder-mead
    if N_HIDDEN_LAYERS == 0
        open(
            joinpath(@__DIR__, "config.jl"),
            "r+"
        ) do file
            content = read(file, String)
            new_content = replace(content, r"run_mode = \d" => "run_mode = 3")
            seek(file, 0)
            write(file, new_content)
        end
        JQM.mpiexec(exe -> run(`$exe -n 12 $(Base.julia_cmd()) --project main.jl`))
    end

    ## post analysis
    include("post_analysis.jl")
end

