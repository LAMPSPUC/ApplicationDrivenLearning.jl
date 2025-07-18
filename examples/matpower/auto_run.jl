import JobQueueMPI

JQM = JobQueueMPI

dotm_files = [
    "pglib_opf_case24_ieee_rts",
    "pglib_opf_case118_ieee",
    "pglib_opf_case179_goc",
    "pglib_opf_case240_pserc",
    "pglib_opf_case588_sdet",
    "pglib_opf_case300_ieee",
    "pglib_opf_case500_goc",
    "pglib_opf_case793_goc",
    "pglib_opf_case1354_pegase"
]

for i=1:size(dotm_files, 1)
    println("Running case: ", dotm_files[i])
    # change the case name in the config file
    open(
        "config.jl",
        "r+"
    ) do file
        content = read(file, String)
        new_content = replace(
            content, 
            r"CASE_NAME = \".+\"" => "CASE_NAME = \"$(dotm_files[i])\""
        )
        seek(file, 0)
        write(file, new_content)
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
    JQM.mpiexec(exe -> run(`$exe -n 13 $(Base.julia_cmd()) --project main.jl`))

    ## nelder-mead
    if (N_HIDDEN_LAYERS == 0) && (N_DEMANDS <= 100)
        open(
            joinpath(@__DIR__, "config.jl"),
            "r+"
        ) do file
            content = read(file, String)
            new_content = replace(content, r"run_mode = \d" => "run_mode = 3")
            seek(file, 0)
            write(file, new_content)
        end
        JQM.mpiexec(exe -> run(`$exe -n 13 $(Base.julia_cmd()) --project main.jl`))
    end

    ## post analysis
    include("post_analysis.jl")
end

