if model_type == 1
    include("pretrain_uni.jl")
elseif model_type == 2
    include("pretrain_multi.jl")
elseif model_type == 3
    include("pretrain_many.jl")
else
    error("Invalid model type")
end