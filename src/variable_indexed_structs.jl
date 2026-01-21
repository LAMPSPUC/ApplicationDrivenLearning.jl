"""
    VariableIndexedVector(data::Vector{T}, index::Vector{Forecast{JuMP.VariableRef}})

A vector that can be indexed by Forecast variables.

...

# Arguments

  - `data::Vector{T}`: data vector.
  - `index::Vector{Forecast{JuMP.VariableRef}}`: index vector.
"""
struct VariableIndexedVector{T} <: AbstractVector{T}
    data::Vector{T}
    index::Vector{Forecast{JuMP.VariableRef}}

    # Inner constructor to enforce length consistency
    function VariableIndexedVector(data::Vector{T}, index::Vector{Forecast{JuMP.VariableRef}}) where T
        @assert length(data) == length(index) "Data and Variable index must have the same length"
        new{T}(data, index)
    end

    function VariableIndexedVector{T}(::UndefInitializer, index::Vector{Forecast{JuMP.VariableRef}}) where T
        return new{T}(Vector{T}(undef, length(index)), index)
    end
end

Base.size(v::VariableIndexedVector) = size(v.data)
Base.length(v::VariableIndexedVector) = length(v.data)
# linear indexing
Base.getindex(v::VariableIndexedVector, i::Int) = v.data[i]
Base.setindex!(v::VariableIndexedVector, val, i::Int) = (v.data[i] = val)

# helper to find index of a Forecast variable
function _get_idx(v::VariableIndexedVector, var::Forecast{JuMP.VariableRef})
    i = findfirst(isequal(var), v.index)
    if isnothing(i)
        throw(KeyError(var))
    end
    return i
end

# indexing by Forecast variables
function Base.getindex(v::VariableIndexedVector, var::Forecast{JuMP.VariableRef})
    return v.data[_get_idx(v, var)]
end

function Base.setindex!(v::VariableIndexedVector, val, var::Forecast{JuMP.VariableRef})
    v.data[_get_idx(v, var)] = val
end

# indexing by a vector of Forecast variables (bulk get and set)
function Base.getindex(v::VariableIndexedVector, vars::AbstractVector{<:Forecast})
    return [v[var] for var in vars]
end

function Base.setindex!(v::VariableIndexedVector, values::AbstractVector, vars::AbstractVector{<:Forecast})
    @assert length(values) == length(vars) "Number of values must match number of variables"
    return [v[var] = val for (val, var) in zip(values, vars)]
end

"""
    VariableIndexedMatrix(data::Matrix{T}, row_index::Vector{Forecast{JuMP.VariableRef}})

A matrix that can be indexed by Forecast variables in the rows.

...

# Arguments

  - `data::Matrix{T}`: data matrix.
  - `row_index::Vector{Forecast{JuMP.VariableRef}}`: row index.
"""
struct VariableIndexedMatrix{T} <: AbstractMatrix{T}
    data::Matrix{T}
    row_index::Vector{Forecast{JuMP.VariableRef}}

    function VariableIndexedMatrix(data::Matrix{T}, row_index::Vector{Forecast{JuMP.VariableRef}}) where T
        @assert size(data, 1) == length(row_index) "Number of rows in data must match number of row indices"
        new{T}(data, row_index)
    end
end

Base.size(m::VariableIndexedMatrix) = size(m.data)

# linear indexing
Base.getindex(m::VariableIndexedMatrix, i::Int, j::Int) = m.data[i, j]
Base.setindex!(m::VariableIndexedMatrix, val, i::Int, j::Int) = (m.data[i, j] = val)

# column lookup (get column 2) {M[:, 2]}
function Base.getindex(m::VariableIndexedMatrix, c::Int)
    return VariableIndexedVector(m.data[:, c], m.row_index)
end
