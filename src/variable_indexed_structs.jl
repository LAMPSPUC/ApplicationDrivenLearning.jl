import LinearAlgebra
import Base./

"""
    VariableIndexedVector(data::Vector{T}, index::Vector{<:Forecast})

A vector that can be indexed by Forecast variables.

...

# Arguments

  - `data::Vector{T}`: data vector.
  - `index::Vector{<:Forecast}`: index vector.
"""
struct VariableIndexedVector{T} <: AbstractVector{T}
    data::Vector{T}
    index::Vector{<:Forecast}

    # Inner constructor to enforce length consistency
    function VariableIndexedVector(
        data::Vector{T},
        index::Vector{<:Forecast},
    ) where {T}
        @assert length(data) == length(index) "Data and Variable index must have the same length"
        @assert length(unique(index)) == length(index) "Variables must be unique"
        return new{T}(data, index)
    end

    function VariableIndexedVector{T}(
        ::UndefInitializer,
        index::Vector{<:Forecast},
    ) where {T}
        return new{T}(Vector{T}(undef, length(index)), index)
    end
end

Base.size(v::VariableIndexedVector) = size(v.data)
Base.length(v::VariableIndexedVector) = length(v.data)
# linear indexing
Base.getindex(v::VariableIndexedVector, i::Int) = v.data[i]
Base.setindex!(v::VariableIndexedVector, val, i::Int) = (v.data[i] = val)

# define Base rigth divide function (v / 2)
function /(v::VariableIndexedVector, i::Number)
    return VariableIndexedVector(v.data / i, v.index)
end

# define dot product of two VariableIndexedVectors
function LinearAlgebra.dot(v1::VariableIndexedVector, v2::VariableIndexedVector)
    @assert length(v1) == length(v2) "Vectors must have the same length"
    # assert that difference between two indices is empty
    @assert length(setdiff(v1.index, v2.index)) == 0 "Indices must contain the same set of variables"
    @assert length(setdiff(v2.index, v1.index)) == 0 "Indices must contain the same set of variables"

    return sum(v1[v2.index].data .* v2.data)
end

# helper to find index of a Forecast variable
function _get_idx(v::VariableIndexedVector, var::Forecast)
    i = findfirst(isequal(var), v.index)
    if isnothing(i)
        throw(KeyError(var))
    end
    return i
end

# bulk indexing by integer indices
function Base.getindex(v::VariableIndexedVector, indices::AbstractVector{Int})
    return VariableIndexedVector(v.data[indices], v.index[indices])
end

# indexing by Forecast variables
function Base.getindex(v::VariableIndexedVector, var::Forecast)
    return v.data[_get_idx(v, var)]
end

function Base.setindex!(v::VariableIndexedVector, val, var::Forecast)
    return v.data[_get_idx(v, var)] = val
end

# indexing by a vector of Forecast variables (bulk get and set)
function Base.getindex(
    v::VariableIndexedVector,
    vars::AbstractVector{<:Forecast},
)
    return v[[_get_idx(v, var) for var in vars]]
end

function Base.setindex!(
    v::VariableIndexedVector,
    values::AbstractVector,
    vars::AbstractVector{<:Forecast},
)
    @assert length(values) == length(vars) "Number of values must match number of variables"
    return [v[var] = val for (val, var) in zip(values, vars)]
end

"""
    VariableIndexedMatrix(data::Matrix{T}, row_index::Vector{<:Forecast})

A matrix that can be indexed by Forecast variables in the rows.

...

# Arguments

  - `data::Matrix{T}`: data matrix.
  - `row_index::Vector{<:Forecast}`: row index.
"""
struct VariableIndexedMatrix{T} <: AbstractMatrix{T}
    data::Matrix{T}
    row_index::Vector{<:Forecast}

    function VariableIndexedMatrix(
        data::Matrix{T},
        row_index::Vector{<:Forecast},
    ) where {T}
        @assert size(data, 1) == length(row_index) "Number of rows in data must match number of row indices"
        @assert length(unique(row_index)) == length(row_index) "Variables must be unique"
        return new{T}(data, row_index)
    end

    function VariableIndexedMatrix{T}(
        ::UndefInitializer,
        index::Vector{<:Forecast},
        n::Real,
    ) where {T}
        return new{T}(Matrix{T}(undef, length(index), n), index)
    end

    function VariableIndexedMatrix{T}(
        ::Nothing,
        index::Vector{<:Forecast},
        n::Real,
    ) where {T}
        return new{T}(Matrix{T}(nothing, length(index), n), index)
    end
end

# helper to find index of a Forecast variable
function _get_idx(m::VariableIndexedMatrix, var::Forecast)
    i = findfirst(isequal(var), m.row_index)
    if isnothing(i)
        throw(KeyError(var))
    end
    return i
end

Base.size(m::VariableIndexedMatrix) = size(m.data)

# linear indexing
Base.getindex(m::VariableIndexedMatrix, i::Int, j::Int) = m.data[i, j]
function Base.setindex!(m::VariableIndexedMatrix, val, i::Int, j::Int)
    return (m.data[i, j] = val)
end

# column lookup (get column 2) {M[2]}
function Base.getindex(m::VariableIndexedMatrix, c::Int)
    return VariableIndexedVector(m.data[:, c], m.row_index)
end

# row lookup (get values from variable) {M[forecast_var]}
function Base.getindex(m::VariableIndexedMatrix, var::Forecast)
    return m.data[_get_idx(m, var), :]
end

# multi-row lookup {M[[f_var_1, f_var_2]]}
function Base.getindex(m::VariableIndexedMatrix, vars::Vector{<:Forecast})
    return VariableIndexedMatrix(
        m.data[[_get_idx(m, var) for var in vars], :],
        vars,
    )
end

# set column 2 values for all var indices {M[2] = [1,2,3]}
function Base.setindex!(
    m::VariableIndexedMatrix,
    values::VariableIndexedVector,
    c::Int,
)
    return m.data[:, c] = values[m.row_index]
end

# set values for a single var index {M[forecast_var] = [1,2,3]}
function Base.setindex!(m::VariableIndexedMatrix, values::Vector, var::Forecast)
    return m.data[_get_idx(m, var), :] = values
end

# set values for a subset of var indices {M[[f_var_1, f_var_2]] = [[1 2 3]; [4 5 6]]}
function Base.setindex!(
    m::VariableIndexedMatrix,
    values::Matrix,
    vars::Vector{<:Forecast},
)
    return m.data[[_get_idx(m, var) for var in vars], :] = values
end

# define sum of matrix by summing all values for each variable
function Base.sum(m::VariableIndexedMatrix)
    return VariableIndexedVector(sum(m.data, dims = 2)[:, 1], m.row_index)
end

# define Base rigth divide function (M / 2)
function /(m::ApplicationDrivenLearning.VariableIndexedMatrix, i::Number)
    return ApplicationDrivenLearning.VariableIndexedMatrix(
        m.data / i,
        m.row_index,
    )
end

# define dot product of a VariableIndexedVectors and a VariableIndexedMatrix
function LinearAlgebra.dot(v1::VariableIndexedVector, m2::VariableIndexedMatrix)
    @assert length(v1) == size(m2, 1) "Vector must have the same length as the number of rows in matrix"
    # assert that difference between two indices is empty
    @assert length(setdiff(v1.index, m2.row_index)) == 0 "Indices must contain the same set of variables"
    @assert length(setdiff(m2.row_index, v1.index)) == 0 "Indices must contain the same set of variables"

    return sum(v1[m2.row_index].data .* m2.data)
end
