"""
Input normalization for the public entry points.

[`compute_cost`](@ref) and [`train!`](@ref) accept matrices, vectors,
Tables.jl-compatible tables (`DataFrame`, `NamedTuple` of vectors, `CSV.File`,
…) and — for the realized values — a `Dict` keyed by [`Forecast`](@ref)
variables. Everything is normalized to an `AbstractMatrix{<:Real}` *once*, at the
boundary, and the whole pipeline below works on matrices.

# The one rule

**A named container is matched by name; an unnamed container is matched by
position. Neither is ever guessed at.**

An `AbstractArray` carries no column names, so position is its only possible
reading and it keeps the meaning it always had. A table's columns *are* named, so
ignoring those names in favour of their order is how a caller silently ends up
training against the wrong series — reordering the columns of a `DataFrame` would
change the answer without any complaint. Tables are therefore matched against the
schema the predictive model declares (`input_names` / `output_names`, see
[`PredictiveModel`](@ref)) and rejected when it cannot be matched, rather than
falling back to position. `Matrix(df)` is the way to ask for positional matching,
and reads as exactly that request: it drops the names.

There is no exception for a table with a single column. Its order cannot be wrong,
but its *name* still can: a one-input model handed `DataFrame(humidity = ...)`
when it wanted `temp` is a mistake worth catching, and it is only catchable
against a declared schema. Exempting the single-column case would also mean the
smallest model — the one anybody meets first — is the one that teaches the habit
of not declaring anything.

The same rule extends one dimension further, to *rows*. Declaring a `sample_key`
(see [`PredictiveModel`](@ref)) names the column that identifies an observation —
a timestamp or an id — and the realized values are then looked up by key for each
row of `X` rather than trusted to arrive in the same order. Without one, rows are
matched by position, which is the only reading unlabelled rows admit.

Converting at the boundary rather than making the pipeline table-generic is
deliberate: the per-sample loop needs `view(Y, t, :)` and `X[batch, :]`, i.e.
O(1) row access and row slicing, which the Tables.jl row interface does not
provide cheaply. One materialization per call is both simpler and faster than
iterating rows `T` times.
"""

"""
    _table_column_names(data)

Column names of a Tables.jl-compatible `data`, as a `Vector{Symbol}`.
"""
function _table_column_names(data)
    return collect(Symbol, Tables.columnnames(Tables.columns(data)))
end

"""
    _columns_to_matrix(data, names::Vector{Symbol}, what::String)

Build a `(rows x length(names))` matrix from the named columns of the
Tables.jl-compatible `data`, in the order given by `names`.

Used in preference to `Tables.matrix` on purpose. `Tables.matrix` is generic over
the whole table interface and pays for it in latency: on this package's own test
data, the first `Tables.matrix(::DataFrame)` call measured **117 s** of
compilation, against 0.5 s for the column-wise construction below. Since the
columns have to be looked up by name for `Y` anyway, doing the same for `X` costs
nothing and keeps both paths on the fast route.
"""
function _columns_to_matrix(data, names::Vector{Symbol}, what::String)
    if isempty(names)
        throw(ArgumentError("$what has no columns."))
    end
    columns = Tables.columns(data)
    series = Any[Tables.getcolumn(columns, name) for name in names]
    n = length(first(series))
    for (name, s) in zip(names, series)
        if length(s) != n
            throw(
                ArgumentError(
                    "All columns of $what must have the same length; column " *
                    "`$name` has $(length(s)) rows instead of $n.",
                ),
            )
        end
    end
    M = Matrix{promote_type(eltype.(series)...)}(undef, n, length(series))
    for (j, s) in enumerate(series)
        M[:, j] = s
    end
    return _check_real_eltype(M, what)
end

"""
    _forecast_base_name(f::Forecast)

Declared name of a [`Forecast`](@ref) variable, i.e. the name of its plan twin
with the `_plan` suffix removed. Returns `""` for an anonymous variable, which
callers must treat as "cannot be matched by name".

`@variable(m, d, Forecast)` gives `"d"`; `@variable(m, f[1:2], Forecast)` gives
`"f[1]"` and `"f[2]"`.
"""
function _forecast_base_name(f::Forecast)
    n = JuMP.name(f.plan)
    return endswith(n, "_plan") ? chop(n; tail = 5) : n
end

"""
    _check_real_eltype(M::AbstractMatrix, what::String)

Throw an informative `ArgumentError` unless `M` has a `Real` element type.

Tables carrying string or `missing` entries otherwise fail much deeper, inside
Flux or the solver, with an error that does not point back at the data.
"""
function _check_real_eltype(M::AbstractMatrix, what::String)
    if !(eltype(M) <: Real)
        throw(
            ArgumentError(
                "$what must have real-valued entries, got element type " *
                "$(eltype(M)). Columns that are non-numeric, or that contain " *
                "`missing`, have to be dropped or converted first.",
            ),
        )
    end
    return M
end

"""
    _matched_columns(available, declared, n, what, hint)

Names of the table columns to use, in model order.

`available` are the table's own column names, `declared` the schema the
predictive model declares and `n` the number of slots the model has to fill.
Implements the rule described at the top of this file, so both `X` and `Y` are
read the same way and can only differ in their `hint`.

A table with no declared schema is always an error, whatever its width — `n` and
`sample_key` serve only to say so precisely.
"""
function _matched_columns(
    available::Vector{Symbol},
    declared::Union{Vector{Symbol},Nothing},
    n::Int,
    sample_key::Union{Symbol,Nothing},
    what::String,
    hint::String,
)
    if !isnothing(declared)
        absent = filter(!in(available), declared)
        if !isempty(absent)
            throw(
                ArgumentError(
                    "$what is missing the column(s) " *
                    "$(join(string.(absent), ", ")). Table columns are matched " *
                    "by name, not by position. Columns found: " *
                    "$(join(string.(available), ", ")). $hint",
                ),
            )
        end
        # extra columns are ignored: exactly the declared ones are taken, in the
        # declared order, whatever order the table happens to hold them in. That
        # covers `sample_key` too, which is why it is only consulted below
        return declared
    end
    # the key column identifies a row, it is not data - so it is not one of the
    # columns the caller is being asked to account for
    candidates = if isnothing(sample_key)
        available
    else
        filter(!isequal(sample_key), available)
    end
    throw(
        ArgumentError(
            "$what is a table with $(length(candidates)) matchable column(s) " *
            "($(join(string.(candidates), ", "))), but the predictive model does " *
            "not declare which column holds each of its $n slot(s), so they " *
            "cannot be matched by name. This holds for a single column too: its " *
            "order cannot be wrong, but its name still can. $hint",
        ),
    )
end

"""
    _sample_keys(data, forecast::PredictiveModel, what::String)

Values of the `sample_key` column of `data`, or `nothing` when rows cannot be
matched by key.

Returns `nothing` for anything that is not a table: an array or a `Dict` of series
carries no row labels, so position is the only reading available — the same
fallback the column rule makes for an unnamed container. A table that *is* passed
must carry the declared key, since declaring one asserts that it does.
"""
function _sample_keys(data, forecast::PredictiveModel, what::String)
    key = forecast.sample_key
    (isnothing(key) || !Tables.istable(data)) && return nothing
    available = _table_column_names(data)
    if !(key in available)
        throw(
            ArgumentError(
                "$what is missing the `sample_key` column `$key`, which the " *
                "predictive model declares to identify a sample. Columns found: " *
                "$(join(string.(available), ", ")).",
            ),
        )
    end
    return Tables.getcolumn(Tables.columns(data), key)
end

"""
    _row_permutation(kx, ky, sample_key)

Row of `Y` to use for each row of `X`, given their `sample_key` columns, or
`nothing` when the two are already aligned and nothing has to be moved.

This is a lookup, not a merge: every key of `X` must appear exactly once in `Y`,
and rows of `Y` that `X` does not ask for are ignored — the row-wise counterpart
of ignoring an unwanted column. `X` fixes the sample order, since that is the
order the returned per-sample costs and gradients are in.
"""
function _row_permutation(kx, ky, sample_key::Symbol)
    # the overwhelmingly common case: the two tables are already in step, so
    # nothing is looked up, permuted or copied
    kx == ky && return nothing
    for (keys, what) in ((kx, "input data"), (ky, "realized values"))
        if !allunique(keys)
            throw(
                ArgumentError(
                    "The `sample_key` column `$sample_key` of the $what repeats " *
                    "at least one key, so it cannot identify a sample.",
                ),
            )
        end
    end
    index = Dict(k => i for (i, k) in enumerate(ky))
    perm = Vector{Int}(undef, length(kx))
    for (i, k) in enumerate(kx)
        j = get(index, k, 0)
        if j == 0
            throw(
                ArgumentError(
                    "The realized values have no row with `$sample_key` = " *
                    "$(repr(k)), which the input data asks for. Both tables must " *
                    "cover the same samples; the realized values may hold more.",
                ),
            )
        end
        perm[i] = j
    end
    return perm
end

const _X_HINT =
    "Pass `input_names` to `set_forecast_model`, write the `input_output_map` " *
    "with `Symbol` keys, or pass `Matrix(X)` to match the columns by position " *
    "instead."

const _Y_HINT =
    "Pass `output_names` to `set_forecast_model`, name the columns after the " *
    "forecast variables, pass a `Dict` keyed by the `Forecast` variables, or " *
    "pass `Matrix(Y)` to match the columns by position instead."

"""
    _to_input_matrix(X, forecast::PredictiveModel)

Normalize the input data `X` to a `(samples x features)` matrix.

Accepts an `AbstractMatrix` (returned as is), an `AbstractVector` (treated as a
single feature, i.e. reshaped to `(T, 1)`) or any Tables.jl-compatible table,
whose columns are selected by `forecast.input_names`.
"""
function _to_input_matrix(X::AbstractMatrix, ::PredictiveModel)
    return _check_real_eltype(X, "Input data `X`")
end

function _to_input_matrix(X::AbstractVector, ::PredictiveModel)
    return _check_real_eltype(reshape(X, length(X), 1), "Input data `X`")
end

function _to_input_matrix(X, forecast::PredictiveModel)
    if !Tables.istable(X)
        throw(
            ArgumentError(
                "Input data `X` must be a matrix, a vector or a " *
                "Tables.jl-compatible table (for example a `DataFrame`), got " *
                "$(typeof(X)).",
            ),
        )
    end
    names = _matched_columns(
        _table_column_names(X),
        forecast.input_names,
        forecast.input_size,
        forecast.sample_key,
        "Input data `X`",
        _X_HINT,
    )
    return _columns_to_matrix(X, names, "Input data `X`")
end

"""
    _output_column_names(forecast::PredictiveModel)

Name of the `Y` column holding each of `forecast.output_variables`, or `nothing`
when the forecast variables cannot name their own columns.

Defaults to the declared names of the variables themselves, so that
`@variable(model, demand, Forecast)` reads a `demand` column with nothing to
configure. An explicit `output_names` overrides them, which is what container
declarations need: `@variable(model, d[1:2], Forecast)` names its variables
`d[1]` and `d[2]`, and a table is unlikely to carry columns called that.
"""
function _output_column_names(forecast::PredictiveModel)
    if !isnothing(forecast.output_names)
        return forecast.output_names
    end
    names = Symbol.(_forecast_base_name.(forecast.output_variables))
    # an anonymous variable has no name to match, and a repeated name would feed
    # two different variables from the same column
    if all(!isempty, string.(names)) && allunique(names)
        return names
    end
    return nothing
end

"""
    _to_output_matrix(Y, forecast::PredictiveModel)

Normalize the realized values `Y` to a `(samples x variables)` matrix whose
columns follow the order of `forecast.output_variables`.

Accepts an `AbstractMatrix` or `AbstractVector` — matched positionally — or any
Tables.jl-compatible table, whose columns are matched by name against
[`_output_column_names`](@ref).
"""
function _to_output_matrix(Y::AbstractMatrix, ::PredictiveModel)
    return _check_real_eltype(Y, "Realized values `Y`")
end

function _to_output_matrix(Y::AbstractVector, ::PredictiveModel)
    return _check_real_eltype(reshape(Y, length(Y), 1), "Realized values `Y`")
end

function _to_output_matrix(Y, forecast::PredictiveModel)
    if !Tables.istable(Y)
        throw(
            ArgumentError(
                "Realized values `Y` must be a matrix, a vector, a " *
                "Tables.jl-compatible table (for example a `DataFrame`) or a " *
                "`Dict` mapping each `Forecast` variable to its series, got " *
                "$(typeof(Y)).",
            ),
        )
    end
    names = _matched_columns(
        _table_column_names(Y),
        _output_column_names(forecast),
        forecast.output_size,
        forecast.sample_key,
        "Realized values `Y`",
        _Y_HINT,
    )
    return _columns_to_matrix(Y, names, "Realized values `Y`")
end

"""
    _to_matrices(X, Y, forecast::PredictiveModel)

Normalize both arguments to matrices, aligning the rows of `Y` to those of `X`
when a `sample_key` is declared and both carry it.

The two are normalized *together* because the row check is the one thing that
cannot be decided from either argument alone. Every public entry point goes
through here, so declaring a `sample_key` once is enough — there is no per-call
keyword to forget, and forgetting it would silently mean an unchecked alignment.
"""
function _to_matrices(X, Y, forecast::PredictiveModel)
    Xm = _to_input_matrix(X, forecast)
    Ym = _to_output_matrix(Y, forecast)
    kx = _sample_keys(X, forecast, "Input data `X`")
    ky = _sample_keys(Y, forecast, "Realized values `Y`")
    if isnothing(kx) || isnothing(ky)
        return Xm, Ym
    end
    perm = _row_permutation(kx, ky, forecast.sample_key)
    return Xm, isnothing(perm) ? Ym : Ym[perm, :]
end

"""
    _to_matrices(X, Y_dict::Dict{<:Forecast,<:Vector}, forecast::PredictiveModel)

Variant for realized values given as a `Dict` keyed by [`Forecast`](@ref)
variables.

This is why the entry points need no `Dict`-specific method of their own: a
`Dict` satisfies `Tables.istable`, but one keyed by variables is not a table of
series, and dispatching here — on the one small function that reads the
containers — keeps that distinction out of the public signatures. Such a `Dict`
carries no row labels either, so there is nothing to align.
"""
function _to_matrices(
    X,
    Y_dict::Dict{<:Forecast,<:Vector},
    forecast::PredictiveModel,
)
    return _to_input_matrix(X, forecast),
    _dict_to_var_indexed_matrix(Y_dict, forecast.output_variables)
end
