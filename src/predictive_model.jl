using Flux
using Statistics
import Zygote
import Functors
import Optimisers

# =============================================================================
# The unit: one architecture and the wiring around it
# =============================================================================

"""
    ForecastModel(; inputs = nothing, architecture, outputs)

One forecast model: an `architecture`, the input columns it reads and the
[`Forecast`](@ref) variables it predicts.

A forecast is described by a *vector* of these, handed to
[`set_forecast_model`](@ref), which assembles them into the
[`FullForecastModel`](@ref) stored on the [`Model`](@ref). Everything in a unit is
about the same one architecture, so there are no parallel vectors to keep
aligned and no map iteration order to know.

# Arguments

  - `inputs`: the input columns this architecture reads, as column positions
    (`[1, 3]`) or column names (`[:temp, :hour]`). A scalar is accepted for a
    one-input architecture. `nothing` means "the whole input row", which is only
    resolvable when the row width is known — either because this is the only
    unit, or because another unit selects by name and so fixes the schema.
    Writing the names here is what declares the input schema; there is no
    separate place to declare it.

  - `architecture`: anything callable on a `(features x samples)` matrix — a
    `Flux.Dense`, a `Flux.Chain`, or your own layer. It is **not** copied here,
    so that reusing one object across units can be reported as an error; it is
    deep-copied when the units are assembled, so training never touches the
    caller's object.
  - `outputs`: the [`Forecast`](@ref) variables this architecture predicts, in
    the order it produces them. Say where a variable's realized values live in
    `Y` by writing `variable => :column` for a named container, or
    `variable => position` for a matrix; either way the answer sits next to its
    variable, so there is no second ordering to line up. Left unsaid, a variable
    reads the `Y` column carrying its own declared name.

    A name and a position are two spellings of one thing, so all units of a
    forecast use one or the other. Positions are the only way to pin the layout
    of a matrix `Y`: without them its columns are read in the order the
    `Forecast` variables were declared on the [`Model`](@ref), which nothing
    states and nothing checks.

All units of one forecast must select their inputs the same way — all by
position or all by name — since a mix leaves the input schema undefined.

# Examples

One network reading two named columns:

```julia
model = ApplicationDrivenLearning.Model()
@variable(model, demand, ApplicationDrivenLearning.Forecast)

set_forecast_model(
    model,
    ForecastModel(
        inputs = [:temp, :hour],
        architecture = Flux.Dense(2 => 1),
        outputs = [demand],
    ),
)
```

One network per forecast variable, each with its own parameters, naming the `Y`
columns because the variables are called `load[1]` and `load[2]`:

```julia
@variable(model, load[1:2], ApplicationDrivenLearning.Forecast)

set_forecast_model(
    model,
    [
        ForecastModel(
            inputs = :temp,
            architecture = Flux.Dense(1 => 1),
            outputs = [load[1] => :load_a],
        ),
        ForecastModel(
            inputs = :wind,
            architecture = Flux.Dense(1 => 1),
            outputs = [load[2] => :load_b],
        ),
    ],
)
```

To reuse one architecture for several forecasts, give each unit its own copy —
`deepcopy(nn)`, or a function returning a fresh network. Passing the same object
to two units is an error rather than a silent choice between "same shape" and
"same weights".
"""
struct ForecastModel
    inputs::Union{Vector{Int},Vector{Symbol},Nothing}
    architecture::Any
    outputs::Vector{<:Forecast}
    output_columns::Union{Vector{Symbol},Vector{Int},Nothing}
end

function ForecastModel(; inputs = nothing, architecture, outputs)
    ins = _unit_inputs(inputs)
    vars, columns = _unit_outputs(outputs)
    if !isnothing(ins)
        _check_unit_arity(
            architecture,
            length(ins),
            vars,
            "This `ForecastModel`",
        )
    end
    return ForecastModel(ins, architecture, vars, columns)
end

const _UNIT_HINT =
    "`ForecastModel` now describes ONE architecture and the wiring around " *
    "it, and is built with keywords:\n\n" *
    "    ForecastModel(\n" *
    "        inputs = [:temp, :hour],\n" *
    "        architecture = Flux.Dense(2 => 1),\n" *
    "        outputs = [demand],\n" *
    "    )\n\n" *
    "Pass one, or a vector of them, to `set_forecast_model`. The constructors " *
    "that took a vector of networks and a map from input columns to forecast " *
    "variables are gone: that wiring now lives in the units, and " *
    "`set_forecast_model` assembles them into the `FullForecastModel` the " *
    "prediction loop runs."

# Loud failure for every retired signature. The old `PredictiveModel` took a
# network, or a vector of networks and a map, positionally; those shapes are gone
# and on this type they would otherwise be a bare `MethodError` - or worse, quietly
# mean something new - so each retired arity says what to write instead. Arity 4 is
# the real constructor and is deliberately absent from this list.
for n in (1, 2, 3, 5, 6, 7, 8)
    args = ntuple(_ -> :(::Any), n)
    @eval function ForecastModel($(args...); kwargs...)
        return throw(ArgumentError($_UNIT_HINT))
    end
end

"""
    _unit_inputs(inputs)

Normalize a unit's `inputs` to a `Vector{Int}`, a `Vector{Symbol}` or `nothing`.

A scalar is wrapped, so a one-input architecture can be written
`inputs = :temp`. Mixing positions and names inside one unit is rejected: the
two are resolved differently and there is no reading of `[1, :temp]` that is not
a guess. A repeated column is rejected too — it would feed two of the
architecture's inputs from one column, which is a typo far more often than it is
a request.
"""
function _unit_inputs(inputs)
    isnothing(inputs) && return nothing
    inputs isa Integer && return [Int(inputs)]
    inputs isa Symbol && return [inputs]
    if inputs isa AbstractVector && !isempty(inputs)
        if all(x -> x isa Integer, inputs)
            positions = collect(Int, inputs)
            if any(<(1), positions)
                throw(
                    ArgumentError(
                        "`inputs` must be column positions of 1 or more, got " *
                        "$(repr(inputs)).",
                    ),
                )
            end
            return _assert_distinct_inputs(positions)
        elseif all(x -> x isa Symbol, inputs)
            return _assert_distinct_inputs(collect(Symbol, inputs))
        end
    end
    throw(
        ArgumentError(
            "`inputs` must be column positions (`[1, 3]`), column names " *
            "(`[:temp, :hour]`), a single position or name, or `nothing` for " *
            "the whole input row. Got $(repr(inputs)).",
        ),
    )
end

"""
    _assert_distinct_inputs(inputs)

Return `inputs` unless it names the same column twice.

Kept as its own check so that the message can say which column, and because the
positional and the named branch of [`_unit_inputs`](@ref) both need it.
"""
function _assert_distinct_inputs(inputs)
    allunique(inputs) && return inputs
    repeated = unique([c for c in inputs if count(isequal(c), inputs) > 1])
    throw(
        ArgumentError(
            "`inputs` reads the same input column more than once: " *
            "$(join(repr.(repeated), ", ")). Two of the architecture's inputs " *
            "would be fed from one column.",
        ),
    )
end

"""
    _unit_outputs(outputs)

Split a unit's `outputs` into its [`Forecast`](@ref) variables and the `Y` columns
holding them, or `nothing` when no entry said where its column is.

Entries are a `Forecast` variable, `variable => :column` or
`variable => position`, and the meaning is the same in all three: where in `Y` a
variable's realized values are.

A name and a position are two spellings of that one thing, so a unit uses one or
the other. Names may be left out of some entries and are then filled from the
variables' own declared names, which is what a bare entry asks for. A position
cannot be filled in: a variable's name says nothing about where a matrix keeps it,
and the row order of a prediction is not known to a unit at all.
"""
function _unit_outputs(outputs)
    entries = _output_entries(outputs)
    if isempty(entries)
        throw(
            ArgumentError(
                "`outputs` must list at least one `Forecast` variable.",
            ),
        )
    end
    vars = Vector{Any}(undef, length(entries))
    columns = Vector{Union{Symbol,Int,Nothing}}(undef, length(entries))
    for (i, entry) in enumerate(entries)
        if entry isa Forecast
            vars[i] = entry
            columns[i] = nothing
        elseif entry isa Pair && entry.first isa Forecast
            vars[i] = entry.first
            columns[i] = _unit_output_column(entry.second, i)
        else
            throw(ArgumentError(_bad_output_entry_message(entry, i)))
        end
    end
    variables = identity.(vars)  # narrow Vector{Any} to Vector{Forecast{...}}
    if !allunique(variables)
        throw(
            ArgumentError(
                "`outputs` lists the same forecast variable more than once: " *
                "$(join(_forecast_labels(variables), ", ")). A variable is " *
                "predicted by exactly one architecture.",
            ),
        )
    end
    return variables, _unit_output_columns(variables, columns)
end

"""
    _output_entries(outputs)

The entries of a unit's `outputs` as a plain vector, whether it was written as a
vector of them, a one-dimensional JuMP container, or a single variable.

`collect` is not enough. A container declared over non-integer axes -
`@variable(model, d[[:a, :b]], Forecast)` - is an `AbstractVector` that supports
neither `collect` nor iteration, so `collect` on it fails inside `Base` with a
`MethodError` about `CartesianIndices` and says nothing about `outputs`.
`Iterators.flatten` reaches its elements and flattens exactly one level, so a
container nested *inside* a vector still arrives as one entry - and is reported as
one, by [`_bad_output_entry_message`](@ref).
"""
function _output_entries(outputs)
    outputs isa AbstractVector || return Any[outputs]
    return collect(Iterators.flatten((outputs,)))
end

"""
    _bad_output_entry_message(entry, i)

Why an `outputs` entry is not one the unit can read.

A whole container where one variable belongs is the mistake worth its own text:
the generic message would report the type and leave the reader to work out that
the brackets are one level too many.
"""
function _bad_output_entry_message(entry, i::Int)
    inner = entry isa Pair ? entry.first : entry
    if inner isa AbstractArray && eltype(inner) <: Forecast
        fix = if entry isa Pair
            "A `Y` column belongs to one variable, so give each variable its " *
            "own entry: `d[1] => :col_a, d[2] => :col_b`."
        elseif ndims(inner) == 1
            "A container declared with `@variable(model, d[1:2], Forecast)` " *
            "is already the list of variables: write `outputs = d`, or list " *
            "its elements as `outputs = [d[1], d[2]]`."
        else
            "A container of more than one dimension has no obvious order to " *
            "produce its variables in, so flatten it deliberately: " *
            "`outputs = vec(d)`."
        end
        return "`outputs` entry $i holds $(length(inner)) forecast " *
               "variables ($(typeof(inner))), not one. " *
               fix
    end
    return "`outputs` entry $i is a $(typeof(entry)); every entry must be a " *
           "`Forecast` variable, `variable => :column` naming the `Y` column " *
           "that holds it, or `variable => position` giving that column's " *
           "position in a matrix `Y`."
end

"""
    _unit_output_column(selector, i)

The `Y` column an `outputs` entry points at: a `Symbol` naming it or an `Int`
giving its position in a matrix.

`Bool` is excluded deliberately. It is an `Integer`, so `variable => true` would
otherwise be accepted as column 1 - a reading nobody wrote.
"""
function _unit_output_column(selector, i::Int)
    if selector isa Symbol
        return selector
    elseif selector isa Integer && !(selector isa Bool)
        if selector < 1
            throw(
                ArgumentError(
                    "`outputs` entry $i gives the `Y` column position " *
                    "$selector; positions start at 1.",
                ),
            )
        end
        return Int(selector)
    end
    throw(
        ArgumentError(
            "`outputs` entry $i gives the `Y` column as $(repr(selector)). " *
            "Write `variable => :column` to address it by name, or " *
            "`variable => 3` to address it by position in a matrix `Y`.",
        ),
    )
end

"""
    _unit_output_columns(variables, columns)

Reduce a unit's per-entry selectors to one spelling: a `Vector{Symbol}` of column
names, a `Vector{Int}` of column positions, or `nothing` when every entry was bare.
"""
function _unit_output_columns(variables, columns)
    positional = findfirst(c -> c isa Int, columns)
    named = findfirst(c -> c isa Symbol, columns)
    if !isnothing(positional) && !isnothing(named)
        throw(
            ArgumentError(
                "`outputs` entry $named names its `Y` column and entry " *
                "$positional gives its position. A name and a position are two " *
                "spellings of the same thing, so one unit uses one or the other.",
            ),
        )
    end
    if !isnothing(positional)
        bare = findfirst(isnothing, columns)
        if !isnothing(bare)
            label = _forecast_labels(variables)[bare]
            throw(
                ArgumentError(
                    "`outputs` entry $positional gives its `Y` column by " *
                    "position but entry $bare gives none. Unlike a name, a " *
                    "position cannot be filled in from the variable: $label " *
                    "says nothing about which column of a matrix holds it. " *
                    "Give every entry a position, or address the columns by name.",
                ),
            )
        end
        positions = Int[c for c in columns]
        if !allunique(positions)
            throw(
                ArgumentError(
                    "`outputs` gives the same `Y` column position more than " *
                    "once: $(join(string.(positions), ", ")). Two variables " *
                    "cannot read one column of realized values.",
                ),
            )
        end
        return positions
    end
    isnothing(named) && return nothing
    resolved = Vector{Symbol}(undef, length(columns))
    for i in eachindex(columns)
        if isnothing(columns[i])
            fallback = _forecast_base_name(variables[i])
            if isempty(fallback)
                throw(
                    ArgumentError(
                        "`outputs` names the `Y` column of some variables but " *
                        "not of entry $i, which is an anonymous forecast " *
                        "variable and so has no name of its own to fall back " *
                        "on. Name every column, or none.",
                    ),
                )
            end
            resolved[i] = Symbol(fallback)
        else
            resolved[i] = columns[i]
        end
    end
    return resolved
end

"""
    _forecast_labels(variables)

Printable names for a collection of [`Forecast`](@ref) variables, for use in
error messages. Anonymous variables show as `<anonymous>`.
"""
function _forecast_labels(variables)
    return [
        isempty(_forecast_base_name(v)) ? "<anonymous>" :
        _forecast_base_name(v) for v in variables
    ]
end

"""
    _probe_eltype(architecture)

Element type to build the shape probe with: that of the architecture's own first
parameter, so the probe does not provoke the eltype promotion warnings Flux emits
when a `Float32` layer is handed `Float64` data. `Float64` when the architecture
has no parameters at all.
"""
function _probe_eltype(architecture)
    params = Flux.trainables(architecture)
    isempty(params) && return Float64
    return eltype(first(params))
end

"""
    _check_unit_arity(architecture, n_in::Int, outputs, what::String)

Throw unless `architecture` maps `n_in` inputs to exactly `length(outputs)`
outputs.

Checked by *applying* the architecture to an `(n_in x 1)` matrix of zeros rather
than by reading `layer.weight`, for two reasons. It works for anything callable,
where reading weights only works for `Flux.Chain` and `Flux.Dense` and would have
to refuse every other architecture. And it is the same call the prediction loop is
about to make — `nn(X[input_idx, :])` — so an architecture that cannot answer it
here cannot be trained either, and saying so at construction beats a
`DimensionMismatch` from inside the sample loop.

The probe runs on a deep copy, so a stateful layer does not carry the probe's
state into training.
"""
function _check_unit_arity(architecture, n_in::Int, outputs, what::String)
    probe = deepcopy(architecture)
    x = zeros(_probe_eltype(architecture), n_in, 1)
    prediction = try
        probe(x)
    catch err
        throw(
            ArgumentError(
                "$what could not apply its architecture " *
                "($(typeof(architecture))) to the $(n_in)-input sample its " *
                "`inputs` describe:\n" *
                sprint(showerror, err) *
                "\n\nThe architecture has to be callable on a " *
                "`(features x samples)` matrix, which is how the prediction " *
                "loop calls it. Check that `inputs` lists as many columns as " *
                "the architecture takes.",
            ),
        )
    end
    n_out = size(prediction, 1)
    if n_out != length(outputs)
        throw(
            ArgumentError(
                "$what has an architecture ($(typeof(architecture))) that " *
                "produces $n_out output(s) from $n_in input(s), but lists " *
                "$(length(outputs)) forecast variable(s): " *
                "$(join(_forecast_labels(outputs), ", ")).",
            ),
        )
    end
    return nothing
end

# =============================================================================
# The resolved unit: what the prediction loop actually runs
# =============================================================================

"""
    ResolvedUnit(inputs, architecture, outputs, rows)

One [`ForecastModel`](@ref) with its wiring resolved to integers: the input
columns it reads as *positions*, and the rows of a prediction it writes as
*positions*.

Internal, and the payload of a [`FullForecastModel`](@ref). It exists so that the
prediction loop runs without looking anything up: `rows` answers "where do this
architecture's outputs go" once, when the forecast is assembled, instead of on
every forward pass.

`architecture` is the only trainable field, and `Functors` rebuilds the struct
positionally on every `Optimisers.update!`, so this constructor is the default
all-positional one and does no work.

# Arguments

  - `inputs::Vector{Int}`: positions of the input columns this architecture
    reads, in the order it reads them.
  - `architecture`: anything callable on a `(features x samples)` matrix.
  - `outputs::Vector{<:Forecast}`: the variables it predicts, in the order it
    produces them.
  - `rows::Vector{Int}`: for each of `outputs`, the row of a prediction it is
    written to. Parallel to `outputs`, and kept next to them because reordering a
    prediction — which [`set_forecast_model`](@ref) does — means recomputing
    these from those.
"""
struct ResolvedUnit
    inputs::Vector{Int}
    architecture::Any
    outputs::Vector{<:Forecast}
    rows::Vector{Int}
end

Functors.@functor ResolvedUnit (architecture,)

"""
    _resolved_units(units, indices, output_variables)

Resolve [`ForecastModel`](@ref) units against the row order of a prediction.

The architecture is deep-copied here and nowhere else. Here, because this is the
one place a caller's object enters the container, so one copy leaves it untouched
without paying for a copy on the rebuild path. Nowhere else, because a unit also
holds `Forecast` variables, and deep-copying one of those clones the whole JuMP
model it belongs to — yielding variables that compare unequal to the originals,
and so resolve to no row at all.
"""
function _resolved_units(units, indices, output_variables)
    return ResolvedUnit[
        ResolvedUnit(
            indices[i],
            deepcopy(units[i].architecture),
            collect(units[i].outputs),
            _find_elements_position(output_variables, units[i].outputs),
        ) for i in eachindex(units)
    ]
end

"""
    _reindexed_units(units, output_variables)

The same `ResolvedUnit`s with their `rows` recomputed against a new row
order.

Used by [`set_forecast_model`](@ref), which reorders a prediction to follow the
variables declared on the [`Model`](@ref). The architectures are carried over as
they are, not copied: they were copied when the units were assembled, and copying
again here would detach them from anything already set up on them.
"""
function _reindexed_units(units, output_variables)
    return ResolvedUnit[
        ResolvedUnit(
            unit.inputs,
            unit.architecture,
            unit.outputs,
            _find_elements_position(output_variables, unit.outputs),
        ) for unit in units
    ]
end

# =============================================================================
# The container: the forecast of a whole Model
# =============================================================================

"""
    FullForecastModel(units, output_variables, input_size, output_size; input_names, output_columns, sample_key)

The assembled forecast of a [`Model`](@ref): every architecture, the input
columns each reads, the [`Forecast`](@ref) variables each predicts, and the
schema of the data containers.

Built by [`set_forecast_model`](@ref) from a vector of [`ForecastModel`](@ref)
units, which is how it should be constructed — this explicit form exists because
the fields are what the prediction loop, the optimizers and the parallel backends
work on, and because `Functors` rebuilds the struct positionally.

# Arguments

  - `units`: the `ResolvedUnit`s, one per [`ForecastModel`](@ref) the
    forecast was assembled from. Each carries its own architecture, the input
    columns it reads and the prediction rows it writes, so there are no parallel
    vectors to keep in step. The architectures are deep-copied when the units are
    assembled, so the caller's objects are left untouched.
  - `output_variables`: the forecast variables in the order the model produces
    them, i.e. the row order of a prediction. Every row is written by exactly one
    unit, which is checked here.
  - `input_size::Int`: number of columns the input has.
  - `output_size::Int`: number of forecast variables.
  - `input_names::Union{Vector{Symbol},Nothing}`: name of each input column, in
    the order the model expects them. This is the model's declared input schema:
    it is what lets a table input be matched by name instead of by column order.
    `nothing` means the model declares no schema, and a table input is then
    rejected rather than guessed at.
  - `output_columns::Union{Vector{Symbol},Vector{Int},Nothing}`: where in `Y`
    each variable of `output_variables` is, in that same order — as table column
    names, or as column positions in a matrix. `nothing` means the declared names
    of the [`Forecast`](@ref) variables themselves are used. Assembled from the
    `variable => :column` and `variable => position` pairs of the units, which is
    also why only one of the two spellings can be present.
  - `sample_key::Union{Symbol,Nothing}`: name of the column that identifies a
    *row*, the way `input_names` identifies a column — a timestamp or an id.
    When both `X` and `Y` are tables carrying it, the realized values are looked
    up by key for each row of `X` instead of being trusted to arrive in the same
    order. `nothing` means rows are matched by position.
"""
struct FullForecastModel
    units::Vector{ResolvedUnit}
    output_variables::Vector{<:Forecast}
    input_size::Int
    output_size::Int
    input_names::Union{Vector{Symbol},Nothing}
    output_columns::Union{Vector{Symbol},Vector{Int},Nothing}
    sample_key::Union{Symbol,Nothing}

    function FullForecastModel(
        units::AbstractVector,
        output_variables::Vector{<:Forecast},
        input_size::Int,
        output_size::Int;
        input_names::Union{Vector{Symbol},Nothing} = nothing,
        output_columns::Union{Vector{Symbol},Vector{Int},Nothing} = nothing,
        sample_key::Union{Symbol,Nothing} = nothing,
    )
        # every construction path funnels through here, including the positional
        # rebuild `Functors` performs on each optimiser step, so this is where the
        # invariants that must never be violated live. They are all O(1) or a
        # single pass over the forecast variables
        if length(output_variables) != output_size
            throw(
                DimensionMismatch(
                    "`output_variables` has $(length(output_variables)) " *
                    "entry(ies) but `output_size` is $output_size.",
                ),
            )
        elseif !allunique(output_variables)
            throw(
                ArgumentError(
                    "`output_variables` repeats a forecast variable, so one " *
                    "row of every prediction would be written twice and " *
                    "another left uninitialized.",
                ),
            )
        end
        _assert_rows_partition(units, output_size)
        _check_names(input_names, input_size, "input_names", "takes", "input")
        _check_output_columns(output_columns, output_size)
        return new(
            units,
            output_variables,
            input_size,
            output_size,
            input_names,
            output_columns,
            sample_key,
        )
    end
end

"""
    FullForecastModel(units, output_variables, input_size, output_size, input_names, output_columns, sample_key)

All-positional form of the constructor above.

Required, not merely convenient: `Functors.@functor` rebuilds the struct by
calling its constructor with every field in order, which is what happens on every
`Optimisers.update!` — so a field that is only reachable by keyword would make
the model untrainable.
"""
function FullForecastModel(
    units::AbstractVector,
    output_variables::Vector{<:Forecast},
    input_size::Int,
    output_size::Int,
    input_names::Union{Vector{Symbol},Nothing},
    output_columns::Union{Vector{Symbol},Vector{Int},Nothing},
    sample_key::Union{Symbol,Nothing},
)
    return FullForecastModel(
        units,
        output_variables,
        input_size,
        output_size;
        input_names = input_names,
        output_columns = output_columns,
        sample_key = sample_key,
    )
end

"""
    _assert_rows_partition(units, output_size)

Throw unless the units write every row of a prediction exactly once.

This is the invariant the two parallel vectors this container used to hold could
not express, and whose violation was silent: a row no unit writes is read back
out of an uninitialized `Zygote.Buffer`, and a row two units write loses one of
the two predictions. Both surfaced as a cost computed from garbage rather than as
an error.
"""
function _assert_rows_partition(units, output_size::Int)
    written = zeros(Int, output_size)
    for (i, unit) in enumerate(units)
        if length(unit.rows) != length(unit.outputs)
            throw(
                DimensionMismatch(
                    "Unit $i predicts $(length(unit.outputs)) variable(s) but " *
                    "lists $(length(unit.rows)) prediction row(s); they are " *
                    "read in step.",
                ),
            )
        end
        for row in unit.rows
            if (row < 1) || (row > output_size)
                throw(
                    ArgumentError(
                        "Unit $i writes prediction row $row, which is outside " *
                        "the $output_size row(s) the model produces.",
                    ),
                )
            end
            written[row] += 1
        end
    end
    for row in eachindex(written)
        written[row] == 1 && continue
        message = if written[row] == 0
            "No unit writes prediction row $row, so it would be read back " *
            "uninitialized."
        else
            "$(written[row]) units write prediction row $row, so one of their " *
            "predictions would be lost."
        end
        throw(ArgumentError(message))
    end
    return nothing
end

"""
    _check_output_columns(columns, output_size)

Throw unless `columns` is `nothing` or gives each of `output_size` outputs a
distinct place in `Y` — a column name, or a column position of at least 1.

Both spellings are checked here rather than at assembly because `Functors`
rebuilds the container positionally on every optimiser step, and a repeated
column would feed two forecast variables from one series.
"""
function _check_output_columns(
    columns::Union{Vector{Symbol},Vector{Int},Nothing},
    output_size::Int,
)
    isnothing(columns) && return nothing
    if columns isa Vector{Symbol}
        return _check_names(
            columns,
            output_size,
            "output_columns",
            "produces",
            "output",
        )
    elseif length(columns) != output_size
        throw(
            ArgumentError(
                "`output_columns` gives $(length(columns)) `Y` column " *
                "position(s) but the predictive model produces $output_size " *
                "output(s): $(join(string.(columns), ", ")).",
            ),
        )
    elseif !allunique(columns)
        throw(
            ArgumentError(
                "`output_columns` must not repeat a `Y` column position, got " *
                "$(join(string.(columns), ", ")).",
            ),
        )
    elseif minimum(columns) < 1
        throw(
            ArgumentError(
                "`output_columns` gives the `Y` column position " *
                "$(minimum(columns)); positions start at 1.",
            ),
        )
    end
    return nothing
end

"""
    _output_width(forecast)

Number of columns a matrix `Y` must have for this forecast to read it.

`output_size` unless the units gave their `Y` columns by position, in which case
the widest position they name - the columns in between may be present and unread,
exactly as they may be for `inputs`.
"""
function _output_width(forecast::FullForecastModel)
    columns = forecast.output_columns
    return columns isa Vector{Int} ? maximum(columns) : forecast.output_size
end

"""
    _check_names(names, n, what, verb, noun)

Throw an `ArgumentError` unless `names` is `nothing` or a vector of `n` distinct
symbols. Duplicates matter as much as the count: two entries with the same name
would make one column feed two different slots.
"""
function _check_names(
    names::Union{Vector{Symbol},Nothing},
    n::Int,
    what::String,
    verb::String,
    noun::String,
)
    isnothing(names) && return nothing
    if length(names) != n
        throw(
            ArgumentError(
                "`$what` has $(length(names)) entry(ies) but the predictive " *
                "model $verb $n $noun(s): $(join(string.(names), ", ")).",
            ),
        )
    elseif !allunique(names)
        throw(
            ArgumentError(
                "`$what` must not repeat a name, got " *
                "$(join(string.(names), ", ")).",
            ),
        )
    end
    return nothing
end

"""
    FullForecastModel(units::AbstractVector{<:ForecastModel}; sample_key = nothing)

Assemble [`ForecastModel`](@ref) units into the forecast of a whole model.

This is what [`set_forecast_model`](@ref) calls, and the only construction path
worth using. Everything the explicit constructor takes separately is derived here
from one source: each unit's input columns and output rows are resolved to
positions once, so nothing downstream has to look them up again.

The input schema is derived from the units and declared nowhere else: a unit whose
`inputs` are column *names* contributes them, in order of first appearance, and
that is what `input_names` becomes. Units that select by position declare no
schema, so a table input is refused rather than read by column order.

`sample_key` is the exception, and the only keyword here: it names the column that
identifies a *row* rather than a column, so no unit owns it — there is one `X` and
one `Y`, matched row-wise against each other.
"""
function FullForecastModel(
    units::AbstractVector{<:ForecastModel};
    sample_key::Union{Symbol,Nothing} = nothing,
)
    if isempty(units)
        throw(
            ArgumentError(
                "A forecast needs at least one `ForecastModel`, got an " *
                "empty vector.",
            ),
        )
    end
    _assert_distinct_architectures(units)
    indices, names, input_size = _resolve_unit_inputs(units)
    # deferred until the row width is known: a unit reading the whole row could
    # not be shape-checked at its own construction
    for (i, unit) in enumerate(units)
        if isnothing(unit.inputs)
            _check_unit_arity(
                unit.architecture,
                length(indices[i]),
                unit.outputs,
                "Unit $i",
            )
        end
    end
    output_variables = reduce(vcat, [collect(unit.outputs) for unit in units])
    _assert_no_repeated_outputs(units)
    return FullForecastModel(
        _resolved_units(units, indices, output_variables),
        output_variables,
        input_size,
        length(output_variables);
        input_names = names,
        output_columns = _assemble_output_columns(units),
        sample_key = sample_key,
    )
end

# A vector holding something other than units: name the offending entry rather
# than surface a `MethodError` that says only that no method matched. Left
# undocumented on purpose - its signature is the one the call operator's docstring
# occupies, and the documented constructor is the typed one above.
function FullForecastModel(units::AbstractVector; kwargs...)
    for (i, unit) in enumerate(units)
        if !(unit isa ForecastModel)
            throw(
                ArgumentError(
                    "Entry $i of the forecast is a $(typeof(unit)); every " *
                    "entry must be a `ForecastModel`. " *
                    _UNIT_HINT,
                ),
            )
        end
    end
    return FullForecastModel(collect(ForecastModel, units); kwargs...)
end

"""
    _assert_distinct_architectures(units)

Throw unless every unit has its own `architecture` object.

Two units holding the *same* object would share one set of parameters, which is a
different model from two units of the same shape — and only one of the two
readings can be what a caller who reused a variable meant. Rather than pick, this
says so and names the fix.
"""
function _assert_distinct_architectures(units)
    for i in eachindex(units), j in eachindex(units)
        j <= i && continue
        if units[i].architecture === units[j].architecture
            throw(
                ArgumentError(
                    "Units $i and $j hold the same `architecture` object " *
                    "($(typeof(units[i].architecture))), so they would train " *
                    "one set of parameters between them. Give each unit its " *
                    "own architecture — pass `deepcopy(nn)`, or build a fresh " *
                    "network per unit — so that each has its own weights.",
                ),
            )
        end
    end
    return nothing
end

"""
    _assert_no_repeated_outputs(units)

Throw unless every [`Forecast`](@ref) variable is predicted by at most one unit.

Within a unit this is already checked; across units it is the other half of the
same rule. A variable claimed twice would have its prediction row written twice
and leave another row of the buffer untouched, which surfaces as a cost computed
from uninitialized memory rather than as an error.
"""
function _assert_no_repeated_outputs(units)
    owner = Dict{Any,Int}()
    for (i, unit) in enumerate(units)
        for variable in unit.outputs
            previous = get(owner, variable, 0)
            if previous != 0
                label = _forecast_labels([variable])[1]
                throw(
                    ArgumentError(
                        "Forecast variable $label is predicted by both unit " *
                        "$previous and unit $i. Each variable is predicted by " *
                        "exactly one architecture.",
                    ),
                )
            end
            owner[variable] = i
        end
    end
    return nothing
end

"""
    _assemble_output_columns(units)

Where in `Y` every forecast variable is, in unit order, or `nothing` when no unit
said.

All units have to answer the same way, since `Y` is one container: names cannot be
resolved against positions or the other way round. Under the name spelling a unit
that named nothing contributes the declared names of its own variables, so naming
the columns of one unit does not force the others to repeat what their variables
already say. Under the position spelling there is no such fallback, so every unit
has to give one.
"""
function _assemble_output_columns(units)
    named = findfirst(u -> u.output_columns isa Vector{Symbol}, units)
    positional = findfirst(u -> u.output_columns isa Vector{Int}, units)
    if !isnothing(named) && !isnothing(positional)
        throw(
            ArgumentError(
                "Unit $named addresses its `Y` columns by name and unit " *
                "$positional by position. Every unit of one forecast has to " *
                "address them the same way, since there is one `Y`.",
            ),
        )
    end
    if !isnothing(positional)
        columns = Int[]
        for (i, unit) in enumerate(units)
            if isnothing(unit.output_columns)
                throw(
                    ArgumentError(
                        "Unit $positional gives its `Y` columns by position " *
                        "but unit $i gives none. A position cannot be filled " *
                        "in from a variable's name, so either every unit gives " *
                        "one or the columns are addressed by name.",
                    ),
                )
            end
            append!(columns, unit.output_columns)
        end
        return columns
    end
    isnothing(named) && return nothing
    names = Symbol[]
    for (i, unit) in enumerate(units)
        if !isnothing(unit.output_columns)
            append!(names, unit.output_columns)
            continue
        end
        for variable in unit.outputs
            fallback = _forecast_base_name(variable)
            if isempty(fallback)
                throw(
                    ArgumentError(
                        "Some units name the `Y` column of their outputs, so " *
                        "every column has to be named, but unit $i predicts " *
                        "an anonymous forecast variable with no name of its " *
                        "own to fall back on. Write its column as " *
                        "`variable => :column`.",
                    ),
                )
            end
            push!(names, Symbol(fallback))
        end
    end
    return names
end

"""
    _resolve_unit_inputs(units)

Resolve every unit's `inputs` to column *positions*, and return them together
with the input schema those names imply and the input width.

Positions are resolved once here, at construction, so that everything downstream —
the call operator, [`apply_gradient!`](@ref), the optimizers — keeps working on
the integer representation it always used. The names survive only as
`input_names`, which is what selects the columns of a table input.

The schema is whatever the units say and nothing else: writing `inputs` as names
declares it, in order of first appearance, and writing them as positions declares
none, so a table input is then refused rather than matched by order. There is no
second place to say it.

All units must select the same way; mixing positions and names leaves the input
schema undefined, since there would be no order to resolve the names against.
"""
function _resolve_unit_inputs(units)
    named = findfirst(u -> u.inputs isa Vector{Symbol}, units)
    positional = findfirst(u -> u.inputs isa Vector{Int}, units)
    if !isnothing(named) && !isnothing(positional)
        throw(
            ArgumentError(
                "Unit $positional selects its inputs by position and unit " *
                "$named by name. Every unit of one forecast has to select the " *
                "same way, since the input schema is shared.",
            ),
        )
    end
    _assert_whole_row_resolvable(units, named)
    names, positions = if isnothing(named)
        nothing, _positional_positions(units)
    else
        _resolve_named_inputs(units)
    end
    input_size =
        isnothing(names) ? _positional_input_size(units) : length(names)
    # a unit reading the whole row: only resolvable now that the width is known
    for i in eachindex(units)
        if isnothing(units[i].inputs)
            positions[i] = collect(1:input_size)
        end
    end
    return positions, names, input_size
end

"""
    _assert_whole_row_resolvable(units, named)

Throw unless a unit reading the whole input row (`inputs = nothing`) has a row
width to read.

It has one when it is the only unit, since the width is then its architecture's,
or when some unit selects by name, since the schema is then the columns those
units name. What it does not have is a width alongside other *positional* units:
the largest column they read is a lower bound on the row, not the row.
"""
function _assert_whole_row_resolvable(units, named)
    whole = findfirst(u -> isnothing(u.inputs), units)
    if isnothing(whole) || length(units) == 1 || !isnothing(named)
        return nothing
    end
    throw(
        ArgumentError(
            "Unit $whole reads the whole input row (`inputs = nothing`) " *
            "alongside $(length(units) - 1) other unit(s) that select by " *
            "position, so the width of that row is not determined - the widest " *
            "column they read is a lower bound on it, not the row. List its " *
            "`inputs` explicitly.",
        ),
    )
end

function _positional_positions(units)
    positions = Vector{Vector{Int}}(undef, length(units))
    for i in eachindex(units)
        # `copy`, so that the stored map does not alias a unit's own vector
        positions[i] =
            isnothing(units[i].inputs) ? Int[] : copy(units[i].inputs)
    end
    return positions
end

function _resolve_named_inputs(units)
    names = Symbol[]
    for unit in units
        isnothing(unit.inputs) && continue
        for name in unit.inputs
            name in names || push!(names, name)
        end
    end
    position = Dict(name => i for (i, name) in enumerate(names))
    positions = Vector{Vector{Int}}(undef, length(units))
    for i in eachindex(units)
        positions[i] = if isnothing(units[i].inputs)
            Int[]
        else
            [position[name] for name in units[i].inputs]
        end
    end
    return names, positions
end

"""
    _positional_input_size(units)

Input width of a forecast whose units select by position: the largest column any
unit reads, or — for the single unit that reads the whole row — the input size of
its own architecture.
"""
function _positional_input_size(units)
    listed = [unit for unit in units if !isnothing(unit.inputs)]
    isempty(listed) || return maximum(maximum(u.inputs) for u in listed)
    architecture = units[1].architecture
    try
        return _network_io_sizes(architecture)[1]
    catch
        throw(
            ArgumentError(
                "The only unit reads the whole input row " *
                "(`inputs = nothing`), so its width has to come from its " *
                "architecture, but the input size of a " *
                "$(typeof(architecture)) cannot be read. List its `inputs`.",
            ),
        )
    end
end

"""
    _network_io_sizes(network)

Input and output size of a single Flux network, read off its parameterised
layers.
"""
function _network_io_sizes(network::Flux.Chain)
    param_layers = [layer for layer in network if _has_params(layer)]
    return size(param_layers[1].weight, 2), size(param_layers[end].weight, 1)
end

_network_io_sizes(network::Flux.Dense) = reverse(size(network.weight))

"""
    Flux.trainable(model::FullForecastModel)

Make FullForecastModel compatible with Flux's training interface by specifying
that only the units field is trainable - and, through `ResolvedUnit`'s own
functor, only the architecture within each unit.
"""
Flux.trainable(model::FullForecastModel) = (units = model.units,)

# Tells Flux to only look at the 'units' field when setting up or traversing
Functors.@functor FullForecastModel (units,)

"""
    _find_elements_position(vec, elements)

Return the position in `vec` of each entry of `elements`. Used to map the
forecast variables a unit predicts onto the rows of the prediction matrix -
once, when the units are resolved, and never on the prediction path itself.
Entries not present in `vec` yield `nothing`.
"""
function _find_elements_position(vec, elements)
    return [findfirst(i -> i == j, vec) for j in elements]
end

"""
    (model::FullForecastModel)(X::AbstractMatrix)

Predict the output of the model for a given input matrix.
"""
function (model::FullForecastModel)(X::AbstractMatrix)
    pred_size = size(X, 2)  # length of the input data
    # buffer to store the predicted output
    Yhat = Zygote.Buffer(
        Matrix{eltype(X)}(undef, model.output_size, pred_size),
        (model.output_size, pred_size),
    )

    # every unit already knows the columns it reads and the rows it writes, so
    # this is the whole prediction - no lookup, and nothing to keep in step
    for unit in model.units
        Yhat[unit.rows, :] = unit.architecture(X[unit.inputs, :])
    end
    return copy(Yhat)
end

"""
    (model::FullForecastModel)(x::AbstractVector)

Predict the output of the model for a given input vector.
"""
function (model::FullForecastModel)(x::AbstractVector)
    # buffer to store the predicted output
    yhat = Zygote.Buffer(
        Vector{eltype(x)}(undef, model.output_size),
        model.output_size,
    )

    for unit in model.units
        yhat[unit.rows] = unit.architecture(x[unit.inputs])
    end
    return copy(yhat)
end

"""
    extract_params(model)

Extract the parameters of a FullForecastModel into a single vector.
"""
function extract_params(model::FullForecastModel)
    @timeit_debug _TIMER "extract_params" begin
        # NOTE: keep the splat. `reduce(vcat, xs)` returns `xs[1]` untouched
        # when there is a single network, and `_extract_flux_params` itself
        # returns `vec(p)` - an alias of the live weights - when that network
        # has a single trainable array. `vcat` always copies, which is what
        # callers such as `best_θ` in the gradient loop rely on.
        return vcat(
            [
                _extract_flux_params(unit.architecture) for unit in model.units
            ]...,
        )
    end
end

"""
    apply_params(model, θ)

Return model after fixing the parameters from an adequate vector of parameters.
"""
function apply_params(model::FullForecastModel, θ)
    @timeit_debug _TIMER "apply_params" begin
        return _fix_flux_params_multi_model(
            [unit.architecture for unit in model.units],
            θ,
        )
    end
end

"""
    apply_gradient!(model, dCdy, X, opt_state)

Apply per-sample cost gradients to the model parameters.

The optimization layer provides `dCdy`, the gradient of the assessment cost with
respect to the forecasts, for each sample. Because the optimization step itself
is not differentiable by the AD backend, these gradients are propagated through
the forecast model with a linear surrogate loss whose parameter-gradient equals
the chain-rule term `(1/T) Σₜ (dC/dŷₜ)·(dŷₜ/dθ)`.

...

# Arguments

  - `model::FullForecastModel`: model to be updated.
  - `dCdy::AbstractMatrix{<:Real}`: per-sample cost gradients, size
    `(T, output_size)`, with row `t` aligned to sample `t` (row `t` of `X`).
  - `X::AbstractMatrix{<:Real}`: input data, size `(T, input_size)`.
  - `opt_state`: Optimisers optimisation state.
    ...
"""
function apply_gradient!(
    model::FullForecastModel,
    dCdy::AbstractMatrix{<:Real},
    X::AbstractMatrix{<:Real},
    opt_state,
)
    grad = _parameter_gradient(model, dCdy, X)
    return @timeit_debug _TIMER "optimiser_update" Optimisers.update!(
        opt_state,
        model,
        grad,
    )
end

"""
    _parameter_gradient(model::FullForecastModel, dCdy, X)

Gradient of the assessed cost with respect to the model parameters, as a
structure mirroring `model`.

The surrogate loss is the one described on [`apply_gradient!`](@ref): its
parameter-gradient equals the chain-rule term `(1/T) Σₜ (dC/dŷₜ)·(dŷₜ/dθ)`, so
the result is `dC/dθ` for the *mean* cost over the `T` samples — matching what
`compute_cost` returns with `aggregate = true`.
"""
function _parameter_gradient(
    model::FullForecastModel,
    dCdy::AbstractMatrix{<:Real},
    X::AbstractMatrix{<:Real},
)
    surrogate_loss(m, X) = sum(dCdy' .* m(X')) / size(X, 1)
    return @timeit_debug _TIMER "zygote_backward" Zygote.gradient(
        surrogate_loss,
        model,
        X,
    )[1]
end

"""
    _flat_parameter_gradient(model::FullForecastModel, dCdy, X)

Gradient of the assessed cost with respect to the parameters, as a *flat vector*
laid out exactly as [`extract_params`](@ref) lays out the parameters.

That correspondence is load-bearing: `Optim` and NLopt take `θ` and `g` as
parallel vectors and cannot notice a permutation between them, so a mismatch
would not error — it would train towards the wrong place.

It is therefore obtained by construction rather than by assumption.
`Optimisers.destructure` returns `(flat, re)` where `re` rebuilds the model from
a flat vector; differentiating `θ -> loss(re(θ))` yields a gradient in exactly
`flat`'s layout, whatever that layout happens to be, because the same `re`
defines both directions.

The tempting alternative — walk the model's `trainables` and the gradient's
`trainables` side by side — is wrong, and silently so. A gradient is a plain
nested `NamedTuple` mirror with no `Flux.trainable` method of its own, so
`trainables` keeps leaves the model excludes: for a `Dense` the mirror yields
three arrays (`weight`, `bias`, `σ`) against the model's two.

What still has to hold is that `destructure`'s layout agrees with
`extract_params`'; both walk per network, then per trainable leaf, then `vec`,
and `test/test_solver_backends.jl` pins the equality. This is also why no two
units may hold the same architecture object: `destructure` counts a shared array
once while `extract_params` counts it per network, and the two lengths would
disagree.
"""
function _flat_parameter_gradient(
    model::FullForecastModel,
    dCdy::AbstractMatrix{<:Real},
    X::AbstractMatrix{<:Real},
)
    flat, re = Optimisers.destructure(model)
    surrogate_loss(θ) = sum(dCdy' .* re(θ)(X')) / size(X, 1)
    return @timeit_debug _TIMER "zygote_backward_flat" Zygote.gradient(
        surrogate_loss,
        flat,
    )[1]
end
