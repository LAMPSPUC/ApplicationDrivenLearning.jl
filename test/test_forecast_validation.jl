# Everything a forecast refuses to be built as.
#
# One assertion per rule, and where the message names the offending thing the
# message is checked too: "errors gracefully" is only a testable claim if what the
# error says is part of the test. The rules exist because their absence is not a
# loud failure - an unpredicted variable leaves a row of the prediction buffer
# uninitialized, and a wrong `inputs` list surfaces as a `DimensionMismatch` from
# somewhere inside the sample loop.
#
# `_ADL` rather than the `ADL` bound by `test_api.jl`, so this file does not depend
# on include order.
const _ADL = ApplicationDrivenLearning

_val_model = _ADL.Model()
@variable(_val_model, vf[1:2], _ADL.Forecast)
# anonymous forecast variables: no name of their own for a `Y` column to match
_anon_model = _ADL.Model()
_va = @variable(_anon_model, [1:2], _ADL.Forecast)
# containers that are not a plain `Vector`: one over non-integer axes, which
# supports neither `collect` nor iteration, and one of two dimensions
@variable(_val_model, vax[[:p, :q]], _ADL.Forecast)
@variable(_val_model, vgrid[1:2, 1:3], _ADL.Forecast)

# a fresh architecture per call: no two units may hold the same object
_vnn() = Flux.Dense(1 => 1) |> f64
_vnn2() = Flux.Dense(1 => 2) |> f64

@testset "unit rejects malformed inputs" begin
    @test_throws ArgumentError _ADL.ForecastModel(
        inputs = [0],
        architecture = _vnn(),
        outputs = [vf[1]],
    )
    # positions and names cannot be mixed inside one unit
    @test_throws ArgumentError _ADL.ForecastModel(
        inputs = [1, :temp],
        architecture = _vnn(),
        outputs = [vf[1]],
    )
    @test_throws ArgumentError _ADL.ForecastModel(
        inputs = Int[],
        architecture = _vnn(),
        outputs = [vf[1]],
    )
    @test_throws ArgumentError _ADL.ForecastModel(
        inputs = "temp",
        architecture = _vnn(),
        outputs = [vf[1]],
    )

    # the same column twice would feed two of the architecture's inputs from one
    # column, which is a typo far more often than a request
    err_dup = try
        _ADL.ForecastModel(
            inputs = [:a, :a],
            architecture = Flux.Dense(2 => 1) |> f64,
            outputs = [vf[1]],
        )
        nothing
    catch e
        e
    end
    @test err_dup isa ArgumentError
    @test occursin("same input column", err_dup.msg)
    @test_throws ArgumentError _ADL.ForecastModel(
        inputs = [1, 1],
        architecture = Flux.Dense(2 => 1) |> f64,
        outputs = [vf[1]],
    )

    # a scalar is accepted, by position or by name, for a one-input architecture
    @test _ADL.ForecastModel(
        inputs = 1,
        architecture = _vnn(),
        outputs = [vf[1]],
    ).inputs == [1]
    @test _ADL.ForecastModel(
        inputs = :temp,
        architecture = _vnn(),
        outputs = [vf[1]],
    ).inputs == [:temp]
end

@testset "unit rejects malformed outputs" begin
    @test_throws ArgumentError _ADL.ForecastModel(
        inputs = 1,
        architecture = _vnn(),
        outputs = [],
    )
    # a Symbol is not a forecast variable, whatever it is called
    @test_throws ArgumentError _ADL.ForecastModel(
        inputs = 1,
        architecture = _vnn(),
        outputs = [:demand],
    )
    # the column name in a pair has to be a Symbol
    @test_throws ArgumentError _ADL.ForecastModel(
        inputs = 1,
        architecture = _vnn(),
        outputs = [vf[1] => "col"],
    )
    # a variable twice in one unit, reported by name
    err = try
        _ADL.ForecastModel(
            inputs = 1,
            architecture = _vnn2(),
            outputs = [vf[1], vf[1]],
        )
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("f[1]", err.msg)

    # a single variable needs no vector around it
    @test _ADL.ForecastModel(
        inputs = 1,
        architecture = _vnn(),
        outputs = vf[1],
    ).outputs == [vf[1]]

    # named and unnamed columns may be mixed: the unnamed variable names its own
    mixed = _ADL.ForecastModel(
        inputs = 1,
        architecture = _vnn2(),
        outputs = [vf[1], vf[2] => :second],
    )
    @test mixed.output_columns == [Symbol("vf[1]"), :second]

    # unless it is anonymous, in which case there is nothing to fall back on
    @test_throws ArgumentError _ADL.ForecastModel(
        inputs = 1,
        architecture = _vnn2(),
        outputs = [_va[1], _va[2] => :second],
    )

    # an anonymous variable still shows up in a message, as such
    err_anon = try
        _ADL.ForecastModel(
            inputs = 1,
            architecture = _vnn2(),
            outputs = [_va[1], _va[1]],
        )
        nothing
    catch e
        e
    end
    @test err_anon isa ArgumentError
    @test occursin("<anonymous>", err_anon.msg)

    # A WHOLE CONTAINER where one variable belongs - one level of brackets too
    # many. It has its own message, because reporting only the type leaves the
    # reader to work out that `[d]` should have been `d`.
    err_wrapped = try
        _ADL.ForecastModel(inputs = 1, architecture = _vnn2(), outputs = [vf])
        nothing
    catch e
        e
    end
    @test err_wrapped isa ArgumentError
    @test occursin("holds 2 forecast variables", err_wrapped.msg)
    @test occursin("outputs = d", err_wrapped.msg)

    # the same mistake with a `Y` column attached: one column cannot belong to a
    # whole container, so the fix is one entry per variable rather than `d` alone
    for entry in (vf => :c, vf => 1)
        err_pair = try
            _ADL.ForecastModel(
                inputs = 1,
                architecture = _vnn2(),
                outputs = [entry],
            )
            nothing
        catch e
            e
        end
        @test err_pair isa ArgumentError
        @test occursin("holds 2 forecast variables", err_pair.msg)
        @test occursin("its own entry", err_pair.msg)
    end

    # a container of more than one dimension, wrapped or not: the order it would
    # be flattened in is not obvious, so it is never guessed at
    for outs in ([vgrid], vgrid)
        err_grid = try
            _ADL.ForecastModel(
                inputs = 1,
                architecture = Flux.Dense(1 => 6) |> f64,
                outputs = outs,
            )
            nothing
        catch e
            e
        end
        @test err_grid isa ArgumentError
        @test occursin("holds 6 forecast variables", err_grid.msg)
        @test occursin("vec(d)", err_grid.msg)
    end

    # ... while the container itself, unwrapped, is the correct way to write it
    @test length(
        _ADL.ForecastModel(
            inputs = 1,
            architecture = _vnn2(),
            outputs = vf,
        ).outputs,
    ) == 2

    # including one declared over non-integer axes, which neither `collect` nor
    # iteration can read - that used to surface as a `MethodError` from `Base`
    @test length(
        _ADL.ForecastModel(
            inputs = 1,
            architecture = _vnn2(),
            outputs = vax,
        ).outputs,
    ) == 2
end

@testset "unit checks its architecture against its wiring" begin
    # too few inputs for the architecture: the probe cannot even be applied
    err = try
        _ADL.ForecastModel(
            inputs = [1],
            architecture = Flux.Dense(2 => 1) |> f64,
            outputs = [vf[1]],
        )
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("inputs", err.msg)

    # right inputs, wrong number of variables
    err2 = try
        _ADL.ForecastModel(
            inputs = [1],
            architecture = _vnn(),
            outputs = [vf[1], vf[2]],
        )
        nothing
    catch e
        e
    end
    @test err2 isa ArgumentError
    @test occursin("produces 1 output(s) from 1 input(s)", err2.msg)

    # a parameterless callable is a legitimate architecture: the probe falls back
    # to `Float64` and the unit is built
    doubling = _ADL.ForecastModel(
        inputs = 1,
        architecture = x -> 2 .* x,
        outputs = [vf[1]],
    )
    @test doubling.architecture isa Function
end

@testset "forecast rejects malformed unit vectors" begin
    @test_throws ArgumentError _ADL.FullForecastModel(_ADL.ForecastModel[])
    # something that is not a unit, named by position
    err = try
        _ADL.FullForecastModel([_vnn()])
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("Entry 1", err.msg)

    # a `Vector{Any}` of units goes through the same checking fallback and works
    anyvec = Any[
        _ADL.ForecastModel(
            inputs = 1,
            architecture = _vnn(),
            outputs = [vf[1]],
        ),
        _ADL.ForecastModel(
            inputs = 1,
            architecture = _vnn(),
            outputs = [vf[2]],
        ),
    ]
    @test _ADL.FullForecastModel(anyvec).output_size == 2

    # a variable claimed by two units, with both units named
    err2 = try
        _ADL.FullForecastModel([
            _ADL.ForecastModel(
                inputs = 1,
                architecture = _vnn(),
                outputs = [vf[1]],
            ),
            _ADL.ForecastModel(
                inputs = 2,
                architecture = _vnn(),
                outputs = [vf[1]],
            ),
        ])
        nothing
    catch e
        e
    end
    @test err2 isa ArgumentError
    @test occursin("unit 1 and unit 2", err2.msg)

    # some units naming their `Y` columns and one predicting an anonymous variable
    @test_throws ArgumentError _ADL.FullForecastModel([
        _ADL.ForecastModel(
            inputs = 1,
            architecture = _vnn(),
            outputs = [_va[1] => :first],
        ),
        _ADL.ForecastModel(
            inputs = 1,
            architecture = _vnn(),
            outputs = [_va[2]],
        ),
    ])

    # naming one unit's column does not force the others to repeat what their own
    # variables already say
    named_one = _ADL.FullForecastModel([
        _ADL.ForecastModel(
            inputs = 1,
            architecture = _vnn(),
            outputs = [vf[1] => :first],
        ),
        _ADL.ForecastModel(
            inputs = 1,
            architecture = _vnn(),
            outputs = [vf[2]],
        ),
    ])
    @test named_one.output_columns == [:first, Symbol("vf[2]")]
end

@testset "forecast resolves the input schema or says why not" begin
    # one unit reading the whole row: the width comes from its architecture
    @test _ADL.FullForecastModel([
        _ADL.ForecastModel(
            architecture = Flux.Dense(2 => 2) |> f64,
            outputs = [vf[1], vf[2]],
        ),
    ]).input_size == 2

    # alongside another positional unit it is ambiguous: the widest column they
    # read bounds the row from below but does not determine it
    err = try
        _ADL.FullForecastModel([
            _ADL.ForecastModel(architecture = _vnn(), outputs = [vf[1]]),
            _ADL.ForecastModel(
                inputs = 2,
                architecture = _vnn(),
                outputs = [vf[2]],
            ),
        ])
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("not determined", err.msg)

    # two units both reading the whole row is the same ambiguity, and the fix is
    # to say which columns each reads
    @test_throws ArgumentError _ADL.FullForecastModel([
        _ADL.ForecastModel(
            architecture = Flux.Dense(2 => 1) |> f64,
            outputs = [vf[1]],
        ),
        _ADL.ForecastModel(
            architecture = Flux.Dense(2 => 1) |> f64,
            outputs = [vf[2]],
        ),
    ])
    listed = _ADL.FullForecastModel([
        _ADL.ForecastModel(
            inputs = [1, 2],
            architecture = Flux.Dense(2 => 1) |> f64,
            outputs = [vf[1]],
        ),
        _ADL.ForecastModel(
            inputs = [1, 2],
            architecture = Flux.Dense(2 => 1) |> f64,
            outputs = [vf[2]],
        ),
    ])
    @test listed.input_size == 2
    @test isnothing(listed.input_names)
    @test listed.units[1].inputs == [1, 2]
    @test listed.units[1].outputs == [vf[1]]

    # an architecture whose input size cannot be read has to list its inputs
    @test_throws ArgumentError _ADL.FullForecastModel([
        _ADL.ForecastModel(
            architecture = Flux.Scale(2) |> f64,
            outputs = [vf[1], vf[2]],
        ),
    ])

    # a unit selecting by name alongside one reading the whole row: the named unit
    # fixes the schema and the other reads all of it
    with_whole_row = _ADL.FullForecastModel([
        _ADL.ForecastModel(
            inputs = [:a],
            architecture = _vnn(),
            outputs = [vf[1]],
        ),
        _ADL.ForecastModel(architecture = _vnn(), outputs = [vf[2]]),
    ])
    @test with_whole_row.input_names == [:a]
    @test with_whole_row.units[2].inputs == [1]

    # units disagreeing on *how* they select leave no schema to resolve
    err_mixed = try
        _ADL.FullForecastModel([
            _ADL.ForecastModel(
                inputs = 1,
                architecture = _vnn(),
                outputs = [vf[1]],
            ),
            _ADL.ForecastModel(
                inputs = :b,
                architecture = _vnn(),
                outputs = [vf[2]],
            ),
        ])
        nothing
    catch e
        e
    end
    @test err_mixed isa ArgumentError
    @test occursin("by position and unit 2 by name", err_mixed.msg)
end

@testset "the container keeps its own invariants" begin
    # Reachable by building a `FullForecastModel` by hand, which is what `Functors`
    # does positionally on every optimiser step - so these checks belong on the
    # constructor and not only on the path from units.
    RU = _ADL.ResolvedUnit
    one_unit() = [RU([1], _vnn(), [vf[1]], [1])]

    @test_throws DimensionMismatch _ADL.FullForecastModel(
        one_unit(),
        [vf[1]],
        1,
        2,
    )
    # a unit whose outputs and rows are not read in step
    @test_throws DimensionMismatch _ADL.FullForecastModel(
        [RU([1], _vnn2(), [vf[1], vf[2]], [1])],
        [vf[1], vf[2]],
        1,
        2,
    )
    @test_throws ArgumentError _ADL.FullForecastModel(
        [RU([1], _vnn2(), [vf[1], vf[1]], [1, 2])],
        [vf[1], vf[1]],
        1,
        2,
    )

    # The rows have to partition the prediction. Neither half of this was
    # expressible while the container held a networks vector and a map beside it,
    # and both used to surface as a cost computed from uninitialized memory.
    err_gap = try
        _ADL.FullForecastModel(one_unit(), [vf[1], vf[2]], 1, 2)
        nothing
    catch e
        e
    end
    @test err_gap isa ArgumentError
    @test occursin("No unit writes prediction row 2", err_gap.msg)

    err_twice = try
        _ADL.FullForecastModel(
            [RU([1], _vnn(), [vf[1]], [1]), RU([1], _vnn(), [vf[2]], [1])],
            [vf[1], vf[2]],
            1,
            2,
        )
        nothing
    catch e
        e
    end
    @test err_twice isa ArgumentError
    @test occursin("2 units write prediction row 1", err_twice.msg)

    err_range = try
        _ADL.FullForecastModel([RU([1], _vnn(), [vf[1]], [3])], [vf[1]], 1, 1)
        nothing
    catch e
        e
    end
    @test err_range isa ArgumentError
    @test occursin("outside the 1 row(s)", err_range.msg)

    # the declared schema against the width it describes. Not reachable from the
    # units, which derive one from the other - only from here
    @test_throws ArgumentError _ADL.FullForecastModel(
        one_unit(),
        [vf[1]],
        1,
        1;
        input_names = [:a, :b],
    )
end

@testset "units are resolved once, and reordering re-resolves them" begin
    # `set_forecast_model` reorders a prediction to follow the variables declared
    # on the model, so a unit's `rows` are recomputed against that order rather
    # than looked up on every forward pass.
    m = _ADL.Model()
    @variable(m, y >= 0, _ADL.Policy)
    @variable(m, g[1:2], _ADL.Forecast)
    @objective(_ADL.Plan(m), Min, y.plan)
    @objective(_ADL.Assess(m), Min, y.assess)

    # the units predict g[2] before g[1], the model declares them the other way
    reversed = _ADL.FullForecastModel([
        _ADL.ForecastModel(inputs = 1, architecture = _vnn(), outputs = [g[2]]),
        _ADL.ForecastModel(inputs = 2, architecture = _vnn(), outputs = [g[1]]),
    ])
    @test reversed.output_variables == [g[2], g[1]]
    @test [u.rows for u in reversed.units] == [[1], [2]]

    nn = reversed.units[1].architecture
    _ADL.set_forecast_model(m, reversed)
    @test m.forecast.output_variables == [g[1], g[2]]
    # ... and every unit now writes the row its variable occupies there
    @test [u.rows for u in m.forecast.units] == [[2], [1]]
    @test [u.inputs for u in m.forecast.units] == [[1], [2]]
    # reordering carries the architectures over rather than copying them: a copy
    # would detach them from an optimiser state already set up on the forecast
    @test m.forecast.units[1].architecture === nn

    # assembling from units *does* copy, so the caller's object is left untouched
    mine = _vnn()
    assembled = _ADL.FullForecastModel([
        _ADL.ForecastModel(inputs = 1, architecture = mine, outputs = [vf[1]]),
    ])
    @test !(assembled.units[1].architecture === mine)
end

@testset "retired constructor signatures fail loudly" begin
    # The old `PredictiveModel` took a network, or a vector of networks and a map,
    # positionally. Those shapes are gone, so each retired arity says what to write
    # instead rather than reporting that no method matched.
    for call in (
        () -> _ADL.ForecastModel(Flux.Chain(_vnn())),
        () -> _ADL.ForecastModel(Flux.Chain(_vnn()); input_names = [:a]),
        () -> _ADL.ForecastModel([_vnn()], [Dict([1] => [vf[1]])]),
        () ->
            _ADL.ForecastModel([_vnn()], [Dict([1] => [vf[1]])], [vf[1]], 1, 1),
        () -> _ADL.ForecastModel(
            [_vnn()],
            [Dict([1] => [vf[1]])],
            [vf[1]],
            1,
            1,
            nothing,
            nothing,
            nothing,
        ),
    )
        err = try
            call()
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("built with keywords", err.msg)
    end
end

@testset "outputs address their Y column by name or by position" begin
    # The two spellings say the same thing - where in `Y` a variable's realized
    # values are - so a unit uses one or the other, and the container carries
    # whichever it was given.
    named = _ADL.ForecastModel(
        inputs = 1,
        architecture = _vnn(),
        outputs = [vf[1] => :a],
    )
    @test named.output_columns == [:a]

    positional = _ADL.ForecastModel(
        inputs = 1,
        architecture = _vnn(),
        outputs = [vf[1] => 2],
    )
    @test positional.output_columns == [2]

    bare =
        _ADL.ForecastModel(inputs = 1, architecture = _vnn(), outputs = [vf[1]])
    @test isnothing(bare.output_columns)

    # `Int32` and friends are accepted and narrowed
    @test _ADL.ForecastModel(
        inputs = 1,
        architecture = _vnn(),
        outputs = [vf[1] => Int32(3)],
    ).output_columns == [3]

    # a `Bool` is an `Integer`, and would otherwise quietly mean column 1
    err_bool = try
        _ADL.ForecastModel(
            inputs = 1,
            architecture = _vnn(),
            outputs = [vf[1] => true],
        )
        nothing
    catch e
        e
    end
    @test err_bool isa ArgumentError
    @test occursin("gives the `Y` column as true", err_bool.msg)

    # a `String` stays rejected, and the message names both accepted spellings
    err_string = try
        _ADL.ForecastModel(
            inputs = 1,
            architecture = _vnn(),
            outputs = [vf[1] => "a"],
        )
        nothing
    catch e
        e
    end
    @test err_string isa ArgumentError
    @test occursin("variable => :column", err_string.msg)
    @test occursin("variable => 3", err_string.msg)

    # positions start at 1
    err_zero = try
        _ADL.ForecastModel(
            inputs = 1,
            architecture = _vnn(),
            outputs = [vf[1] => 0],
        )
        nothing
    catch e
        e
    end
    @test err_zero isa ArgumentError
    @test occursin("positions start at 1", err_zero.msg)

    # ... and no two variables read one column
    err_dup = try
        _ADL.ForecastModel(
            inputs = 1,
            architecture = _vnn2(),
            outputs = [vf[1] => 2, vf[2] => 2],
        )
        nothing
    catch e
        e
    end
    @test err_dup isa ArgumentError
    @test occursin("same `Y` column position more than once", err_dup.msg)
end

@testset "outputs cannot mix the two spellings" begin
    # inside one unit
    err_unit = try
        _ADL.ForecastModel(
            inputs = 1,
            architecture = _vnn2(),
            outputs = [vf[1] => :a, vf[2] => 2],
        )
        nothing
    catch e
        e
    end
    @test err_unit isa ArgumentError
    @test occursin("two spellings of the same thing", err_unit.msg)

    # a bare entry alongside a position: unlike a name, a position has no
    # fallback, and the message says which variable it could not invent one for
    err_bare = try
        _ADL.ForecastModel(
            inputs = 1,
            architecture = _vnn2(),
            outputs = [vf[1] => 1, vf[2]],
        )
        nothing
    catch e
        e
    end
    @test err_bare isa ArgumentError
    @test occursin("cannot be filled in from the variable", err_bare.msg)
    @test occursin("vf[2]", err_bare.msg)

    # across units
    err_units = try
        _ADL.FullForecastModel([
            _ADL.ForecastModel(
                inputs = 1,
                architecture = _vnn(),
                outputs = [vf[1] => :a],
            ),
            _ADL.ForecastModel(
                inputs = 2,
                architecture = _vnn(),
                outputs = [vf[2] => 2],
            ),
        ])
        nothing
    catch e
        e
    end
    @test err_units isa ArgumentError
    @test occursin("by name and unit 2 by position", err_units.msg)

    # a unit giving positions alongside one giving nothing at all
    err_partial = try
        _ADL.FullForecastModel([
            _ADL.ForecastModel(
                inputs = 1,
                architecture = _vnn(),
                outputs = [vf[1] => 1],
            ),
            _ADL.ForecastModel(
                inputs = 2,
                architecture = _vnn(),
                outputs = [vf[2]],
            ),
        ])
        nothing
    catch e
        e
    end
    @test err_partial isa ArgumentError
    @test occursin("but unit 2 gives none", err_partial.msg)

    # the name spelling keeps its fallback: unit 2 contributes its own names
    mixed_ok = _ADL.FullForecastModel([
        _ADL.ForecastModel(
            inputs = 1,
            architecture = _vnn(),
            outputs = [vf[1] => :a],
        ),
        _ADL.ForecastModel(
            inputs = 2,
            architecture = _vnn(),
            outputs = [vf[2]],
        ),
    ])
    @test mixed_ok.output_columns == [:a, Symbol("vf[2]")]
end

@testset "the container checks declared Y column positions" begin
    RU = _ADL.ResolvedUnit
    two() = [RU([1], _vnn2(), [vf[1], vf[2]], [1, 2])]

    @test_throws ArgumentError _ADL.FullForecastModel(
        two(),
        [vf[1], vf[2]],
        1,
        2;
        output_columns = [1],
    )
    @test_throws ArgumentError _ADL.FullForecastModel(
        two(),
        [vf[1], vf[2]],
        1,
        2;
        output_columns = [2, 2],
    )
    err_zero = try
        _ADL.FullForecastModel(two(), [vf[1], vf[2]], 1, 2; output_columns = [0, 1])
        nothing
    catch e
        e
    end
    @test err_zero isa ArgumentError
    @test occursin("positions start at 1", err_zero.msg)

    # the width a matrix `Y` has to have is the widest position, not the number
    # of outputs - the columns in between may be present and unread
    gapped = _ADL.FullForecastModel(
        two(),
        [vf[1], vf[2]],
        1,
        2;
        output_columns = [1, 4],
    )
    @test _ADL._output_width(gapped) == 4
    @test _ADL._output_width(
        _ADL.FullForecastModel(two(), [vf[1], vf[2]], 1, 2),
    ) == 2
end

@testset "the removed name is gone rather than stubbed" begin
    # `PredictiveModel` is not exported and no method of it survives, so old code
    # gets an `UndefVarError` naming the binding. A throwing stub would report the
    # two replacements by name, which reads better - but it also keeps a removed
    # type in the API surface and in `@doc`, and the message would outlive anyone
    # who needed it.
    @test !isdefined(ApplicationDrivenLearning, :PredictiveModel)
    @test :PredictiveModel ∉ names(ApplicationDrivenLearning)
end
