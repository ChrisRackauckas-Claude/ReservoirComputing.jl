using Test
using Random
using ReservoirComputing
using LuxCore: setup
using LinearSolve: SVDFactorization

# Delete with src/deprecated.jl at v1.0.

const _DEPWARN_ERROR = Base.JLOptions().depwarn == Int8(2)
# Julia ≥ 1.12 does not depwarn when a `@deprecate_binding` name imported by `using` is used.
const _BINDING_DEPWARN_THROWS = _DEPWARN_ERROR && VERSION < v"1.12"

# Under `--depwarn=error` `@test_deprecated` only checks for an `ErrorException` and returns
# a `Test.Pass`, so check the depwarn message there and return `nothing`.
function test_deprecated_call(f, msg::Regex)
    _DEPWARN_ERROR || return @test_deprecated msg f()
    @test_throws msg f()
    return nothing
end

@testset "StandardRidge is deprecated alias of RidgeRegression" begin
    @test Base.isdeprecated(ReservoirComputing, :StandardRidge)
    if _BINDING_DEPWARN_THROWS
        @test_throws "use of deprecated variable" StandardRidge
    else
        @test StandardRidge === RidgeRegression
        @test StandardRidge(1.0e-3) isa RidgeRegression
        @test StandardRidge(1.0e-3).reg == 1.0e-3
        @test StandardRidge().reg == 0.0
        @test StandardRidge(Float32, 1.0f-2).reg isa Float32
    end
end

@testset "package precompiles with --depwarn=error" begin
    code = "Base.compilecache(Base.identify_package(\"ReservoirComputing\"))"
    cmd = `$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) --depwarn=error -e $code`
    @test success(pipeline(cmd; stdout = devnull, stderr = devnull))
end

@testset "train! is deprecated and matches train" begin
    rng = MersenneTwister(31)
    in_dims, res_dims, out_dims = 2, 12, 1
    n_steps = 20
    train_data = randn(rng, Float32, in_dims, n_steps)
    target_data = randn(rng, Float32, out_dims, n_steps)

    model = ESN(in_dims, res_dims, out_dims)
    ps, st = setup(rng, model)
    ridge = RidgeRegression(2.0e-3)

    ps_kw, _ = train(
        model, train_data, target_data, ps, st; objective = ridge
    )
    trained = test_deprecated_call(r"^`train!` is deprecated") do
        train!(model, train_data, target_data, ps, st, ridge)
    end
    _DEPWARN_ERROR || @test first(trained).readout.weight ≈ ps_kw.readout.weight
end

@testset "train(objective, states, target_data) is deprecated and matches fitting" begin
    rng = MersenneTwister(53)
    states = randn(rng, Float64, 5, 30)
    targets = randn(rng, Float64, 2, 30)
    ridge = RidgeRegression(1.0e-3)

    weights_new = ReservoirComputing.__fit_readout(
        ridge, states, targets; solver = ReservoirComputing.QRFactorization()
    )
    result = test_deprecated_call(r"^`train\(objective, states, target_data;") do
        train(ridge, states, targets; solver = ReservoirComputing.QRFactorization())
    end
    _DEPWARN_ERROR || @test result ≈ weights_new
end

@testset "toepliz_init is deprecated alias of toeplitz_init" begin
    @test Base.isdeprecated(ReservoirComputing, :toepliz_init)
    if _BINDING_DEPWARN_THROWS
        @test_throws "use of deprecated variable" toepliz_init
    else
        @test toepliz_init === toeplitz_init
        @test toepliz_init(MersenneTwister(7), 5, 5) == toeplitz_init(MersenneTwister(7), 5, 5)
    end
end

@testset "train! still accepts solver kwarg" begin
    rng = MersenneTwister(37)
    in_dims, res_dims, out_dims = 3, 10, 2
    n_steps = 28
    train_data = randn(rng, Float32, in_dims, n_steps)
    target_data = randn(rng, Float32, out_dims, n_steps)

    model = ESN(in_dims, res_dims, out_dims)
    ps, st = setup(rng, model)

    trained = test_deprecated_call(r"^`train!` is deprecated") do
        train!(
            model, train_data, target_data, ps, st, RidgeRegression(1.0e-3);
            solver = SVDFactorization(),
        )
    end
    if !_DEPWARN_ERROR
        ps_trained = first(trained)
        @test size(ps_trained.readout.weight) == (out_dims, res_dims)
        @test all(isfinite, ps_trained.readout.weight)
    end
end
