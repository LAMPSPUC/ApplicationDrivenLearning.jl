include("matpower_parser.jl")
include("builder_system_data.jl")
include("builder_time_series.jl")
include("optimization_models.jl")
include("ols.jl")


# load case from file
if isfile(case_path)
    case = read_matlab_file(case_path)
else
    error("File not found")
end

# convert case to internal format
# pd: power data
# ts: time series
pd, ts = convert_matpower(
    case,
    N_LAGS,
    N_DEMANDS,
    N_ZONES,
    TEST_SIZE,
    DEFF_COEF,
    SPILL_COEF,
    COEF_VARIATION,
    demand_path,
    TRAIN_SIZE,  # train_dataset_size_current
    1  # sample_current
)
pd.F *= (0.95 ^ TRASMISSION_REDUCTION)

# 'ts' d_train and d_test comes from demand file values multiplied by expected values

# least squares exogenous version
# get least squares coefficients: size=(n_demand_lags + 1, n_demand)
θd_ls = multi_ls(ts.d_train, pd.n_demand_lags, pd.n_demand)
# ls forecast error
forecast_error = ls_noise_var(ts.d_train, pd.n_demand_lags, pd.n_demand)

# base value for reserve is 1.96 times the forecast error
reserve_base_value = 1.96 * forecast_error

# set reserve (up and down) starting value for all zones
θru_st = OffsetArray(
    fill(reserve_base_value, pd.n_zones, 1 + pd.n_reserve_lags),
    1:pd.n_zones,
    0:pd.n_reserve_lags,
)
θrd_st = deepcopy(θru_st)
for r = 1:pd.n_zones, p = 1:pd.n_reserve_lags
    θru_st[r, p] = 0.0
    θrd_st[r, p] = 0.0
end
# distribute reserve among zones
if pd.n_zones > 1
    frac = zone_frac(pd)
    for r = 1:pd.n_zones
        θru_st[r, 0] = reserve_base_value * frac[r]
        θrd_st[r, 0] = reserve_base_value * frac[r]
    end
end
