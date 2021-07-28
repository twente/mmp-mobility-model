# README

*Documentation under construction.*

## Mobility Simulations
`MOB_SIM(config, trials = 1000, path = SAVE_PATH, overwrite = False)`

- `config`: dictionary containing mobility simulation parameters.
- `trials`: number of trials (integer)
- `path`: path to save the simulation results to
- `overwrite`: overwrite simulation results if they already exist

## Calculate Metrics
`calc_metrics(path, selection = [], overwrite = False)`

- `path`: path of simulation results
- `selection`: list of metrics to calculate (if not specified, calculates everything)
- `overwrite`: force all metrics to be recalculated

Metrics will be saved to the same folder as `_metrics.pkl`.

## Build Combinatorial MMP Model
`build_UNAGG(path, out_path = None, overwrite = False)`

- `path`: path of mobility simulation results that the model will be based on
- `out_path`: output path of the MMP model
- `overwrite`: overwrite model if it already exists

## Build Reduced MMP Model

`build_AGG(path, out_path = None, overwrite = False)`

- `path`: path of mobility simulation results that the model will be based on
- `out_path`: output path of the MMP model
- `overwrite`: overwrite model if it already exists

## MMP Simulation
`MMP_SIM(path, trials = 1000, T = None, overwrite = False)`

- `path`: path of the MMP model
- `trials`: number of trials to perform, defaults to the same as the mobility process
- `T`: number of time steps per trials, defaults to same as mobility process
- `overwrite`: overwrite results if they already exist

Results are saved to the same path. Link metrics are autocalculated during simulation.