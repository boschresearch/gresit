# Simulations

All code necessary to reproduce the simulation results from our paper

```
@inproceedings{
    goebler2025gresit,
    title={Nonlinear Causal Discovery for Grouped Data},
    author={Goebler, K., Windisch, T., Drton, M.},
    booktitle={The 41st Conference on Uncertainty in Artificial Intelligence},
    year={2025},
}
```

is contained in the folder [`simulation`](https://github.com/bosch-research/gresit/tree/main/simulation).

Essentially, executing `./run_simulations.sh` executes the benchmarks for different numbers of nodes,
groups, and samples, sequentially. In our experiments, we use `slurm` to massively parallelize the individual runs.
