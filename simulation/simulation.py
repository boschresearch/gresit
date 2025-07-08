"""Experiment set up and run."""

import argparse
import json

from gresit.independence_tests import HSIC
from gresit.params import (
    DataParams,
    ExperimentParams,
    GraNDAGParams,
    RandomRegParams,
    ResitParams,
    PCParams,
)
from gresit.simulation_utils import BenchMarker
from gresit.synthetic_data import (
    FCNN,
    GaussianProcesses,
    GenERData,
)
from gresit.torch_models import Multioutcome_MLP


def _get_p(num_nodes: int, factor: int = 1) -> float:
    return 2 * factor / (num_nodes - 1)


def _get_experiment_name(
    n_samples: int = 20000,
    n_nodes: int = 10,
    group_size: int = 5,
    snr: int = 2,
    equation_type: str = "GP",
) -> str:
    return f"n_{n_nodes}_gs_{group_size}_snr_{snr}_equ_{equation_type}_{n_samples}"


def _make_params(
    n_samples: int = 20000,
    n_nodes: int = 10,
    group_size: int = 5,
    snr: int = 2,
    equation_type: str = "GP",
) -> ExperimentParams:
    equation_cls: type[GaussianProcesses] | type[FCNN]
    if equation_type == "GP":
        equation_cls = GaussianProcesses
    else:
        equation_cls = FCNN

    return ExperimentParams(
        algos=[
            PCParams(alpha=0.1),
            PCParams(alpha=0.2),
            ResitParams(
                regressor=Multioutcome_MLP,
                kwargs={
                    "n_epochs": 500,
                    "learning_rate": 0.01,
                    "loss": "mse",
                    "dropout_proba": 0.0,
                    "batch_size": 500,
                    "val_size": 0.3,
                },
                test=HSIC,
                pruning_method="murgs",
                test_size=0.2,
            ),
            ResitParams(
                regressor=Multioutcome_MLP,
                kwargs={
                    "n_epochs": 500,
                    "learning_rate": 0.01,
                    "loss": "mse",
                    "dropout_proba": 0.0,
                    "batch_size": 512,
                    "val_size": 0.3,
                },
                test=HSIC,
                pruning_method="independence",
                test_size=0.2,
            ),
            GraNDAGParams(n_iterations=100000, with_group_constraint=True),
            RandomRegParams(),
        ],
        data=DataParams(
            generator=GenERData,
            number_of_nodes=n_nodes,
            equation_cls=equation_cls,
            group_size=group_size,
            edge_density=_get_p(num_nodes=n_nodes),
            noise_distribution="lognormal",
            snr=snr,
        ),
        number_of_samples=n_samples,
    )


def run_experiment(
    n_samples: int = 20000,
    n_nodes: int = 10,
    group_size: int = 5,
    snr: int = 2,
    equation_type: str = "GP",
    n_runs: int = 5,
) -> None:
    """Run Experiment.

    Args:
        n_samples (int, optional): _description_. Defaults to 20000.
        n_nodes (int, optional): _description_. Defaults to 10.
        group_size (int, optional): _description_. Defaults to 5.
        snr (int, optional): _description_. Defaults to 2.
        equation_type (str, optional): _description_. Defaults to "GP".
        n_runs (int, optional): _description_. Defaults to 5.
    """
    params = _make_params(
        n_samples=n_samples,
        n_nodes=n_nodes,
        group_size=group_size,
        snr=snr,
        equation_type=equation_type,
    )

    experiment_name = _get_experiment_name(
        n_samples=n_samples,
        n_nodes=n_nodes,
        group_size=group_size,
        snr=snr,
        equation_type=equation_type,
    )

    bm = BenchMarker()
    results = bm.run_benchmark(
        params=params,
        num_runs=n_runs,
        metrics=[
            "precision",
            "recall",
            "f1",
            "shd",
            "sid",
            "ancestor_aid",
            "ancester_ordering_aid",
        ],
        cpdag_strategy="best_dag",
    )

    with open(f"results_{experiment_name}.json", "w", encoding="utf-8") as outfile:
        json.dump(results, outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", default=2000, type=int)
    parser.add_argument("--n_nodes", default=10, type=int)
    parser.add_argument("--group_size", default=5, type=int)
    parser.add_argument("--equation_type", default="GP", type=str)
    parser.add_argument("--n_runs", default=5, type=int)

    config = parser.parse_args().__dict__

    run_experiment(**config)
