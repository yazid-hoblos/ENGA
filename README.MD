
## ENGA — Enhanced Networked Genetic Algorithm

A Python implementation of networked genetic algorithms that use genotypic and phenotypic similarity to form scale-free inter-chromosomal links, helping balance exploration and exploitation during evolution.

This repository contains several GA variants (including ENGA, NGA, GA, CRGA, DCGA) and a small benchmarking harness to compare them on common test functions.

## Quick start

1. Clone the repository:

```bash
git clone https://github.com/yazid-hoblos/ENGA.git
cd ENGA
```

2. Install requirements:

```bash
pip install -r requirements.txt
```

3. Run the main benchmark script:

```bash
python src/main.py --run all
```

Command-line options

- `--run` — select which benchmark(s) to execute. Accepts one or more of the following values: `enhanced`, `networked`, `ga`, `crga`, `dcga`, or `all`.

Examples:

```bash
# run all benchmarks
python src/main.py --run all

# run only the enhanced networked GA
python src/main.py --run enhanced

# run multiple specified benchmarks (enhanced and plain ga)
python src/main.py --run enhanced ga
```

## Typical usage example (from `src/main.py`)

```py
from benchmark import Benchmark
from benchmark_functions import get_all_functions
from enhanced_networked_genetic_algorithm import EnhancedNetworkGeneticAlgorithm

benchmark_functions = get_all_functions()
enhanced_nga_benchmark = Benchmark(
    genetic_algorithm=EnhancedNetworkGeneticAlgorithm,
    number_of_runs=1,
    benchmark_functions=benchmark_functions,
    number_of_generations=2000,
    population_size=125,
    number_elites=0,
    probability_crossover=0.95,
    probability_mutation=0.1,
    verbose=True,
    save_metrics=True,
)

enhanced_nga_benchmark.run()
```

## Files and project layout

Key files (located in `src/`):

- `main.py` — example entry point and benchmark orchestration.
- `genetic_algorithm.py` — base GA class and shared utilities.
- `networked_genetic_algorithm.py`, `enhanced_networked_genetic_algorithm.py`, `networked_assignment.py` — networked GA variants and helpers.
- `benchmark.py`, `benchmark_functions.py` — benchmarking harness and test functions.
- `crossover.py`, `mutation.py`, `selection.py`, `networked_selection.py` — GA operators.
- `utils/` — small utility modules (`math.py`, `drawable.py`).

Output and logs:

- `logs/metrics/` — saved metrics (fitness, diversity, selection pressure, ...)
- `logs/results/` — benchmark run outputs and archived results

Images used in the README / examples are in the top-level `Images/` folder.

## Requirements

See `requirements.txt` for Python package dependencies. The project expects Python 3.8+ (recommend 3.10+).

## Contributing

Contributions are welcome. If you'd like to contribute:

1. Fork the repo and create a feature branch.
2. Add tests or examples for new behaviour when applicable.
3. Open a pull request describing your change.

If you plan to work on an issue, comment on it to avoid duplicated effort.

## Authors

- [@Murf-y](https://github.com/Murf-y)
- [@yazid-hoblos](https://github.com/yazid-hoblos)

## Acknowledgements

- [Numpy](https://numpy.org/)
- [Networkx](https://networkx.org/)
- [Pyvis](https://github.com/WestHealth/pyvis)
