
## ENGA — Enhanced Networked Genetic Algorithm

A Python implementation of networked genetic algorithms that use genotypic and phenotypic similarity to form scale-free inter-chromosomal links, helping balance exploration and exploitation during evolution.

![Animated Network](logs/metrics/animated_networks/animated_network_25_22_11.gif)

*Animated evolution of node rankings across generations. Node colors indicate fitness ranking: green (top 2), teal (next 10), blue-green (next 15), and dark blue (others). Labels show node scores.*

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

- `--save-metrics` — optional flag. When present, the benchmarks will save metrics (fitness, diversity, selection pressure, etc.) into `logs/metrics/`. This flag is off by default.

- `--generations N` — set the number of generations (overrides per-benchmark default). Default (if omitted): 2000.
- `--population-size N` — set the population size for all benchmarks. Default (if omitted): 125 (124 for CRGA in original config).
- `--number-elites N` — number of elites to keep. Default (per-benchmark): 0 for most algorithms, 10 for the plain GA.
- `--crossover-prob P` — crossover probability (0.0–1.0). Default: 0.95.
- `--mutation-prob P` — mutation probability (0.0–1.0). Default: 0.10.
- `--nruns N` — number of independent runs to perform. Default: 1.
- `--no-verbose` — disable verbose output. Default: verbose on.

Examples:

```bash
# run all benchmarks
python src/main.py --run all

# run only the enhanced networked GA
python src/main.py --run enhanced

# run multiple specified benchmarks (enhanced and plain ga)
python src/main.py --run enhanced ga
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

## Metrics & logs

This project can save a rich set of metrics and visualizations from each benchmark run. Metrics are written under `logs/metrics/` when enabled (see below). Keep in mind metrics (especially animated GIFs) can consume disk and I/O during long runs — enable them only when you need the outputs.

### Key Components

- `logs/metrics/fitness/` — PNG plots of average and best fitness over generations.
- `logs/metrics/phenotype_diversity/` — PNG plots showing phenotype diversity over time.
- `logs/metrics/genotype_diversity/` — PNG plots for genotype-space diversity.
- `logs/metrics/selection_pressure/` — PNG plots tracking selection pressure each generation.
- `logs/metrics/degree/`, `logs/metrics/degree_exponent/`, `logs/metrics/clustering_coefficient/`, `logs/metrics/path_length/` — network-related PNG visualizations.
- `logs/metrics/number_of_nodes_to_remove/` — PNG plots for node-removal statistics.
- `logs/metrics/animated_networks/` — optional animated network visualizations saved as GIF files.

### How to enable saving

- From the command line: pass `--save-metrics` to `src/main.py` to enable metric saving for benchmark runs (this toggles the `save_metrics` flag when `Benchmark` objects are created).
- Programmatically: set `save_metrics=True` when constructing a `Benchmark(...)` or call `DrawManager.draw_animated_network(history, save=True)` to save an animation explicitly.

### Network Animation

As highlighted in the animation above, in the network architecture, high‑fitness individuals migrate from the periphery to the center as evolution creates hubs — node color shows fitness and centrality reflects accumulated connections. This highlights selection pressure, hub formation, and convergence.

## Typical usage example

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
