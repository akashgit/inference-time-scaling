# Particle Filter and MCMC Implementations

A collection of Python implementations for particle filtering and MCMC methods, with tools for comparing their performance.

## Files

- `particle_filter.py` - Standard particle filter implementation
- `simple_particle_filter.py` - Basic particle filter for learning purposes
- `simple_particle_gibbs.py` - Particle Gibbs sampler
- `parallel_tempering.py` - Parallel tempering MCMC
- `compare_simple_filters.py` - Tools to compare filter performance
- `compare_pf_pg.py` - Compare particle filter vs particle Gibbs
- `particle_comparison.py` - General particle method comparisons
- `mcmc_comparison.py` - MCMC method comparisons

## Setup

1. Install dependencies:
pip install numpy scipy matplotlib pandas

2. Run comparisons:

python compare_simple_filters.py # Compare particle filter variants
python compare_pf_pg.py # Compare PF vs Particle Gibbs
python particle_comparison.py # General comparison suite

## Requirements

- Python 3.6+
- NumPy
- SciPy
- Matplotlib
- Pandas

## License

MIT License

