# SpinChain Public API

These imports are the stable public interface. Breaking changes require a major version bump.

## Solvers

```python
from spinchain.solvers.base import BaseSolver
from spinchain.solvers.simulated_annealing import SimulatedAnnealingSolver
```

### `BaseSolver` (ABC)

Abstract base class for all QUBO solver backends.

- `solve(bqm: dimod.BinaryQuadraticModel) -> dimod.SampleSet` — Solve the BQM and return solutions ranked by energy.
- `name: str` (abstract property) — Solver identifier for logging/benchmarking.

### `SimulatedAnnealingSolver`

Simulated annealing solver via `dwave-neal`.

- `__init__(num_reads: int = 100, num_sweeps: int = 1000, beta_range: tuple[float, float] | None = None)`
- `solve(bqm: dimod.BinaryQuadraticModel) -> dimod.SampleSet`
- `name -> "simulated_annealing"` (property)

## Formulation

```python
from spinchain.formulation.qubo_builder import QUBOBuilder
```

### `QUBOBuilder`

Builds a `dimod.BinaryQuadraticModel` from fragment coefficients.

- `__init__(penalty_strength: float = 5.0)`
- `build(linear_weights: np.ndarray, quadratic_weights: np.ndarray, target_fragments: int | None = None) -> dimod.BinaryQuadraticModel` — Build BQM from linear (1D) and quadratic (2D) weight arrays, with optional cardinality constraint.
- `bqm_to_qubo(bqm: dimod.BinaryQuadraticModel) -> tuple[dict, float]` — Convert BQM to raw QUBO dict format for custom solvers.

## Internal (not stable)

Everything else is internal and may change without notice:

- `CoefficientBuilder` — coefficient design from fragment statistics
- `FragmentExtractor`, `StabilityRanker` — pipeline internals
- `server.py` — MCP server entry point
- `tracing.py` — trace logging
- `analyze.py` — trace analysis utilities

## Consumers

- [Anneal](https://github.com/ameyakhot/anneal) — imports `QUBOBuilder` and `SimulatedAnnealingSolver`
