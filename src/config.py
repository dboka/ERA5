from dataclasses import dataclass

@dataclass
class Paths:
    grid_csv: str
    stations_csv: str
    out_dir: str = "out"

@dataclass
class RunConfig:
    year: int = 2013
    month: int = 1

    # OI params (start values)
    eps: float = 1.0          # mm, stabilizer for log
    L: float = 50000.0        # m, horizontal decorrelation length
    pobs_d: float = 0.3       # (dimensionless) obs error ratio in d-space
    max_points: int = 30
    cv_radius_m: float = 2000.0

    omp_threads: int = 4

    # Safety
    max_cells: int = 2_000_000  # prevent accidental huge allocations
