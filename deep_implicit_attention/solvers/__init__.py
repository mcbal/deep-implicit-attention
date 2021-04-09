from dataclasses import asdict, dataclass


@dataclass()
class SolverOptions:
    """Class for keeping track of an item in inventory."""

    tol: float = 1e-5
    max_iter: int = 50

    def to_dict(self):
        return asdict(self)


@dataclass()
class AndersonSolverOptions(SolverOptions):
    """Class for keeping track of an item in inventory."""

    m: int = 5
    lam: float = 1e-4
    beta: float = 1.0
