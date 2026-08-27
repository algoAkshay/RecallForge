"""Small structured metric types shared by the offline evaluation CLI and tests."""
from dataclasses import asdict, dataclass

FALSE_MEMORY_COST = 2
FALSE_WEB_COST = 1


@dataclass(frozen=True)
class RoutingMetrics:
    total: int; correct: int; memory_decisions: int; web_decisions: int; false_memory: int; false_web: int
    @property
    def accuracy(self): return self.correct / self.total if self.total else 0.0
    @property
    def weighted_error(self): return FALSE_MEMORY_COST * self.false_memory + FALSE_WEB_COST * self.false_web
    def as_dict(self): return {**asdict(self), "accuracy": self.accuracy, "weighted_error": self.weighted_error}


@dataclass(frozen=True)
class ConfusionMetrics:
    tp: int; tn: int; fp: int; fn: int; neutral: int = 0
    @property
    def total(self): return self.tp + self.tn + self.fp + self.fn
    @property
    def accuracy(self): return (self.tp + self.tn) / self.total if self.total else 0.0
    @property
    def precision(self): return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0
    @property
    def recall(self): return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0
    def as_dict(self): return {**asdict(self), "total": self.total, "accuracy": self.accuracy, "precision": self.precision, "recall": self.recall}
