import random
import numpy as np


class Measurements:
    """Keep exact averages and a fixed-size sample for percentiles."""
    def __init__(self, capacity=4096):
        self.count = 0
        self.sums = {}
        self.samples = []
        self.capacity = capacity
        self.rng = random.Random(0)

    def add(self, values):
        self.count += 1
        for key, value in values.items():
            self.sums[key] = self.sums.get(key, 0.0) + value
        if len(self.samples) < self.capacity:
            self.samples.append(values)
        else:
            index = self.rng.randrange(self.count)
            if index < self.capacity:
                self.samples[index] = values

    def summary(self):
        return {key: {"mean": total / self.count,
                      "p50": float(np.percentile([s[key] for s in self.samples], 50)),
                      "p95": float(np.percentile([s[key] for s in self.samples], 95))}
                for key, total in self.sums.items()} if self.count else {}
