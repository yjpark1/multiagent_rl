import numpy as np

class LinearAnnealedPolicy:
    def __init__(self, value_max=1., value_min=0., nb_max_random_step=1000):
        self.value_max = value_max
        self.value_min = value_min
        self.nb_max_random_step = nb_max_random_step

    def get_current_value(self, step):
        # Linear annealed: f(x) = ax + b.
        a = -float(self.value_max - self.value_min) / float(self.nb_max_random_step)
        b = float(self.value_max)
        value = max(self.value_min, a * float(step) + b)
        return value