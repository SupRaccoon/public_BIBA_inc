import numpy as np


class Angle:
    def __init__(self, identifier, low_border=0, high_border=np.pi,
                 min_tolerance=None):
        self._identifier = identifier
        self._randomizer = np.random.RandomState(self.identifier)
        self._index = 0
        self._pre_index = 0
        self._abs_low_border = low_border
        self._low_border = low_border
        self._abs_high_border = high_border
        self._high_border = high_border
        self._min_tolerance = min_tolerance or np.pi / 180
        self._values_pull = np.arange(self._low_border,
                                      self._high_border,
                                      self._min_tolerance)
        self.gen_angle()
        self.make_angle()

    @property
    def index(self):
        return self._index

    @property
    def pre_index(self):
        return self._pre_index

    @property
    def value(self):
        return self._values_pull[self._index]

    @property
    def prev_value(self):
        return self._values_pull[self._pre_index]

    @property
    def values_pull_size(self):
        return self._values_pull.size

    @property
    def identifier(self):
        return self._identifier

    def gen_angle(self, pool=None):
        _pool = pool or self.values_pull_size
        self._pre_index = self._randomizer.randint(_pool)
        return self._pre_index

    def make_angle(self):
        self._index = self._pre_index

    def set_index(self, value):
        if value > len(self._values_pull) - 1:
            value = value - len(self._values_pull)
        elif value < -len(self._values_pull) + 1:
            value = value + len(self._values_pull)

        self._index = value

    def set_pre_index(self, value):
        self._pre_index = value
