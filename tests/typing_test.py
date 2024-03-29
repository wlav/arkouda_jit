import numpy as np
import numba as nb

from common import ArkoudaJITTest
from context import arkouda as ak


class TypingTests(ArkoudaJITTest):
    """Explicit tests of typing corner cases"""

    def test_constructors(self):
        """Verify dtype typing"""

        def calc_f64():      # float64
            A = ak.zeros(5)
            A[3] = 42.
            c = A[3]
            return c + 1.

        def calc_i64():
            A = ak.zeros(5, dtype=ak.int64)
            A[3] = 42
            c = A[3]
            return c + 1

        assert self.verify(locals())

        for f, rtype in ((calc_f64, np.float64),
                         (calc_i64, np.int64)):
            c = f()
            assert type(c) == rtype

    def test_numeric(self):
        """Verify return types of numering operations"""

        def calc1():         # float64
            s = 0
            s += ak.sum(ak.arange(5))
            return s

        def calc2():
            s = 0
            s += ak.sum(ak.arange(5, dtype=ak.int64))
            return s

        assert self.verify(locals())

        s = calc1()
        assert type(s) == np.int64
