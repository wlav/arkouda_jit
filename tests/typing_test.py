import numpy as np
import numba as nb

from common import ArkoudaJITTest
from context import arkouda as ak


class TypingTests(ArkoudaJITTest):
    """Explicit tests of typing corner cases"""

    def test_constructors(self):
        """Verify dtype typing"""

        def calc_f64():       # float64
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
