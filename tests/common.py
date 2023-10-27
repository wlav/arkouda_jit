from base_test import ArkoudaTest

import arkouda as ak
import arkjit
import inspect
import numpy as np


class ArkoudaJITTest(ArkoudaTest):
    """Base class for JIT tests"""

    def compare(self, forg, *args, **kwds):
        res0 = forg(*args, **kwds)

        fopt = arkjit.optimize()(forg)
        res1 = fopt(*args, **kwds)

        if isinstance(res0, (ak.pdarray, ak.Strings)):
            assert sum(res0.to_ndarray() == res1.to_ndarray()) == len(res0)
        else:
            if type(res0) == np.float64:
                assert round(res0-res1, 14) == 0
            else:
                assert res0 == res1

    def verify(self, stack):
        if isinstance(stack, dict):
            stack = stack.values()

        for count, c in enumerate((f for f in stack if inspect.isfunction(f)), 1):
            self.compare(c)
        return count
