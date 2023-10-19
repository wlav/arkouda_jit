from base_test import ArkoudaTest

import arkjit
import inspect


class ArkoudaJITTest(ArkoudaTest):
    """Base class for JIT tests"""

    def compare(self, forg, *args, **kwds):
        res0 = forg(*args, **kwds)

        fopt = arkjit.optimize()(forg)
        res1 = fopt(*args, **kwds)

        assert sum(res0.to_ndarray() == res1.to_ndarray()) == len(res0)

    def verify(self, stack):
        if isinstance(stack, dict):
            stack = stack.values()

        for count, c in enumerate((f for f in stack if inspect.isfunction(f)), 1):
            self.compare(c)
        return count
