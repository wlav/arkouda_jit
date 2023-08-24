from base_test import ArkoudaTest
from context import arkouda as ak

import arkjit


class ArrayConstructionTests(ArkoudaTest):
    """Test JITed pdarray constructors"""

    def compare(self, forg, *args, **kwds):
        res0 = forg(*args, **kwds)
        print('result0:', res0)

        print("optimizing:", forg)
        fopt = arkjit.optimize()(forg)
        print("done!")
        res1 = fopt(*args, **kwds)
        print("result1:", res1)

        assert sum(res0.to_ndarray() == res1.to_ndarray()) == len(res0)

    def test_arange(self):
        """Test JITing of arkouda.arange"""

        def calc1():
            A = ak.arange(10)
            return A

        self.compare(calc1)

        def calc2():
            A = ak.arange(10, dtype=ak.float64)
            return A

        self.compare(calc2)

        def calc3():
            A = ak.arange(4, 40, 2)
            return A

        def calc4():
            A = ak.arange(4, 40, 2, dtype=ak.float64)
            return A

        for c in [calc1, calc2, calc3, calc4]:
             self.compare(c)

    def test_fromlist(self):
        """Test construction of pdarrays from Python lists"""

        def calc1():
            l = [0, 1, 2, 3, 4]
            A = ak.array(l)
            return A

        self.compare(calc1)
