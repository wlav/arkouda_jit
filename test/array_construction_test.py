import numpy as np

from common import ArkoudaJITTest
from context import arkouda as ak


class ArrayConstructionTests(ArkoudaJITTest):
    """Test JITed pdarray constructors"""

    def test_from_list(self):
        """JITing of pdarray construction from Python lists"""

        def aj_calc1():
            l = [0, 1, 2, 3, 4]
            return ak.array(l)

        assert self.verify(locals())

    def test_from_numpy(self):
        """JITing of pdarray construction from numpy arrays"""

        def calc1():
            a = np.array([0, 1, 2, 3, 4])
            return ak.array(a)

        assert self.verify(locals())

    def test_constants(self):
        """JITing of constant constructors"""

        def calc1():
            return ak.zeros(5, dtype=ak.int64)

        def calc1a():
            zeros = ak.zeros(5, dtype=ak.int64)
            return ak.zeros_like(zeros)

        def calc2():
            return ak.zeros(5, dtype=ak.float64)

        def calc3a():
            zeros = ak.zeros(5, dtype=ak.float64)
            return ak.zeros_like(zeros)

        def calc3():
            return ak.zeros(5, dtype=ak.bool_)

        def calc3a():
            zeros = ak.zeros(5, dtype=ak.bool_)
            return ak.zeros_like(zeros)

        def calc4():
            return ak.ones(5, dtype=ak.int64)

        def calc4a():
            ones = ak.ones(5, dtype=ak.int64)
            return ak.ones_like(ones)

        def calc5():
            return ak.ones(5, dtype=ak.float64)

        def calc5a():
            ones = ak.ones(5, dtype=ak.float64)
            return ak.ones_like(ones)

        def calc6():
            return ak.ones(5, dtype=ak.bool_)

        def calc6a():
            ones = ak.ones(5, dtype=ak.bool_)
            return ak.ones_like(ones)

        assert self.verify(locals())

    def test_arange(self):
        """JITing of arkouda.arange"""

        def calc1():
            return ak.arange(10)

        def calc1a():
            return ak.arange(10, dtype=ak.float64)

        def calc2():
            return ak.arange(0, 5, 1)

        def calc3():
            return ak.arange(5, 0, -1)

        def calc4():
            return ak.arange(0, 10, 2)

        def calc4a():
            return ak.arange(0, 10, 2, dtype=ak.float64)

        def calc5():
            return ak.arange(-5, -10, -1)

        assert self.verify(locals())

    def test_linspace(self):
        """JITing of arkouda.linspace"""

        def calc1():
            return ak.linspace(0, 1, 5)

        def calc2():
            return ak.linspace(start=1, stop=0, length=5)

        def calc3():
            return ak.linspace(start=-5, stop=0, length=5)

        assert self.verify(locals())

    def test_random(self):
        """JITing of arkouda.random"""

        def calc1():
            return ak.randint(1, 5, 10, seed=2)

        def calc2():
            return ak.randint(1, 5, 3, dtype=ak.float64, seed=2)

        def calc3():
            return ak.randint(1, 5, 10, dtype=ak.bool_, seed=2)

        assert self.verify(locals())

