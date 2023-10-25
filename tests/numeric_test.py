import numpy as np

from common import ArkoudaJITTest
from context import arkouda as ak


class NumericTests(ArkoudaJITTest):
    """Test JITed pdarray numeric/arithmetic operations"""

    def test_scalar(self):
        """JITing of scalar operations"""

        def calc1():
            A = ak.arange(10)
            A += 2
            return A

        def calc1a():
            A = ak.arange(10)
            A += 2.
            return A

        def calc2():
            A =  ak.arange(10)
            return A + A

        def calc3():
            A = ak.arange(10)
            return 2 * A

        def calc3a():
            A = ak.arange(10)
            return 2. * A

        def calc4():
            A = ak.arange(10)
            return A == A

        assert self.verify(locals())

    def test_elementwise(self):
        """JITing of elementwise functions"""

        def calc1():
            return ak.abs(ak.arange(-5, -1))

        def calc2():
            return ak.abs(ak.linspace(-5, -1, 5))

        def calc3():
            A = ak.array([1, 10, 100])
            return ak.log(A)

        def calc4():
            A = ak.array([1, 10, 100])
            return ak.log(A) / np.log(10)

        def calc5():
            A = ak.array([1, 10, 100])
            return ak.log(A) / np.log(2)

        def calc6():
            return ak.exp(ak.arange(1, 5))

        def calc7():
            return ak.exp(ak.arange(1, 5))

        def calc8():
            return ak.exp(ak.uniform(5, 1.0, 5.0, seed=2))

        assert self.verify(locals())

    def test_scans(self):
        """JITing of arkouda scans"""

        def calc1():
            return ak.cumsum(ak.arange(1, 5))

        def calc2():
            return ak.cumsum(ak.uniform(5, 1.0, 5.0, seed=2))

        def calc3():
            return ak.cumsum(ak.randint(0, 1, 5, seed=2, dtype=ak.bool))

        def calc4():
            return ak.cumprod(ak.arange(1, 5))

        def calc5():
            return ak.cumprod(ak.uniform(5, 1.0, 5.0, seed=2))

        assert self.verify(locals())

    def test_reductions(self):
        """JITing of arkouda reductions"""

        def calc1():
            return ak.any(ak.arange(1, 5))

        def calc2():
            return ak.all(ak.arange(5))

        def calc3():
            return ak.is_sorted(ak.arange(5))

        def calc4():
            return ak.sum(ak.arange(5, dtype=ak.int64))

        def calc4a():
            return ak.sum(ak.arange(5, dtype=ak.float64))

        def calc5():
            return ak.prod(ak.arange(5, dtype=ak.int64))

        def calc5a():
            return ak.prod(ak.arange(5, dtype=ak.float64))

        def calc6():
            return ak.min(ak.uniform(5, 1.0, 5.0, seed=2))

        def calc7():
            return ak.max(ak.uniform(5, 1.0, 5.0, seed=2))

        def calc8():
            return ak.argmin(ak.uniform(5, 1.0, 5.0, seed=2))

        def calc9():
            return ak.argmax(ak.uniform(5, 1.0, 5.0, seed=2))

        def calc10():
            return ak.mean(ak.uniform(5, 1.0, 5.0, seed=2))

        def calc11():
            return ak.var(ak.uniform(5, 1.0, 5.0, seed=2))

        def calc12():
            return ak.std(ak.uniform(5, 1.0, 5.0, seed=2))

        def calc13():
            A = ak.array([10, 5, 1, 3, 7, 2, 9, 0])
            return ak.mink(A, 3)

        def calc14():
            A = ak.array([10, 5, 1, 3, 7, 2, 9, 0])
            return ak.maxk(A, 3)

        def calc14():
            A = ak.array([10, 5, 1, 3, 7, 2, 9, 0])
            return ak.argmink(A, 3)

        def calc15():
            A = ak.array([10, 5, 1, 3, 7, 2, 9, 0])
            return ak.argmaxk(A, 3)

        assert self.verify(locals())

    def test_where(self):
        """JITing of arkouda where"""

        def calc1():
            a1 = ak.arange(1, 10)
            a2 = ak.ones(9, dtype=np.int64)
            cond = a1 < 5
            return ak.where(cond, a1, a2)

        def calc2():
            a1 = ak.arange(1,10)
            a2 = ak.ones(9, dtype=np.int64)
            cond = a1 == 5
            return ak.where(cond, a1, a2)

        def calc3():
            a1 = ak.arange(1,10)
            a2 = 10
            cond = a1 < 5
            return ak.where(cond, a1, a2)

        assert self.verify(locals())
