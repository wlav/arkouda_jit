import numpy as np

from common import ArkoudaJITTest
from context import arkouda as ak


class SummarizingTests(ArkoudaJITTest):
    """Test JITed pdarray summarizing methods"""

    def test_descriptive(self):
        """JITing of pdarray descriptive methods"""

        def calc1():
            A = ak.randint(-10, 11, 1000, seed=2)
            return A.min()

        def calc2():
            A = ak.randint(-10, 11, 1000, seed=2)
            return A.max()

        def calc3():
            A = ak.randint(-10, 11, 1000, seed=2)
            return A.sum()

        def calc4():
            A = ak.randint(-10, 11, 1000, seed=2)
            return A.mean()

        def calc5():
            A = ak.randint(-10, 11, 1000, seed=2)
            return A.var()

        def calc6():
            A = ak.randint(-10, 11, 1000, seed=2)
            return A.std()

        assert self.verify(locals())

    def test_histogram(self):
        """JITing of pdarray histogram method"""

        def calc1():
            A = ak.arange(0, 10, 1)
            nbins = 3
            h, b = ak.histogram(A, bins=nbins)
            return b

        assert self.verify(locals())
