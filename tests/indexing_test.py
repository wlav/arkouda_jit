import numpy as np

from common import ArkoudaJITTest
from context import arkouda as ak


class IndexingTests(ArkoudaJITTest):
    """Test JITed pdarray indexing operations"""

    def test_indexing(self):
        """JITing of indexing operations"""

        def calc1():
            A = ak.arange(10)
            return A[5]

        def calc2():
            A =  ak.arange(10)
            A[5] = 42
            return A

        assert self.verify(locals())

    def test_slicing(self):
        """JITing of slicing operations"""

        def calc1():
            A = ak.arange(0, 10, 1)
            return A[2:6]

        def calc2():
            A = ak.arange(0, 10, 1)
            return A[::2]

        def calc3():
            A = ak.arange(0, 10, 1)
            return A[3::-1]

        def calc4():
            A = ak.arange(0, 10, 1)
            A[1::2] = ak.zeros(5)
            return A

        assert self.verify(locals())

    def test_gather_scatter(self):
        """JITing of gather/scatter operations"""

        def calc1():
            A = ak.arange(10, 20, 1)
            inds = ak.array([8, 2, 5])
            return A[inds]

        """
        # fails in Arkouda (?), not in the JIT
        def calc2():
            A = ak.arange(10, 20, 1)
            inds = ak.array([8, 2, 5])
            A[inds] = ak.zeros(3)
            return A
        """

        assert self.verify(locals())

    def test_logical(self):
        """JITing of logical indexing"""

        def calc1():
            A = ak.arange(0, 10, 1)
            inds = ak.zeros(10, dtype=ak.bool)
            inds[2] = True
            inds[5] = True
            return A[inds]

        def calc1a():
            A = ak.arange(0, 10, 1)
            inds = ak.zeros(10, dtype=ak.bool)
            inds[2] = True
            inds[5] = True
            A[inds] = 42
            return A

        def calc2():
            B = ak.arange(0, 10, 1)
            lim = 10//2
            B[B < lim] = B[:lim] * -1
            return B

        assert self.verify(locals())
