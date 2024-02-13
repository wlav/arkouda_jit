import numpy as np

from common import ArkoudaJITTest
from context import arkouda as ak


# helper to test auto compilation/inlining
def sqr(A):
    return A*A


class PassTests(ArkoudaJITTest):
    """Test JITed pdarray set operations"""

    def test_cse(self):
        """Common subexpression elimination"""

        def calc1():
            A = ak.arange(10)
            B = A*A + A*A
            return B

        assert self.verify(locals(), passes=("cse",))

        assert self.counts[(calc1, 'binop', True)] == 2

    def test_auto_inline(self):
        """Inline functions to enable CSE"""

        def calc1():
            A = ak.arange(10)
            B = sqr(A) + A*A
            return B

        def calc2():
            A = ak.arange(10)
            B = A*A + sqr(A)
            return B

        def calc3():
            A = ak.arange(10)
            B = A*A + sqr(A)
            return B

        assert self.verify(locals(), passes=("auto", "cse",))

        assert self.counts[(calc1, 'binop', False)] == 3

        assert self.counts[(calc1, 'binop', True)] == 2
        assert self.counts[(calc2, 'binop', True)] == 2
        assert self.counts[(calc3, 'binop', True)] == 2
