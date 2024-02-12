import numpy as np

from common import ArkoudaJITTest
from context import arkouda as ak


# helper to test auto comilation/inlining
def sqr(A):
    return A*A


class PassTests(ArkoudaJITTest):
    """Test JITed pdarray set operations"""

    def setup_class(cls):
        PassTests.binop_counter = 0
        PassTests._binop = ak.pdarray._binop

    @staticmethod
    def _count_binop(pda, *args, **kwds):
        PassTests.binop_counter += 1
        return PassTests._binop(pda, *args, **kwds)

    def setup_method(self, meth):
        PassTests.binop_counter = 0
        ak.pdarray._binop = PassTests._count_binop

    def teardown_method(self, meth):
        ak.pdarray._binop = PassTests._binop
        PassTests.binop_counter = 0

    def test_cse(self):
        """Common subexpression elimination"""

        def calc1():
            A = ak.arange(10)
            B = A*A + A*A
            return B

        assert self.verify(locals(), passes=("cse",))

        assert PassTests.binop_counter == \
                 3 + 2            # calc1

    def test_auto_inline(self):
        """Inline functions to enable CSE"""

        def calc1():
            A = ak.arange(10)
            B = sqr(A) + A*A #sqr(A)
            return B

        assert self.verify(locals(), passes=("auto", "cse",))

        assert PassTests.binop_counter == \
                 3 + 2            # calc1
