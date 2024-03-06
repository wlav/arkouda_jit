import numpy as np

from common import ArkoudaJITTest
from context import arkouda as ak


class SortingTests(ArkoudaJITTest):
    """Test JITed pdarray sorting methods"""

    def test_argsort(self):
        """JITing of pdarray argsort"""

        def calc1():
            A = ak.randint(0, 10, 10, seed=2)
            perm = ak.argsort(A)
            return A[perm]

        assert self.verify(locals())
