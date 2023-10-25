import numpy as np

from common import ArkoudaJITTest
from context import arkouda as ak


class SetOperationsTests(ArkoudaJITTest):
    """Test JITed pdarray set operations"""

    def test_concatenate(self):
        """JITing of arkouda.concatenate"""

        def calc1():
            return ak.concatenate([ak.array([1, 2, 3]), ak.array([4, 5, 6])])

        def calc2():
            return ak.concatenate(
                [ak.array([True, False, True]), ak.array([False, True, True])])

        def calc3():
            return ak.concatenate(
                [ak.array(['one', 'two']), ak.array(['three', 'four', 'five'])])

        assert self.verify(locals())
