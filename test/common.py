from base_test import ArkoudaTest

import arkouda as ak
import arkjit
import gc
import inspect
import numpy as np


class ArkoudaJITTest(ArkoudaTest):
    """Base class for JIT tests"""

    binop_counter = 0

    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)
        self.counts = dict()

    def _assert_equal(self, res0, res1):
        if isinstance(res0, (ak.pdarray, ak.Strings)):
            assert sum(res0.to_ndarray() == res1.to_ndarray()) == len(res0)
        elif isinstance(res0, np.ndarray):
            assert sum(res0 == res1) == len(res0)
        elif type(res0) == np.float64:
            assert round(res0-res1, 14) == 0
        else:
            assert res0 == res1

    def compare(self, passes, forg, *args, **kwds):
        mem_level = len(ak.list_symbol_table())


        self.reset_counters()
        res0 = forg(*args, **kwds)
        self.record_counters(forg, optimized=False)

        fopt = arkjit.optimize(passes=passes)(forg)

        self.reset_counters()
        res1 = fopt(*args, **kwds)
        self.record_counters(forg, optimized=True)

        if isinstance(res0, (tuple, list)) and res0:
            for r0, r1 in zip(res0, res1):
                self._assert_equal(r0, r1)
            del r0, r1
        else:
            self._assert_equal(res0, res1)

        del res0, res1
        gc.collect()

        assert mem_level == len(ak.list_symbol_table())

    def setup_class(cls):
        ArkoudaJITTest._binop = ak.pdarray._binop

    @staticmethod
    def _count_binop(pda, *args, **kwds):
        ArkoudaJITTest.binop_counter += 1
        return ArkoudaJITTest._binop(pda, *args, **kwds)

    def setup_method(self, meth):
        ak.pdarray._binop = ArkoudaJITTest._count_binop

    def teardown_method(self, meth):
        ak.pdarray._binop = ArkoudaJITTest._binop

    def reset_counters(self):
        ArkoudaJITTest.binop_counter = 0

    def record_counters(self, func, optimized):
        self.counts[(func, 'binop', optimized)] = ArkoudaJITTest.binop_counter

    def verify(self, stack, passes="all", args=(), kwds={}):
        if isinstance(stack, dict):
            stack = stack.values()

        for i, func in enumerate((f for f in stack if inspect.isfunction(f)), 1):
            self.compare(passes, func, *args, **kwds)

        return i
