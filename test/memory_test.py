import arkjit
import numpy as np
import numba as nb

from common import ArkoudaJITTest
from context import arkouda as ak


class MemoryTests(ArkoudaJITTest):
    """Explicit tests of memory use corner cases"""

    def test_pipeline(self):
        """Verify arkouda-specific life-time management"""

        import arkjit.compiler as ark_cmp
        import numba.core.compiler as nb_cmp

        state = nb_cmp.StateDict()
        state.func_ir = None
        state.flags = nb_cmp.Flags()

        nb_pb  = nb_cmp.DefaultPassBuilder()
        ark_pb = ark_cmp.ArkoudaPassBuilder()

        # verify that the Arkouda pass builder only replaces
        for pl in ['define_nopython_pipeline',
                   'define_nopython_lowering_pipeline']:
            nb_pm  = getattr(nb_pb, pl)(state)
            ark_pm = getattr(ark_pb, pl)(state)
            assert len(nb_pm.passes) == len(ark_pm.passes)

    def test_returns(self):
        """Memory leak tests of JITed calls"""

        def calc1():
            return [ak.arange(10), ak.arange(10)]

        def calc2():
            return (ak.arange(10), ak.arange(10))

        def calc3():
            return 1, ak.arange(10)

        assert self.verify(locals())

    def test_getitem(self):
        """Memory access control of elements indexed from a container"""

        def calc1():
            t1 = [ak.arange(10)]
            t2 = [ak.arange(10)]
            t = t1[0]*t2[0]
            return t

        def calc2():
            k = [ak.arange(0, 10, 1)]
            ak.sum(k[0])
            return k

        assert self.verify(locals())

    def test_append(self):
        """Memory access control of list.append"""

        def calc1():
            A = ak.arange(0, 10, 1)
            B = ak.arange(0, 10, 1)
            l = list()
            l.append(A)
            l.append(B)
            return l

        def calc2():
            out = []
            for r in range(4):
                M = ak.arange(0, 10, 1)
                out.append(M)

            return None

        def calc3():
            out = []
            for r in range(4):
                M = ak.arange(0, 10, 1)
                out.append(M)

            return out

        def calc4():         # cross-check
            out = []
            for r in range(4):
                out.append(r)

            return out

        assert self.verify(locals())

    def test_aliasing(self):
        """Memory control of aliased containers"""

        @arkjit.optimize()
        def prep1():
            out = []
            for r in range(4):
                M = ak.arange(0, 10, 1)
                out.append(M)
            return out

        @arkjit.optimize(inline='never')
        def prep2():
            out = []
            for r in range(4):
                M = ak.arange(0, 10, 1)
                out.append(M)
            return out

        def calc1():
            out = prep1()
            s = 0
            for i in range(len(out)):
                s += ak.sum(out[i])
            return s

        def calc2():
            out = prep2()
            s = 0
            for i in range(len(out)):
                s += ak.sum(out[i])
            return s

        assert self.verify((calc1, calc2))
