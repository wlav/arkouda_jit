""" Arkouda JIT compiler based on Numba
"""

import inspect
import sys

import numba.core.compiler as nb_cmp
import numba.core.decorators as nb_dec
import numba.core.typed_passes as nb_typed_pass
import numba.core.types as nb_types
import numba.core.untyped_passes as nb_untyped_pass
import numba.extending as nb_ext

from .passes import ArkoudaCSE, ArkoudaFunctionPass

__all__ = ["optimize", "ArkoudaCompiler"]


# -- general function inliner -----------------------------------------------
def _dummy():
    pass


class InlineUndefinedFunctionType(nb_types.UndefinedFunctionType):
    def __init__(self, dispatcher):
        pysig = inspect.signature(dispatcher.py_func)
        super(InlineUndefinedFunctionType, self).__init__(len(pysig.parameters), [dispatcher])


class CompilerState:
    def __init__(self, f, state):
        self.f = f
        self.state = state

    def __enter__(self):
        self.f.state = self.state

    def __exit__(self, tp, val, trace):
        self.f.state = None


class TypeofInlineFunction:
    """Automatically inline helpers written on the interactive prompt."""

    def __init__(self):
        self._is_active = False

    def __enter__(self):
        self._is_active = True

    def __exit__(self, tp, val, trace):
        self._is_active = False

    def __call__(self, val, c):
        if not self._is_active or val.__module__ != "__main__":
            return None

        # register dispatcher with Numba
        disp = optimize(inline="always")(val)

        # place the dispatcher back into the module to ensure next call goes
        # through the normal lookup, withing entering this fallback
        setattr(sys.modules[val.__module__], val.__name__, disp)

        # type this function as if it was always a dispatcher (JIT wrapper)
        return nb_types.Dispatcher(disp)


typeof_inline_function = TypeofInlineFunction()
nb_ext.typeof_impl.register(type(_dummy))(typeof_inline_function)


# -- compiler which adds custom Arkouda passes ------------------------------
class ArkoudaCompiler(nb_cmp.CompilerBase):
    def compile_extra(self, func):
        with typeof_inline_function, CompilerState(typeof_inline_function, self.state):
            result = super(ArkoudaCompiler, self).compile_extra(func)
        return result

    def define_pipelines(self):
        # start with the complete Numba pipeline
        pm = nb_cmp.DefaultPassBuilder.define_nopython_pipeline(self.state)

        # function pass example
        pm.add_pass_after(ArkoudaFunctionPass, nb_untyped_pass.IRProcessing)

        # common subexpression elimination, first pass after typing is done
        pm.add_pass_after(ArkoudaCSE, nb_typed_pass.AnnotateTypes)

        pm.finalize()
        return [pm]


# -- decorator to select the ArkoudaCompiler --------------------------------
def optimize(*args, **kwds):
    # Numba compiler with Arkouda-specific passes and run in
    # object mode by default (to allow simple re-use of Arkouda's
    # pdarray class as-is to generate messages)
    kwds["pipeline_class"] = ArkoudaCompiler
    kwds["nopython"] = True

    # enable jitting for this function (actual run deferred to first call)
    return nb_dec.jit(*args, **kwds)
