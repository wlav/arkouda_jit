""" Arkouda JIT compiler based on Numba
"""

import inspect
import sys

import numba.core.compiler as nb_cmp
import numba.core.decorators as nb_dec
import numba.core.dispatcher as nb_disp
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
    def __init__(self, target, typer, state):
        self.target = target
        self.typer = typer
        self.state = state

    def __enter__(self):
        self.typer.state = self.state
        if self.typer.top is None:
            self.typer.top = self.target

    def __exit__(self, tp, val, trace):
        self.typer.top = None
        self.typer.state = None


class TypeofInlineFunction:
    """Automatically inline helpers written on the interactive prompt."""

    def __init__(self):
        self._is_active = False
        self.state = None
        self.top = None

    def __enter__(self):
        self._is_active = True

    def __exit__(self, tp, val, trace):
        self._is_active = False

    def __call__(self, val, c):
        # TODO: not every function can be auto-inlined, and functions may
        # have overloads instead. Therefore, restrict inline to only the
        # main module (interactive use) or functions in the same module as
        # the current function being JITed.
        if not self._is_active or (self.top and\
               (val.__module__ != "__main__" and val.__module__ != self.top.__module__)):
            return None

        # check whether the function has already been replaced (we get here
        # for each call of the function in the closure being compiled)
        disp = getattr(sys.modules[val.__module__], val.__name__)
        if isinstance(disp, nb_disp.Dispatcher):
             return nb_types.Dispatcher(disp)

        # register dispatcher with Numba (will compile on use)
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
    opt_passes = "all"

    def __init__(self, typingctx, targetctx, library, args, return_type, flags, locals):
        super().__init__(
            typingctx, targetctx, library, args, return_type, flags, locals)

        enable_all = self.opt_passes == "all"

        self.cse_pass      = enable_all or "cse" in self.opt_passes
        self.function_pass = enable_all or "function" in self.opt_passes

        self.force_inline  = enable_all or "inline" in self.opt_passes


    def compile_extra(self, func):
        if self.force_inline:
            with typeof_inline_function, CompilerState(func, typeof_inline_function, self.state):
                result = super(ArkoudaCompiler, self).compile_extra(func)
        else:
            result = super(ArkoudaCompiler, self).compile_extra(func)
        return result

    def define_pipelines(self):
        # start with the complete Numba pipeline
        pm = nb_cmp.DefaultPassBuilder.define_nopython_pipeline(self.state)

        # function pass example
        if self.function_pass:
            pm.add_pass_after(ArkoudaFunctionPass, nb_untyped_pass.IRProcessing)

        # common subexpression elimination, first pass after typing is done
        if self.cse_pass:
            pm.add_pass_after(ArkoudaCSE, nb_typed_pass.AnnotateTypes)

        pm.finalize()
        return [pm]


# -- decorator to select the ArkoudaCompiler --------------------------------
def optimize(*args, **kwds):
    # Numba compiler with Arkouda-specific passes and run in
    # object mode by default (to allow simple re-use of Arkouda's
    # pdarray class as-is to generate messages)

    # filter ArkoudaCompiler-specific options
    opt_passes = kwds.get("passes", None)
    if opt_passes is not None:
        # there's no good way to communicate additional options through the
        # Numba compiler definition, this global setting will have to do
        ArkoudaCompiler.opt_passes = opt_passes
        if "inline" in opt_passes and not "inline" in kwds:
            kwds["inline"] = "always"
        del kwds["passes"]

    # use the Arkouda compiler and force nopython mode (soon the only choice)
    kwds["pipeline_class"] = ArkoudaCompiler
    kwds["nopython"] = True

    # enable jitting for this function (actual run deferred to first call)
    return nb_dec.jit(*args, **kwds)
