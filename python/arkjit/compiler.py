""" Arkouda JIT compiler based on Numba
"""

import inspect
import sys

import numba.core.compiler as nb_cmp
import numba.core.decorators as nb_dec
import numba.core.typed_passes as nb_typed_pass
import numba.core.untyped_passes as nb_untyped_pass

from .passes import (AutoFunctionInlinerPass,
                     ArkoudaCSE,
                     ArkoudaFunctionPass,
                    )

__all__ = ["optimize", "ArkoudaCompiler"]


# -- compiler which adds custom Arkouda passes ------------------------------
class ArkoudaCompiler(nb_cmp.CompilerBase):
    opt_passes = "all"

    def __init__(self, typingctx, targetctx, library, args, return_type, flags, locals):
        super().__init__(
            typingctx, targetctx, library, args, return_type, flags, locals)

        enable_all = self.opt_passes == "all"

        self.cse_pass      = enable_all or "cse" in self.opt_passes
        self.function_pass = enable_all or "function" in self.opt_passes

        self.auto_compile  = enable_all or "auto" in self.opt_passes

    def define_pipelines(self):
        # start with the complete Numba pipeline
        pm = nb_cmp.DefaultPassBuilder.define_nopython_pipeline(self.state)

        # function pass example
        if self.function_pass:
            pm.add_pass_after(ArkoudaFunctionPass, nb_untyped_pass.IRProcessing)

        if self.auto_compile:
            # there's no "add_pass_before" function; WithLifting is right
            # before InlineClosureLikes, which needs to see the compiled result
            # for (optional) inlining to work, so use that instead
            pm.add_pass_after(AutoFunctionInlinerPass, nb_untyped_pass.WithLifting)

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
        del kwds["passes"]

    # use inlining by default
    kwds["inline"] = kwds.get("inline", "always")

    # use the Arkouda compiler and force nopython mode (soon the only choice)
    kwds["pipeline_class"] = ArkoudaCompiler
    kwds["nopython"] = True

    # enable jitting for this function (actual run deferred to first call)
    return nb_dec.jit(*args, **kwds)
