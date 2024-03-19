""" Arkouda JIT compiler based on Numba
"""

import inspect
import sys

import numba.core.compiler as nb_cmp
import numba.core.compiler_machinery as nb_mach
import numba.core.decorators as nb_dec
import numba.core.typed_passes as nb_typed_pass
import numba.core.untyped_passes as nb_untyped_pass

from .passes import (AutoFunctionInlinerPass,
                     ArkoudaCSE,
                     ArkoudaDel,
                     ArkoudaFunctionPass,
                     NativeLoweringWithDel,
                     NativeParforLoweringWithDel,
                     )

__all__ = ["optimize", "ArkoudaCompiler"]


# -- compiler which adds custom Arkouda passes ------------------------------
class ArkoudaPassBuilder(nb_cmp.DefaultPassBuilder):
    """
    Redefines the lowering_pipeline to support DecRef/IncRef instructions;
    these methods are otherwise the same as those from the DefaultPassBuilder.
    """

    # Copyright (c) 2012, Anaconda, Inc.
    # All rights reserved.
    #
    # Redistribution and use in source and binary forms, with or without
    # modification, are permitted provided that the following conditions are
    # met:
    #
    # Redistributions of source code must retain the above copyright notice,
    # this list of conditions and the following disclaimer.
    #
    # Redistributions in binary form must reproduce the above copyright
    # notice, this list of conditions and the following disclaimer in the
    # documentation and/or other materials provided with the distribution.
    # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
    # "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
    # LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
    # A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    # HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    # SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    # LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    # DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    # THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

    @staticmethod
    def define_nopython_pipeline(state, name='nopython'):
        """Returns an nopython mode pipeline based PassManager
        """
        # compose pipeline from untyped, typed and lowering parts
        apb = ArkoudaPassBuilder
        pm = nb_mach.PassManager(name)
        untyped_passes = apb.define_untyped_pipeline(state)
        pm.passes.extend(untyped_passes.passes)

        typed_passes = apb.define_typed_pipeline(state)
        pm.passes.extend(typed_passes.passes)

        lowering_passes = apb.define_nopython_lowering_pipeline(state)
        pm.passes.extend(lowering_passes.passes)

        pm.finalize()
        return pm

    @staticmethod
    def define_nopython_lowering_pipeline(state, name='nopython_lowering'):
        pm = nb_mach.PassManager(name)
        # legalise
        pm.add_pass(nb_typed_pass.NoPythonSupportedFeatureValidation,
                    "ensure features that are in use are in a valid form")
        pm.add_pass(nb_typed_pass.IRLegalization,
                    "ensure IR is legal prior to lowering")
        # Annotate only once legalized
        pm.add_pass(nb_typed_pass.AnnotateTypes, "annotate types")
        # lower
        if state.flags.auto_parallel.enabled:
            pm.add_pass(NativeParforLoweringWithDel, "native parfor lowering with del")
        else:
            pm.add_pass(NativeLoweringWithDel, "native lowering with del")
        pm.add_pass(nb_typed_pass.NoPythonBackend, "nopython mode backend")
        pm.add_pass(nb_typed_pass.DumpParforDiagnostics, "dump parfor diagnostics")
        pm.finalize()
        return pm


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
        pm = ArkoudaPassBuilder.define_nopython_pipeline(self.state)

        # function pass example
        if self.function_pass:
            pm.add_pass_after(ArkoudaFunctionPass, nb_untyped_pass.IRProcessing)

        if self.auto_compile:
            # there's no "add_pass_before" function; WithLifting is right
            # before InlineClosureLikes, which needs to see the compiled result
            # for (optional) inlining to work, so use that instead
            pm.add_pass_after(AutoFunctionInlinerPass, nb_untyped_pass.WithLifting)

        last_typed_arkouda_pass = nb_typed_pass.AnnotateTypes

        # common subexpression elimination, first pass after typing is done
        if self.cse_pass:
            pm.add_pass_after(ArkoudaCSE, nb_typed_pass.AnnotateTypes)
            last_typed_arkouda_pass = ArkoudaCSE

        # required pass to ensure proper ref-counting on Arkouda types (this
        # has to be the last typed pass after all Arkouda-specific passes)
        pm.add_pass_after(ArkoudaDel, last_typed_arkouda_pass)

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
