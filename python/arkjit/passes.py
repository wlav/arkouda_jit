""" Arkouda-specific passes for Numba
"""

import types as pytypes

import numba.core.compiler_machinery as nb_cpl
import numba.core.ir as nb_ir
import numba.core.ir_utils as nb_iru

from .numba_ext import (PDArrayType,
                        PDArrayBinOpSignature,
                        )

__all__ = [
    "ArkoudaFunctionPass",
    "ArkoudaCSE",
    "AutoFunctionInlinerPass",
]


# -- function passes --------------------------------------------------------
@nb_cpl.register_pass(mutates_CFG=True, analysis_only=False)
class ArkoudaFunctionPass(nb_cpl.FunctionPass):
    _name = "arkouda_function_pass"

    def __init__(self):
        super().__init__()

    def run_pass(self, state):
        print("Arkouda function pass")
        return False

    def __str__(self):
        return "Arkouda function pass"


@nb_cpl.register_pass(mutates_CFG=True, analysis_only=False)
class AutoFunctionInlinerPass(nb_cpl.FunctionPass):
    _name = "arkjit_auto_inliner"

    def __init__(self):
        nb_cpl.FunctionPass.__init__(self)

    def run_pass(self, state):
        """Automatically compile closures; with forced inline.
        """

        modified = False

        work_list = list(state.func_ir.blocks.items())
        completed = dict()
        while work_list:
            _label, block = work_list.pop()
            for instr in block.body:
                if isinstance(instr, nb_ir.Assign):
                    expr = instr.value
                    if isinstance(expr, nb_ir.Expr) and expr.op == 'call':
                        func_def = nb_iru.get_definition(state.func_ir, expr.func)
                        if isinstance(func_def, nb_ir.Global):
                            if isinstance(func_def.value, pytypes.FunctionType):
                                if not func_def.value in completed:
                                    from arkjit.compiler import optimize
                                    disp = optimize(inline="always")(func_def.value)
                                    completed[func_def.value] = disp

                                func_def.value = completed[func_def.value]
                                modified = True

        return modified


# -- block passes -----------------------------------------------------------
@nb_cpl.register_pass(mutates_CFG=True, analysis_only=False)
class ArkoudaCSE(nb_cpl.LoweringPass):
    """
    Common Subexpression Elmination (CSE) for Arkouda operations
    """

    _name = "arkouda_cse"

    def __init__(self):
        super().__init__()

    def run_pass(self, state):
        print("Arkouda common subexpression elimination pass")

        modified = False

        for block in state.func_ir.blocks.values():
            pda_expr = dict()
            canonical = dict()
            for instr in block.body:
                if not isinstance(instr, nb_ir.Assign):
                    continue

                # lookup original typed python statement (if none, then
                # this IR statement was added by a Numba pass)
                ct = state.calltypes.get(instr.value, None)
                if isinstance(ct, PDArrayBinOpSignature):
                    lhs = canonical.get(instr.value.lhs.name, instr.value.lhs)
                    rhs = canonical.get(instr.value.rhs.name, instr.value.rhs)
                    key = (instr.value.op, lhs, rhs)

                    seen = pda_expr.get(key, None)
                    if seen is not None:
                        instr.value = seen.target
                        modified = True
                    else:
                        pda_expr[key] = instr

                    continue

                # track assignments from inlines
                if isinstance(instr.value, nb_ir.Var):
                    tp = state.typemap.get(instr.value.name, None)
                    if isinstance(tp, PDArrayType):
                        canonical[instr.target.name] = instr.value

                    continue

        return modified

    def __str__(self):
        return "common subexpression elimination for PDArrays"
