""" Arkouda-specific passes for Numba
"""

import numba.core.compiler_machinery as nb_cpl

from .numba_ext import PDArrayBinOpSignature

__all__ = [
    "ArkoudaFunctionPass",
    "ArkoudaCSE",
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

        for block in state.func_ir.blocks.values():
            pda_expr = dict()
            for stmt in block.body:
                try:
                    if isinstance(state.calltypes[stmt.value], PDArrayBinOpSignature):
                        key = (stmt.value.op, stmt.value.lhs, stmt.value.rhs)
                        seen = pda_expr.get(key, None)
                        if seen is not None:
                            stmt.value = seen.target
                        else:
                            pda_expr[key] = stmt
                except KeyError:  # not part of the original Python
                    pass

        return True

    def __str__(self):
        return "common subexpression elimination for PDArrays"
