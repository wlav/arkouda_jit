""" Arkouda-specific passes for Numba
"""

import collections
import types as pytypes

import numba.core.compiler_machinery as nb_cpl
import numba.core.ir as nb_ir
import numba.core.ir_utils as nb_iru
import numba.core.lowering as nb_lower
import numba.core.typed_passes as nb_typed_pass
import numba.core.types as nb_types
import numba.core.typing as nb_typing
import numba.parfors.parfor_lowering as nb_parfor_lower

from llvmlite import ir

from .numba_ext import (PDArrayType,
                        pdarray_f64,
                        PDArrayBinOpSignature,
                        )
from .runtime import (cleanup_container,
                      cleanup_type,
                      )

__all__ = [
    "ArkoudaFunctionPass",
    "ArkoudaCSE",
    "ArkoudaDel",
    "AutoFunctionInlinerPass",
    "NativeLoweringWithDel",
    "NativeParforLoweringWithDel",
]


# -- helpers ----------------------------------------------------------------
class DecRef(nb_ir.Stmt):
    def __init__(self, value, loc):
        assert isinstance(value, str)
        assert isinstance(loc, nb_ir.Loc)
        self.value = value
        self.loc = loc

    def __str__(self):
        return "decref %s" % self.value

class IncRef(nb_ir.Stmt):
    def __init__(self, value, loc):
        assert isinstance(value, str)
        assert isinstance(loc, nb_ir.Loc)
        self.value = value
        self.loc = loc

    def __str__(self):
        return "incref %s" % self.value


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
class ArkoudaDel(nb_cpl.LoweringPass):
    """
    Explicit deletion of Arkouda objects

    Numba ignores del statements b/c it's not possible to know when to call
    them, since the trace normally operates on unboxed objects. Since pdarrays
    remain boxed, normal ref-counting will have the destructor called as
    appropriate by re-inserting decrefs for dels.

    There is one exception to this: a returned object will be deleted in the
    block

    This pass requires the lowering pass to support the DecRef instruction.
    """

    _name = "arkouda_del"

    def __init__(self):
        super().__init__()

    def _incref(self, ref, instr, state, newbody):
        arg = state.typemap.get(ref, None)
        if isinstance(arg, PDArrayType):
            newbody.append(IncRef(ref, instr.loc))
            return True
        return False

    def run_pass(self, state):
        print("Arkouda explicit deletion pass")

        modified = False

        pda_containers = dict()
        for block in state.func_ir.blocks.values():
            newbody = list()
            deferred = collections.defaultdict(list)
            for instr in block.body:
                if isinstance(instr, nb_ir.Del):
                    ref = instr.value
                    arg = state.typemap.get(ref, None)
                    if isinstance(arg, PDArrayType):
                        newbody.append(DecRef(ref, instr.loc))
                        modified = True
                    else:
                        if ref in deferred:
                            for defer in reversed(deferred[ref]):
                                newbody.append(DecRef(defer, instr.loc))
                            del deferred[ref]
                            modified = True
                        elif ref in pda_containers:
                            # loop over the container and decref all elements

                            # container value and type
                            cont, tc = pda_containers[ref]

                            # load the container cleanup function
                            cleanup = nb_ir.Var(block.scope, ref+'.cleanup', instr.loc)
                            state.typemap[cleanup.name] = cleanup_type(state.typingctx, tc)
                            newbody.append(
                                nb_ir.Assign(
                                    nb_ir.Global(cleanup.name, cleanup_container, instr.loc),
                                    cleanup, instr.loc
                                ))

                            # decref all elements of the container
                            res = nb_ir.Var(block.scope, ref+'.cleanup_ok', instr.loc)
                            state.typemap[res.name] = nb_types.bool_
                            e = nb_ir.Expr.call(cleanup, (cont,), {}, instr.loc)
                            state.calltypes[e] = nb_typing.signature(nb_types.bool_, tc)
                            newbody.append(nb_ir.Assign(e, res, instr.loc))

                            del pda_containers[ref]
                            modified = True

                        newbody.append(instr)

                elif isinstance(instr, nb_ir.Assign):
                    # References can be borrowed/stolen if used in certain ops (this
                    # happens for assignments as the lhs needs to persist beyond the
                    # call for this to matter). In principle, decref should only be
                    # deferred, but it is simpler to pass the reference explicitly,
                    # as the borrower will eventually be deleted as well.
                    expr = instr.value
                    if isinstance(expr, nb_ir.Expr):
                        if expr.op == "cast" and "return_value" in instr.target.name:
                            if self._incref(expr.value.name, instr, state, newbody):
                                modified = True
                            else:      # not a pdarray, but may be a container
                                try:
                                    del deferred[expr.value.name]
                                except KeyError:
                                    pass
                        elif expr.op == "getattr":
                            arg = state.typemap.get(expr.value.name, None)
                            if self._incref(expr.value.name, instr, state, newbody):
                                modified = True
                            else:
                                # capture bound container type functions
                                arg = state.typemap.get(expr.value.name, None)
                                if isinstance(arg, (nb_types.List, nb_types.Tuple)) and\
                                        expr.attr in ('append',):
                                    pda_containers[expr.value.name] = (expr.value, arg)
                        elif expr.op == "call":
                            callee = state.typemap.get(expr.func.name, None)
                            if isinstance(callee, nb_types.BoundFunction) and\
                                    callee.key[0] == 'list.append':
                                vars = expr.list_vars()
                                if self._incref(vars[1].name, instr, state, newbody):
                                    modified = True
                        elif expr.op == "static_getitem":
                            # getitem call on containers of PDArray's (PDArray indexing that results
                            # in a PDArray, e.g. slicing, is handled explicitly already)
                            val = state.typemap.get(expr.value.name, None)
                            ret = state.typemap.get(instr.target.name, None)
                            if not isinstance(val, PDArrayType) and isinstance(ret, PDArrayType):
                                ref_target = nb_ir.Var(instr.target.scope, instr.target.name+'.ref', instr.loc)
                                state.typemap[ref_target.name] = ret
                                newbody.append(instr)
                                newbody.append(nb_ir.Assign(instr.target, ref_target, instr.loc))
                                newbody.append(IncRef(ref_target.name, instr.loc))
                                # ref_target is never decref'ed: the original target will be
                                modified = True
                                continue    # special case as instr already added to newbody above
                        elif expr.op == "build_list" or expr.op == "build_tuple":
                            to_defer = list()
                            for item in expr.items:
                                if self._incref(item.name, instr, state, newbody):
                                    to_defer.append(item.name)
                                    modified = True
                            if to_defer:
                                deferred[instr.target.name] += to_defer

                    elif isinstance(expr, nb_ir.Var):
                        if self._incref(expr.name, instr, state, newbody):
                            modified = True

                    newbody.append(instr)

                else:
                    newbody.append(instr)

            if modified:
                block.body = newbody

        return modified

    def __str__(self):
        return "lifetime management for PDArrays"


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


# -- lowering passes --------------------------------------------------------
def _handle_decref(self, inst):
    pyapi = self.context.get_python_api(self.builder)
    value = self.loadvar(inst.value)
    pyapi.decref(value)

def _handle_incref(self, inst):
    pyapi = self.context.get_python_api(self.builder)
    value = self.loadvar(inst.value)
    pyapi.incref(value)

class LowerWithDel(nb_lower.Lower):
    def lower_inst(self, inst):
        if isinstance(inst, DecRef):
            return _handle_decref(self, inst)
        elif isinstance(inst, IncRef):
            return _handle_incref(self, inst)
        return super().lower_inst(inst)

@nb_cpl.register_pass(mutates_CFG=True, analysis_only=False)
class NativeLoweringWithDel(nb_typed_pass.NativeLowering):
    """Add explicit deletion of Arkouda objects to NativeLowering.
    """

    _name = "native_lowering_with_del"

    @property
    def lowering_class(self):
        return LowerWithDel


class ParForLowerWithDel(nb_parfor_lower.ParforLower):
    def lower_inst(self, inst):
        if isinstance(inst, DecRef):
            return _handle_decref(self, inst)
        elif isinstance(inst, IncRef):
            return _handle_incref(self, inst)
        return super().lower_inst(inst)

@nb_cpl.register_pass(mutates_CFG=True, analysis_only=False)
class NativeParforLoweringWithDel(nb_typed_pass.NativeParforLowering):
    """Adds explicit deletion of Arkouda objects to NativeParforLowering.
    """

    _name = "native_parfor_lowering_with_del"

    @property
    def lowering_class(self):
        return ParForLowerWithDel
