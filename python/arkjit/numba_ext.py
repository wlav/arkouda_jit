""" arkouda extensions for numba
"""

import arkouda as ak
import collections
import contextlib
import inspect
import operator
import typing as py_typing

import numpy as np

import numba

import numba.extending as nb_ext
import numba.core.cgutils as nb_cgu
import numba.core.compiler as nb_cmp
import numba.core.compiler_machinery as nb_cpl
import numba.core.decorators as nb_dec
import numba.core.imputils as nb_iutils
import numba.core.pythonapi as nb_pyapi
import numba.core.types as nb_types
import numba.core.typing as nb_typing
import numba.core.typing.templates as nb_tmpl
import numba.core.untyped_passes as nb_untyped_pass
import numba.core.typed_passes as nb_typed_pass

from llvmlite import ir

import sys


__all__ = [
    'optimize',
    'ArkoudaCompiler'
    ]


# -- helpers ----------------------------------------------------------------
ir_byte     = ir.IntType(8)
ir_voidptr  = ir.PointerType(ir_byte)                 # by convention
ir_byteptr  = ir_voidptr                              # for clarity

def type_remap(t: nb_types.Type) -> type:
    if isinstance(t, nb_types.Integer):
        return nb_types.Integer
    if isinstance(t, nb_types.Float):
        return nb_types.Float
    return t

class KeywordPlaceholder(nb_types.PyObject):
    pass


def register_type(pytype, identifier):
    @nb_ext.typeof_impl.register(pytype)
    def typeof_index(val, c) -> nb_types.Type:
        return identifier

    return identifier


# -- general function inliner -----------------------------------------------
def _dummy():
    pass

class InlineUndefinedFunctionType(nb_types.UndefinedFunctionType):
    def __init__(self, dispatcher):
        pysig = inspect.signature(dispatcher.py_func)
        super(InlineUndefinedFunctionType, self).__init__(len(pysig.parameters), [dispatcher])

class TypeofInlineFunction:
    """Automatically inline helpers written on the interactive prompt."""

    def __init__(self):
        self._is_active = False

    def __enter__(self):
        self._is_active = True

    def __exit__(self, tp, val, trace):
        self._is_active = False

    def __call__(self, val, c):
        if not self._is_active or val.__module__ != '__main__':
            return None

      # register dispatcher with Numba
        disp = optimize(inline='always')(val)

      # place the dispatcher back into the module to ensure next call goes
      # through the normal lookup, withing entering this fallback
        setattr(sys.modules[val.__module__], val.__name__, disp)

      # type this function as if it was always a dispatcher (JIT wrapper)
        return nb_types.Dispatcher(disp)

typeof_inline_function = TypeofInlineFunction()
nb_ext.typeof_impl.register(type(_dummy))(typeof_inline_function)


# -- pdarray typing ---------------------------------------------------------

#
# fake type to shuttle arbitrary instances
#

class OpaquePyType(nb_types.Type):
    known_types = set()

    def __init__(self) -> None:
        super(OpaquePyType, self).__init__(name='OpaquePyObject')

opaque_py_type = OpaquePyType()


#
# main arkouda types
#

class PDArrayType(nb_types.Type):
    def __init__(self) -> None:
        super(PDArrayType, self).__init__(name='PDArray')

pdarray_type = register_type(ak.pdarrayclass.pdarray, PDArrayType())

# dtype comes from numpy, where it is a constant; we need it as a variable
# type such that it can be passed as keyword arguments (this is necessary
# b/c the main pdarray creation function requires kwds  and can not unfold)
# TODO: should probably reject dtypes not loaded from the arkouda module
register_type(np.dtype, opaque_py_type)


#
# pdarray signatures
#

class PDArraySignature(nb_tmpl.Signature):
    pass

class PDArrayBinOpSignature(PDArraySignature):
    pass

def pda_signature(kind, return_type: nb_types.Type, fargs: py_typing.Tuple,
        fkwds: py_typing.Dict=None, func: py_typing.Callable=None, **kwds) -> PDArraySignature:
    recvr = kwds.pop('recvr', None)
    assert not kwds
    if kind == 'binop':
        kls = PDArrayBinOpSignature
    else:
        kls = PDArraySignature

    pysig = None
    if fkwds:
        if func is not None:
            pysig = inspect.signature(func)
            assert 'kwargs' in pysig.parameters, "function does not support keywords"
            if 'args' in pysig.parameters:
              # we have the actual types, so unroll them instead
                parms = [
                    inspect.Parameter(f'arg{i}', inspect.Parameter.POSITIONAL_ONLY)
                    for i in range(len(fargs)-len(fkwds))
                ]

              # Numba will try to place the keywords arguments as parameters and can
              # not handle passing the keywords as a simple dict, so fake it (this
              # needs to be reconstructed again when lowering and passed to the C-API)
                for key in fkwds.keys():
                    parms.append(inspect.Parameter(key, inspect.Parameter.POSITIONAL_OR_KEYWORD))

                pysig = inspect.Signature(parameters=parms,
                                          return_annotation=pysig.return_annotation)
        else:
            assert not "keywords used, but no callable provided to determine Python signature"

    return kls(return_type, fargs, recvr, pysig)


#
# pdarray creation
#
def create_creator_overload(func: py_typing.Callable) -> None:
    class PDArrayCreate(nb_tmpl.AbstractTemplate):
        @property
        def key(self) -> py_typing.Callable:
            return func

        def generic(self, args: py_typing.Tuple, kwds: py_typing.Dict) -> PDArraySignature:
          # keywords are passed, unrolled, through additional PyObject* arguments
            if kwds:
                args = args + tuple(KeywordPlaceholder(key) for key in kwds.keys())

          # register a creator lowering implementation for the given argument
          # types (which are assumed to be correct)
            decorate = nb_iutils.lower_builtin(
                func, *tuple(type_remap(x) for x in args))
            decorate(create_creator_lowering(func))

            return pda_signature(
                'constructor', pdarray_type, args, kwds, func=func, recvr=None)

    decorate = nb_tmpl.infer_global(func)
    decorate(PDArrayCreate)


#
# mathematical operations on pdarray
#
class PDArrayBinOp(nb_tmpl.ConcreteTemplate):
    cases = [pda_signature('binop', pdarray_type, (pdarray_type, pdarray_type))]

@nb_tmpl.infer_global(operator.mul)
class PDArrayBinOpMul(PDArrayBinOp):
    pass

@nb_tmpl.infer_global(operator.add)
class PDArrayBinOpAdd(PDArrayBinOp):
    pass

@nb_tmpl.infer_global(operator.sub)
class PDArrayBinOpSub(PDArrayBinOp):
    pass

@nb_tmpl.infer_global(operator.truediv)
class PDArrayBinOpTrueDiv(PDArrayBinOp):
    pass


# -- arkouda types lowering -------------------------------------------------
class CapsuleModel(nb_ext.models.PrimitiveModel):
    def __init__(self, dmm, fe_type):
        be_type = ir_voidptr
        super(CapsuleModel, self).__init__(dmm, fe_type, be_type)

def box_pyobject(typ, val, c):
    ptr = nb_cgu.alloca_once(c.builder, c.pyapi.pyobj)
    c.builder.store(val, ptr)
    return c.builder.load(ptr)

def register_aspyobject_model(numba_type):
    nb_ext.register_model(numba_type)(CapsuleModel)
    nb_ext.box(numba_type)(box_pyobject)

register_aspyobject_model(PDArrayType)
register_aspyobject_model(OpaquePyType)


@nb_iutils.lower_constant(OpaquePyType)
def opaque_constant(context, builder, ty, pyval):
    pyapi = context.get_python_api(builder)
    objptr = context.add_dynamic_addr(builder, id(pyval), info=type(pyval).__name__)
    retptr = nb_cgu.alloca_once(builder, pyapi.pyobj)
    builder.store(objptr, retptr)
    return builder.load(retptr)

# arbitrary case of any opaque type to placeholder type
@nb_iutils.lower_cast(OpaquePyType, KeywordPlaceholder)
def opaque_as_placeholder(context, builder, fromty, toty, val):
    pyapi = context.get_python_api(builder)
    ptr = nb_cgu.alloca_once(builder, pyapi.pyobj)
    builder.store(val, ptr)
    return builder.load(ptr)


#
# pdarray creation
#
def create_creator_lowering(func):
    def imp_creator(context, builder, sig, args):
        pyapi = context.get_python_api(builder)
        gil_state = pyapi.gil_ensure()

      # actual array creation through Python API
        ak_name = context.insert_const_string(builder.module, 'arkouda')
        ak_mod = pyapi.import_module_noblock(ak_name)

        pykwds = None
        for iarg, arg in enumerate(sig.args):
            if isinstance(arg, KeywordPlaceholder):
              # has keywords, split non-keywords and put all keywords in a dict
                pykwds = pyapi.dict_new()
                for sigarg, kwarg in zip(sig.args[iarg:], args[iarg:]):
                    pyapi.dict_setitem_string(pykwds, sigarg.name, kwarg)
                rl_args = args[:iarg]
                break
        else:
          # no keywords case
            rl_args = args

        pyargs = pyapi.tuple_new(len(rl_args))
        for iarg, arg in enumerate(rl_args):
            pyargb = pyapi.from_native_value(sig.args[iarg], arg, None)
            pyapi.tuple_setitem(pyargs, iarg, pyargb)

        pyf = pyapi.object_getattr_string(ak_mod, func.__name__)
        pda = pyapi.call(pyf, pyargs, pykwds)

        pyapi.decref(pyf)
        pyapi.decref(pyargs)
        if pykwds is not None:
            pyapi.decref(pykwds)
        pyapi.decref(ak_mod)

        pyapi.gil_release(gil_state)

      # return result as a void ptr
        result = nb_cgu.alloca_once(builder, pyapi.pyobj)
        builder.store(pda, result)
        return builder.load(result)

    return imp_creator


#
# mathematical operations on pdarray
#
def create_lowering_op(op):
    def imp_op(context, builder, sig, args):
        pyapi = context.get_python_api(builder)
        gil_state = pyapi.gil_ensure()

      # box the arguments
        c = nb_pyapi._BoxContext(context, builder, pyapi, None)
        pyself = box_pyobject(sig.args[0], args[0], c)
        pyargs = (box_pyobject(sig.args[1], args[1], c),)

      # call the Python-side method
        pda = pyapi.call_method(pyself, '__%s__' % op.__name__, pyargs)
        pyapi.incref(pda)

        pyapi.gil_release(gil_state)

      # return result as a void ptr
        result = nb_cgu.alloca_once(builder, pyapi.pyobj)
        builder.store(pda, result)
        return builder.load(result)

    return imp_op

for op in (operator.mul, operator.add, operator.sub, operator.truediv):
    decorate = nb_iutils.lower_builtin(op, pdarray_type, pdarray_type)
    decorate(create_lowering_op(op))


# -- automatic overloading of Arkouda APIs ----------------------------------

#
# pdarray creation
#

import arkouda.pdarraycreation as akcreate

for _fn in akcreate.__all__:
    _f = getattr(akcreate, _fn)
    if callable(_f):
        create_creator_overload(_f)


# -- pdarray optimization passes --------------------------------------------

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
                except KeyError:   # not part of the original Python
                    pass

        return True

    def __str__(self):
        return "common subexpression elimination for PDArrays"

#
# compiler which adds custom Arkouda passes
#
class ArkoudaCompiler(nb_cmp.CompilerBase):
    def compile_extra(self, func):
        with typeof_inline_function:
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

#
# decorator selecting the ArkoudaCompiler
#
def optimize(*args, **kwds):
  # Numba compiler with Arkouda-specific passes and run in
  # object mode by default (to allow simple re-use of Arkouda's
  # pdarray class as-is to generate messages)
    kwds['pipeline_class'] = ArkoudaCompiler
    kwds['nopython' ] = True

  # enable jitting for this function (actual run deferred to first call)
    return nb_dec.jit(*args, **kwds)