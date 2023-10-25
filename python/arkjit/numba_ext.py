""" Arkouda extensions for Numba
"""

import collections
import inspect
import operator
import typing as py_typing

import arkouda as ak
import arkouda.numeric as aknum
import arkouda.pdarrayclass as akclass
import arkouda.pdarraycreation as akcreate
import arkouda.pdarraysetops as aksetops
import numba
import numba.core.cgutils as nb_cgu
import numba.core.imputils as nb_iutils
import numba.core.pythonapi as nb_pyapi
import numba.core.types as nb_types
import numba.core.typing.templates as nb_tmpl
import numba.extending as nb_ext
import numpy as np
from llvmlite import ir

__all__ = [
    "PDArrayType",
    "PDArrayBinOpSignature",
]


# -- helpers ----------------------------------------------------------------
ir_byte = ir.IntType(8)
ir_voidptr = ir.PointerType(ir_byte)  # by convention
ir_byteptr = ir_voidptr  # for clarity

ak_types = [nb_types.int64, nb_types.float64]


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


# -- pdarray typing ---------------------------------------------------------

#
# fake type to shuttle arbitrary instances
#
class OpaquePyType(nb_types.Type):
    known_types = set()

    def __init__(self) -> None:
        super(OpaquePyType, self).__init__(name="OpaquePyObject")

opaque_py_type = OpaquePyType()


#
# main arkouda types
#
class PDArrayType(nb_types.Type):
    def __init__(self) -> None:
        super(PDArrayType, self).__init__(name="PDArray")

pdarray_type = register_type(ak.pdarrayclass.pdarray, PDArrayType())

# dtype comes from numpy, where it is a constant; we need it as a variable
# type such that it can be passed as keyword arguments (this is necessary
# b/c the main pdarray creation function requires kwds  and can not unfold)
# TODO: should probably reject dtypes not loaded from the arkouda module
register_type(np.dtype, opaque_py_type)


#
# overload construction
#
class PDArrayOverloadTemplate(nb_tmpl.AbstractTemplate):
    _lowered = collections.defaultdict(set)

    def typeof_args(self, func: py_typing.Callable, args: py_typing.Tuple, kwds: py_typing.Dict) -> py_typing.Tuple:
        """Construct full list of argument types from typed arguments and keywords"""

        if kwds:
            # place keywords in args, use placeholders where needed
            pysig = inspect.signature(func)

            if 'kwargs' in pysig.parameters:
                # keywords are passed, unrolled, through additional PyObject* arguments
                args = args + tuple(KeywordPlaceholder(key) for key in kwds.keys())
            else:
                # keyword types are mapped to the actual positional parameters
                arg_names = list(pysig.parameters.keys())[len(args):]

                _args = list(args)
                _kwds = dict(kwds)
                for name in arg_names:
                    try:
                        _args.append(_kwds.pop(name))
                    except KeyError:
                        _args.append(numba.typeof(pysig.parameters[name].default))

                    if not _kwds:
                        break

                args = tuple(_args)

        return args

    def register_lowering(self, func: py_typing.Callable, args: py_typing.Tuple, lowering: py_typing.Callable) -> None:
        # register a creator lowering implementation for the given argument
        # types (which are assumed to be correct)
        lower_args = tuple(type_remap(x) for x in args)
        if lower_args not in self._lowered[func]:
            decorate = nb_iutils.lower_builtin(func, *lower_args)
            decorate(lowering(func))
            self._lowered[func].add(lower_args)


#
# pdarray signatures
#
class PDArraySignature(nb_tmpl.Signature):
    pass


class PDArrayBinOpSignature(PDArraySignature):
    pass


def pda_signature(
    kind,
    return_type: nb_types.Type,
    fargs: py_typing.Tuple,
    fkwds: py_typing.Dict = None,
    func: py_typing.Callable = None,
    **kwds,
) -> PDArraySignature:
    recvr = kwds.pop("recvr", None)
    assert not kwds
    if kind == "binop":
        kls = PDArrayBinOpSignature
    else:
        kls = PDArraySignature

    pysig = None
    if fkwds:
        if func is not None:
            pysig = inspect.signature(func)
            if "args" in pysig.parameters:
                # we have the actual types, so unroll them instead
                parms = [
                    inspect.Parameter(f"arg{i}", inspect.Parameter.POSITIONAL_ONLY)
                    for i in range(len(fargs) - len(fkwds))
                ]

                # Numba will try to place the keywords arguments as parameters and can
                # not handle passing the keywords as a simple dict, so fake it (this
                # needs to be reconstructed again when lowering and passed to the C-API)
                for key in fkwds.keys():
                    parms.append(inspect.Parameter(key, inspect.Parameter.POSITIONAL_OR_KEYWORD))

                pysig = inspect.Signature(parameters=parms, return_annotation=pysig.return_annotation)
        else:
            assert not "keywords used, but no callable provided to determine Python signature"

    return kls(return_type, fargs, recvr, pysig)


#
# pdarray creation
#
def create_creator_overload(func: py_typing.Callable) -> None:
    class PDArrayCreate(PDArrayOverloadTemplate):
        @property
        def key(self) -> py_typing.Callable:
            return func

        def generic(self, args: py_typing.Tuple, kwds: py_typing.Dict) -> PDArraySignature:
            typed_args = self.typeof_args(func, args, kwds)
            self.register_lowering(func, typed_args, create_generic_lowering)

            return pda_signature("constructor", pdarray_type, typed_args, kwds, func=func, recvr=None)

    decorate = nb_tmpl.infer_global(func)
    decorate(PDArrayCreate)


#
# pdarray annotated methods overloads
#
def create_annotated_overload(func: py_typing.Callable) -> None:
    class PDArrayFunction(PDArrayOverloadTemplate):
        _lowered = collections.defaultdict(set)

        @property
        def key(self) -> py_typing.Callable:
            return func

        def generic(self, args: py_typing.Tuple, kwds: py_typing.Dict) -> PDArraySignature:
            typed_args = self.typeof_args(func, args, kwds)
            self.register_lowering(func, args, create_generic_lowering)

          # derive return type from annotation or arguments; intent is
          # only to identify pdarray's for optimizing passes: in all
          # cases, the generated lowering expects a generic PyObject
            pysig = inspect.signature(func)
            return_type = None
            if pysig.return_annotation in ('pdarray', ak.pdarray):
                return_type = pdarray_type
            else:
                # some heuristics
                if len(typed_args) == 1 and isinstance(typed_args[0], nb_types.containers.List):
                    # operations on a list of arrays; returns the same type of array
                    # TODO: a common alternative is to return ak.Strings when given
                    # ak.pdarray's as input; would need to specialize on the dtype
                    # of the array to capture those
                    return_type = args[0].key[0]

            if return_type is None:
                return_type = opaque_py_type

            return pda_signature("function", return_type, typed_args, kwds, func=func, recvr=None)

    decorate = nb_tmpl.infer_global(func)
    decorate(PDArrayFunction)


#
# numeric/arithmetic operations on pdarray
#
class PDArrayBinOp(nb_tmpl.ConcreteTemplate):
    cases = [pda_signature("binop", pdarray_type, (pdarray_type, pdarray_type))] + \
        [pda_signature("binop", pdarray_type, (pdarray_type, x)) for x in ak_types] + \
        [pda_signature("binop", pdarray_type, (x, pdarray_type)) for x in ak_types]

def _binop_type(op):
    kls = type('PDArrayBinOp'+op.__name__.upper(), (PDArrayBinOp,), {})
    nb_tmpl.infer_global(op)(kls)
    return kls

PDArrayBinOpMUL         = _binop_type(operator.mul)
PDArrayBinOpIMUL        = _binop_type(operator.imul)
PDArrayBinOpADD         = _binop_type(operator.add)
PDArrayBinOpIADD        = _binop_type(operator.iadd)
PDArrayBinOpSUB         = _binop_type(operator.sub)
PDArrayBinOpISUB        = _binop_type(operator.isub)
PDArrayBinOpTRUEDIV     = _binop_type(operator.truediv)
PDArrayBinOpITRUEDIV    = _binop_type(operator.itruediv)


class PDArrayCmp(nb_tmpl.ConcreteTemplate):
    cases = [pda_signature("cmp", pdarray_type, (pdarray_type, pdarray_type))]

@nb_tmpl.infer_global(operator.eq)
class PDArrayEQ(PDArrayCmp):
    pass

@nb_tmpl.infer_global(operator.ne)
class PDArrayNE(PDArrayCmp):
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
def create_generic_lowering(func):
    def imp_creator(context, builder, sig, args):
        pyapi = context.get_python_api(builder)
        gil_state = pyapi.gil_ensure()

        # actual array creation through Python API
        ak_name = context.insert_const_string(builder.module, "arkouda")
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

        env_manager = context.get_env_manager(builder)

        pyargs = pyapi.tuple_new(len(rl_args))
        for iarg, arg in enumerate(rl_args):
            # from_native_value steals a reference from the original `arg`
            context.nrt.incref(builder, sig.args[iarg], arg)
            pyargb = pyapi.from_native_value(sig.args[iarg], arg, env_manager)
            pyapi.tuple_setitem(pyargs, iarg, pyargb)  # steals reference

        pyf = pyapi.object_getattr_string(ak_mod, func.__name__)
        pda = pyapi.call(pyf, pyargs, pykwds)

        pyapi.decref(pyf)
        pyapi.decref(pyargs)
        if pykwds is not None:
            pyapi.decref(pykwds)
        pyapi.decref(ak_mod)

        pyapi.gil_release(gil_state)

        # return result as a PyObject*
        result = nb_cgu.alloca_once(builder, pyapi.pyobj)
        builder.store(pda, result)
        return builder.load(result)

    return imp_creator


#
# numeric/arithmetic operations on pdarray
#
def create_lowering_op(op):
    def imp_op(context, builder, sig, args):
        pyapi = context.get_python_api(builder)
        gil_state = pyapi.gil_ensure()

        # box the arguments
        c = nb_pyapi._BoxContext(context, builder, pyapi, None)
        self_idx = 0; other_idx = 1
        if not isinstance(sig.args[0], PDArrayType):
            self_idx = 1; other_idx = 0

        pyself = box_pyobject(sig.args[self_idx], args[self_idx], c)
        if isinstance(sig.args[other_idx], PDArrayType):
            pyargs = (box_pyobject(sig.args[other_idx], args[other_idx], c),)
        else:
            env_manager = context.get_env_manager(builder)
            pyargs = (pyapi.from_native_value(sig.args[other_idx], args[other_idx], env_manager),)

        # call the Python-side method
        pda = pyapi.call_method(pyself, "__%s__" % op.__name__, pyargs)
        pyapi.incref(pda)

        pyapi.gil_release(gil_state)

        # return result as a void ptr
        result = nb_cgu.alloca_once(builder, pyapi.pyobj)
        builder.store(pda, result)
        return builder.load(result)

    return imp_op


for op in (operator.mul, operator.imul,
           operator.add, operator.iadd,
           operator.sub, operator.isub,
           operator.truediv, operator.itruediv):
    nb_iutils.lower_builtin(op, pdarray_type, pdarray_type)(create_lowering_op(op))
    for x in ak_types:
        nb_iutils.lower_builtin(op, pdarray_type, x)(create_lowering_op(op))
        nb_iutils.lower_builtin(op, x, pdarray_type)(create_lowering_op(op))

for op in (operator.eq, operator.ne):
    nb_iutils.lower_builtin(op, pdarray_type, pdarray_type)(create_lowering_op(op))


# -- automatic overloading of Arkouda APIs ----------------------------------

#
# pdarray creation
#
for _fn in akcreate.__all__:
    _f = getattr(akcreate, _fn)
    if callable(_f):
        create_creator_overload(_f)


#
# operations
#
for mod in [akclass, aknum, aksetops]:
    for _fn in mod.__all__:
        _f = getattr(mod, _fn)
        if callable(_f):
            create_annotated_overload(_f)

