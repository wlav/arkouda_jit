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
import numba.core.typing.builtins as nb_blt
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
ir_errcode = ir.IntType(32)

ak_types = [nb_types.int64, nb_types.float64, nb_types.bool_]


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
# helper for methods
#
class PDArrayMethod(nb_types.Callable):
    def __init__(self, func):
        super(PDArrayMethod, self).__init__('pdarray.%s' % func.__name__)

        self.sig = None
        self._func = func

    def is_precise(self):
        return True          # only b/c of the use of opaque types where necessary

    def get_call_type(self, context, args, kwds):
        if self.sig is not None:
            return self.sig

        pysig = inspect.signature(self._func)

        return_type = opaque_py_type
        if pysig.return_annotation in ('pdarray', ak.pdarray):
            return_type = pdarray_type

        self.sig = pda_signature("method", return_type, args, kwds, func=self._func, recvr=pdarray_type)

        print("sig:", self.sig)

        @nb_iutils.lower_getattr(pdarray_type, self._func.__name__)#, *args)
        def lower_method_call(context, builder, typ, args, name=self._func.__name__):
            print("GATENKAAS!", len(args), args)

            pyapi = context.get_python_api(builder)
            gil_state = pyapi.gil_ensure()

            # box the arguments
            c = nb_pyapi._BoxContext(context, builder, pyapi, None)

            # regular dispatch of self and 1 or more args
            pyself = box_pyobject(typ, args, c)
            env_manager = context.get_env_manager(builder)
            pyargs = tuple(pyapi.from_native_value(
                sig.args[idx], args[idx], env_manager) for idx in range(1, len(args)))

            # call the Python-side method
            pda = pyapi.call_method(pyself, name, pyargs)

            err_occurred = nb_cgu.is_not_null(builder, pyapi.err_occurred())
            pyapi.gil_release(gil_state)

            with nb_cgu.if_unlikely(builder, err_occurred):
                builder.ret(ir.Constant(ir_errcode, -1))

            # return result as a void ptr
            pyapi.incref(pda)
            result = nb_cgu.alloca_once(builder, pyapi.pyobj)
            builder.store(pda, result)
            return builder.load(result)

        print("returning ....")
        return self.sig

        ol = CppFunctionNumbaType(self._func.__overload__(numba_arg_convertor(args)), self._is_method)

        thistype = None
        if self._is_method:
            thistype = nb_types.voidptr

        self.ret_type = cpp2numba(ol._func.__cpp_reflex__(cpp_refl.RETURN_TYPE))
        ol.sig = nb_typing.Signature(
            return_type=self.ret_type,
            args=args,
            recvr=thistype)

        extsig = ol.sig
        if self._is_method:
            self.ret_type = ol.sig.return_type
            args = (nb_types.voidptr, *args)
            extsig = nb_typing.Signature(
                return_type=ol.sig.return_type, args=args, recvr=None)

        self._impl_keys[args] = ol
        self._arg_set_matched = numba_arg_convertor(args)


        @nb_iutils.lower_builtin(ol, *args)
        def lower_external_call(context, builder, sig, args,
                ty=nb_types.ExternalFunctionPointer(extsig, ol.get_pointer), pyval=self._func, is_method=self._is_method):
            ptrty = context.get_function_pointer_type(ty)
            ptrval = context.add_dynamic_addr(
                builder, ty.get_pointer(pyval), info=str(pyval))
            fptr = builder.bitcast(ptrval, ptrty)
            return context.call_function_pointer(builder, fptr, args)

        return ol.sig

    def get_call_signatures(self):
        print("CALL SIGANTURES:", self.sig)
        return [self.sig], False

    def get_impl_key(self, sig):
        print("FROM KEY:", sig)
        return self

    @property
    def key(self):
        return self._func

@nb_iutils.lower_constant(PDArrayMethod)
def frozen_method(context, builder, ty, pyval):
    return


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

PDArrayBinOpLT          = _binop_type(operator.eq)
PDArrayBinOpLE          = _binop_type(operator.ne)
PDArrayBinOpLT          = _binop_type(operator.lt)
PDArrayBinOpLE          = _binop_type(operator.le)
PDArrayBinOpGT          = _binop_type(operator.gt)
PDArrayBinOpGE          = _binop_type(operator.ge)

@nb_tmpl.infer_global(operator.getitem)
class PDArrayGetItem(nb_tmpl.AbstractTemplate):
    def generic(self, args: py_typing.Tuple, kwds: py_typing.Dict) -> PDArraySignature:
        assert not kwds
        arr_type, idx_type = args
        if isinstance(arr_type, PDArrayType):
            if isinstance(idx_type, nb_types.Integer):
                return pda_signature("getitem", opaque_py_type, (arr_type, idx_type))
            elif isinstance(idx_type, (PDArrayType, nb_types.SliceType)):
                return pda_signature("getitem", pdarray_type, (arr_type, idx_type))

@nb_tmpl.infer_global(operator.setitem)
class PDArraySetItem(nb_tmpl.AbstractTemplate):
    def generic(self, args: py_typing.Tuple, kwds: py_typing.Dict) -> PDArraySignature:
        assert not kwds
        arr_type, idx_type, val_type = args
        if isinstance(arr_type, PDArrayType) and \
                isinstance(idx_type, (PDArrayType, nb_types.Integer, nb_types.SliceType)):
            return pda_signature("setitem", nb_types.none, (arr_type, idx_type, val_type))


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


# there is no boxing method for slices in Numba, so add one
@nb_ext.box(nb_types.SliceType)
def box_slice(typ, val, c):
    ret_ptr = nb_cgu.alloca_once(c.builder, c.pyapi.pyobj)
    sli = nb_cgu.create_struct_proxy(typ)(c.context, c.builder, value=val)

    class_obj = c.pyapi.unserialize(c.pyapi.serialize_object(slice))
    pyargs = tuple(c.box(nb_types.int64, d) for d in (sli.start, sli.stop, sli.step))

    res = c.pyapi.call_function_objargs(class_obj, pyargs)

    for obj in pyargs:
        c.pyapi.decref(obj)

    c.builder.store(res, ret_ptr)       # may be null
    return c.builder.load(ret_ptr)


#
# pdarray creation
#
def create_generic_lowering(func, module="arkouda"):
    def imp_generic_lowering(context, builder, sig, args):
        pyapi = context.get_python_api(builder)
        gil_state = pyapi.gil_ensure()

        # actual array creation through Python API
        mod_name = context.insert_const_string(builder.module, module)
        mod = pyapi.import_module_noblock(mod_name)

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

        pyf = pyapi.object_getattr_string(mod, func.__name__)
        pyres = pyapi.call(pyf, pyargs, pykwds)

        pyapi.decref(pyf)
        pyapi.decref(pyargs)
        if pykwds is not None:
            pyapi.decref(pykwds)
        pyapi.decref(mod)

        err_occurred = nb_cgu.is_not_null(builder, pyapi.err_occurred())
        pyapi.gil_release(gil_state)

        with nb_cgu.if_unlikely(builder, err_occurred):
            builder.ret(ir.Constant(ir_errcode, -1))

        # return result as a PyObject*
        result = nb_cgu.alloca_once(builder, pyapi.pyobj)
        builder.store(pyres, result)
        return builder.load(result)

    return imp_generic_lowering


#
# numeric/arithmetic operations on pdarray
#
def create_lowering_op(op, binop=True):
    def imp_op(context, builder, sig, args):
        pyapi = context.get_python_api(builder)
        gil_state = pyapi.gil_ensure()

        # box the arguments
        c = nb_pyapi._BoxContext(context, builder, pyapi, None)

        if binop:
            # assume associativity
            self_idx = 0; other_idx = 1
            if not isinstance(sig.args[0], PDArrayType):
                self_idx = 1; other_idx = 0

            pyself = box_pyobject(sig.args[self_idx], args[self_idx], c)
            if isinstance(sig.args[other_idx], PDArrayType):
                pyargs = (box_pyobject(sig.args[other_idx], args[other_idx], c),)
            else:
                env_manager = context.get_env_manager(builder)
                pyargs = (pyapi.from_native_value(sig.args[other_idx], args[other_idx], env_manager),)
        else:
            # regular dispatch of self and 1 or more args
            pyself = box_pyobject(sig.args[0], args[0], c)
            env_manager = context.get_env_manager(builder)
            pyargs = tuple(pyapi.from_native_value(
                sig.args[idx], args[idx], env_manager) for idx in range(1, len(args)))

        # call the Python-side method
        pda = pyapi.call_method(pyself, "__%s__" % op.__name__, pyargs)

        err_occurred = nb_cgu.is_not_null(builder, pyapi.err_occurred())
        pyapi.gil_release(gil_state)

        with nb_cgu.if_unlikely(builder, err_occurred):
            builder.ret(ir.Constant(ir_errcode, -1))

        # return result as a void ptr
        pyapi.incref(pda)
        result = nb_cgu.alloca_once(builder, pyapi.pyobj)
        builder.store(pda, result)
        return builder.load(result)

    return imp_op


# arithmetic
for op in (operator.mul, operator.imul,
           operator.add, operator.iadd,
           operator.sub, operator.isub,
           operator.truediv, operator.itruediv,
           operator.eq,  operator.ne,
           operator.lt,  operator.le,
           operator.gt,  operator.ge):
    nb_iutils.lower_builtin(op, pdarray_type, pdarray_type)(create_lowering_op(op))
    for x in ak_types:
        nb_iutils.lower_builtin(op, pdarray_type, x)(create_lowering_op(op))
        nb_iutils.lower_builtin(op, x, pdarray_type)(create_lowering_op(op))

# comparison
for op in (operator.eq, operator.ne):
    nb_iutils.lower_builtin(op, pdarray_type, pdarray_type)(create_lowering_op(op))

# indexing
for idx_type in (PDArrayType, nb_types.Integer, nb_types.SliceType):
    nb_iutils.lower_builtin(operator.getitem, pdarray_type, idx_type)(
        create_lowering_op(operator.getitem, binop=False))

    for x in ak_types + [pdarray_type]:
        nb_iutils.lower_builtin(operator.setitem, pdarray_type, idx_type, x)(
            create_lowering_op(operator.setitem, binop=False))


# -- automatic overloading of Arkouda APIs ----------------------------------

#
# pdarray methods
#
@nb_tmpl.infer_getattr
class PDArrayFieldResolver(nb_tmpl.AttributeTemplate):
    key = PDArrayType

    def generic_resolve(self, typ, attr):
        ft = typ.__dict__.get(attr, None)
        if ft is not None:
            return ft

        if not isinstance(typ, PDArrayType):
            return

        try:
            _f = getattr(ak.pdarray, attr)
            if inspect.isfunction(_f):
                ft = PDArrayMethod(_f)
                typ.__dict__[attr] = ft
                return ft
        except AttributeError:
            pass


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

