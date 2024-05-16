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
import arkouda.sorting as aksorting
import numba
import numba.core.cgutils as nb_cgu
import numba.core.imputils as nb_iutils
import numba.core.pythonapi as nb_pyapi
import numba.core.registry as nb_reg
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
ir_bool = ir.IntType(8)
ir_byte = ir.IntType(8)
ir_i64  = ir.IntType(64)
ir_voidptr = ir.PointerType(ir_byte)  # by convention
ir_byteptr = ir_voidptr  # for clarity
ir_errcode = ir.IntType(32)

ak_types = {
    np.float64:   nb_types.float64,
    np.int64:     nb_types.int64,
    np.bool_:     nb_types.bool_,
}

ak_annotated_types = tuple(ak_types.keys()) + tuple('np.'+t.__name__ for t in ak_types.keys())


def target_refresh():
    """
    Update registries mid-compilation after having registered new lowering
    implementations at run-time.
    """

    nb_reg.cpu_target.target_context.refresh()


def _generalized_type(t: nb_types.Type) -> type:
    if isinstance(t, (nb_types.Integer, nb_types.Float, PDArrayType)):
        return type(t)
    return t

def lower_args(args: tuple) -> tuple:
    return tuple(_generalized_type(x) for x in args)


class KeywordPlaceholder(nb_types.PyObject):
    pass


def register_type(pytype, identifier):
    @nb_ext.typeof_impl.register(pytype)
    def typeof_index(val, c) -> nb_types.Type:
        return identifier

    return identifier


def from_native_with_ref(c, b, pyapi, ty, v, env):
    # from_native_value steals a reference from the original ll value, the
    # numba-runtime incref adds one C-side
    pyarg = pyapi.from_native_value(ty, v, env)
    if isinstance(ty, PDArrayType):
        pyapi.incref(pyarg)
    else:
        c.nrt.incref(b, ty, v)
    return pyarg


def pyargs_from_native(context, builder, argtypes, argvalues):
    pyapi = context.get_python_api(builder)
    boxctxt = nb_pyapi._BoxContext(context, builder, pyapi, None)
    env = context.get_env_manager(builder)

    pyargs = list()
    for ty, v in zip(argtypes, argvalues):
        if isinstance(ty, PDArrayType):
            pyarg = box_pyobject_with_ref(ty, v, boxctxt, pyapi)
        else:
            pyarg = from_native_with_ref(context, builder, pyapi, ty, v, env)
        pyargs.append(pyarg)

    return tuple(pyargs)


def load_result(context, builder, pyres, rtype):
    # return result as actual type, an unrolled tuple for multiple returns,
    # or a PyObject* (for pdarray's and opaque objects)
    pyapi = context.get_python_api(builder)

    if isinstance(rtype, nb_types.Float):
        result = nb_cgu.alloca_once(builder, ir.DoubleType())
        f = pyapi.float_as_double(pyres)
        pyapi.decref(pyres)
        builder.store(f, result)
    elif isinstance(rtype, nb_types.Integer):
        result = nb_cgu.alloca_once(builder, ir_i64)
        i = pyapi.long_as_longlong(pyres)
        pyapi.decref(pyres)
        builder.store(i, result)
    elif isinstance(rtype, nb_types.Boolean):
        result = nb_cgu.alloca_once(builder, ir_bool)
        b32 = pyapi.object_istrue(pyres)
        pyapi.decref(pyres)
        builder.store(builder.trunc(b32, ir_bool), result)
    elif isinstance(rtype, nb_types.Tuple):
        size = sig.return_type.count
        pack = ir.LiteralStructType([pyapi.pyobj]*size)
        result = nb_cgu.alloca_once(builder, pack)
        unrolled = list()
        for i in range(size):
            v = pyapi.tuple_getitem(pyres, i)
            pyapi.incref(v)   # v is borrowed
            unrolled.append(v)
        builder.store(nb_cgu.make_anonymous_struct(builder, unrolled), result)
        pyapi.decref(pyres)
    else:
        result = nb_cgu.alloca_once(builder, pyapi.pyobj)
        builder.store(pyres, result)

    return builder.load(result)


def call_method_with_cleanup(context, builder, pyself, name, pyargs, rtype, gil_state):
    pyapi = context.get_python_api(builder)
    pyres = pyapi.call_method(pyself, name, pyargs)

    err_occurred = nb_cgu.is_not_null(builder, pyapi.err_occurred())

    for pyarg in pyargs:
        pyapi.decref(pyarg)
    pyapi.decref(pyself)
    pyapi.gil_release(gil_state)

    with nb_cgu.if_unlikely(builder, err_occurred):
        builder.ret(ir.Constant(ir_errcode, -1))

    return load_result(context, builder, pyres, rtype)


# -- pdarray typing ---------------------------------------------------------

#
# fake type to shuttle arbitrary instances
#
class OpaquePyType(nb_types.Type):
    def __init__(self, hint=None) -> None:
        try:
            hname = hint.__name__
        except AttributeError:
            hname = str(hint)

        super(OpaquePyType, self).__init__(name="OpaquePyObject_%s" % hname)

        self.hint = hint

opaque_py_type = OpaquePyType()


#
# main arkouda types
#
class PDArrayType(nb_types.Type):
    mutable = True       # ie. type is allowed to be lhs

    def __init__(self, dtype) -> None:
        super(PDArrayType, self).__init__(name="PDArray_%s" % str(dtype))

        self.dtype = dtype

pdarray_f64 = register_type(ak.pdarrayclass.pdarray, PDArrayType(dtype=nb_types.float64))
pdarray_i64 = register_type(ak.pdarrayclass.pdarray, PDArrayType(dtype=nb_types.int64))
pdarray_b1  = register_type(ak.pdarrayclass.pdarray, PDArrayType(dtype=nb_types.bool_))
assert len(ak_types) == 3    # add pdarray types if more added to ak_types
pdarray_type = pdarray_f64   # default type

def get_pdarray_type(dtype):
    if dtype == ak.float64 or dtype == np.float64:
        return pdarray_f64
    elif dtype == ak.int64 or dtype == np.int64:
        return pdarray_i64
    elif dtype == ak.bool  or dtype == np.bool_:
        return pdarray_b1
    return pdarray_type


# dtype comes from numpy, where it is a constant; we need it as a variable
# type such that it can be passed as keyword arguments (this is necessary
# b/c the main pdarray creation function requires kwds and can not unfold)
# TODO: should probably reject dtypes not loaded from the arkouda module
def annotation2nbtype(dtype):
    if isinstance(dtype, np.dtype):
        dtype = dtype.type
    if isinstance(dtype, str):
        dtype = eval(dtype)
    return ak_types[dtype]

dtype_opaque_types = {}
for dt in (np.float64, np.int64, np.bool_):
    dtype_opaque_types[np.dtype(dt)] = OpaquePyType(dt)
    dtype_opaque_types[dt] = dtype_opaque_types[np.dtype(dt)]

@nb_ext.typeof_impl.register(np.dtype)
def typeof_dtype(val, c) -> nb_types.Type:
    return dtype_opaque_types[val]


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

        # TODO: verify given args against pysig args

        # user a recvr ('self') as that will play better with BoundFunction, even
        # as the key used is an unbound method with 'self' explicitly added as an
        # argument to match against
        self.sig = pda_signature(
            "method", return_type, args, kwds, func=self._func, recvr=pdarray_type)

        @nb_iutils.lower_builtin((PDArrayType, self._func.__name__), PDArrayType, lower_args(*args))
        def lower_method(context, builder, typ, val, name=self._func.__name__):
            pyapi = context.get_python_api(builder)
            gil_state = pyapi.gil_ensure()

            # box the arguments: regular dispatch of self and N args; the bound
            # method holds a reference to self already, so none is set here
            boxctxt = nb_pyapi._BoxContext(context, builder, pyapi, None)
            pyself = box_pyobject(typ, val[0], boxctxt)
            pyargs = pyargs_from_native(context, builder, self.sig.args[1:], args[1:])

            # call the Python-side method and return result as void ptr
            return call_method_with_cleanup(
                context, builder, pyself, name, pyargs, typ, gil_state)

        target_refresh()

        return self.sig

    def get_call_signatures(self):
        return [self.sig], False

    def get_impl_key(self, sig):
        return (PDArrayType, self._func.__name__)

    @property
    def key(self):
        return self._func


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
        args = lower_args(args)
        if args not in self._lowered[func]:
            decorate = nb_iutils.lower_builtin(func, *args)
            decorate(lowering(func))
            self._lowered[func].add(args)
            target_refresh()


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

            return_type = pdarray_type
            try:
                dtype = kwds['dtype']
                if isinstance(dtype, OpaquePyType):
                    return_type = get_pdarray_type(dtype.hint)
            except KeyError:
                pass

            return pda_signature("constructor", return_type, typed_args, kwds, func=func, recvr=None)

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
            self.register_lowering(func, typed_args, create_generic_lowering)

            # derive return type from annotation or arguments; intent is
            # only to identify pdarray's for optimizing passes: in all
            # cases, the generated lowering expects a generic PyObject
            pysig = inspect.signature(func)
            return_type = None
            if pysig.return_annotation in ('pdarray', ak.pdarray):
                return_type = pdarray_type
            elif pysig.return_annotation in ak_annotated_types:
                return_type = annotation2nbtype(pysig.return_annotation)
            elif py_typing.get_origin(pysig.return_annotation) == tuple:
                rtargs = list()
                for a in py_typing.get_args(pysig.return_annotation):
                    if a in ('pdarray', ak.pdarray):
                        rtargs.append(pdarray_type)
                    else:
                        rtargs.append(opaque_py_type)
                return_type = nb_types.Tuple(rtargs)
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
        [pda_signature("binop", pdarray_type, (pdarray_type, x)) for x in ak_types.values()] + \
        [pda_signature("binop", pdarray_type, (x, pdarray_type)) for x in ak_types.values()]

def _binop_type(op):
    kls = type('PDArrayBinOp'+op.__name__.upper(),
               (PDArrayBinOp,),
               {'__module__': PDArrayBinOp.__module__})
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
                return pda_signature("getitem", arr_type.dtype, (arr_type, idx_type))
            elif isinstance(idx_type, (PDArrayType, nb_types.SliceType)):
                return pda_signature("getitem", arr_type, (arr_type, idx_type))

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

def box_pyobject_with_ref(typ, val, c, pyapi):
    boxed = box_pyobject(typ, val, c)
    pyapi.incref(boxed)
    return boxed


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

        env = context.get_env_manager(builder)

        pyargs = pyapi.tuple_new(len(rl_args))
        for iarg, arg in enumerate(rl_args):
            pyargb = from_native_with_ref(
                context, builder, pyapi, sig.args[iarg], arg, env)
            # tuple_setitem will steal an additional reference
            context.nrt.incref(builder, sig.args[iarg], arg)
            pyapi.tuple_setitem(pyargs, iarg, pyargb)

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

        return load_result(context, builder, pyres, sig.return_type)

    return imp_generic_lowering


#
# numeric/arithmetic operations on pdarray
#
def create_lowering_op(op, binop=True, inplace=False):
    def imp_op(context, builder, sig, args):
        pyapi = context.get_python_api(builder)
        name = "__%s__" % op.__name__

        gil_state = pyapi.gil_ensure()

        # box the arguments
        boxctxt = nb_pyapi._BoxContext(context, builder, pyapi, None)

        if binop:
            # assume associativity
            self_idx = 0; other_idx = 1
            if not isinstance(sig.args[0], PDArrayType):
                self_idx = 1; other_idx = 0

            pyself = box_pyobject_with_ref(sig.args[self_idx], args[self_idx], boxctxt, pyapi)
            if isinstance(sig.args[other_idx], PDArrayType):
                pyargs = (box_pyobject_with_ref(sig.args[other_idx], args[other_idx], boxctxt, pyapi),)
            else:
                env = context.get_env_manager(builder)
                pyargs = (from_native_with_ref(
                    context, builder, pyapi, sig.args[other_idx], args[other_idx], env),)
        else:
            # regular dispatch of self and 1 or more args; self needs a ref-count as
            # it's taking from the argument tuple, not passed directly
            pyself = box_pyobject_with_ref(sig.args[0], args[0], boxctxt, pyapi)
            pyargs = pyargs_from_native(context, builder, sig.args[1:], args[1:])

        # call the Python-side method and return result as void ptr
        result = call_method_with_cleanup(
            context, builder, pyself, name, pyargs, sig.return_type, gil_state)

        return result

    return imp_op


# arithmetic
for op in (operator.mul, operator.imul,
           operator.add, operator.iadd,
           operator.sub, operator.isub,
           operator.truediv, operator.itruediv,
           operator.eq,  operator.ne,
           operator.lt,  operator.le,
           operator.gt,  operator.ge):
    nb_iutils.lower_builtin(op, PDArrayType, PDArrayType)(create_lowering_op(op))
    for x in ak_types.values():
        nb_iutils.lower_builtin(op, PDArrayType, x)(create_lowering_op(op))
        nb_iutils.lower_builtin(op, x, PDArrayType)(create_lowering_op(op))

# comparison
for op in (operator.eq, operator.ne):
    nb_iutils.lower_builtin(op, PDArrayType, PDArrayType)(create_lowering_op(op))

# indexing
for idx_type in (PDArrayType, nb_types.Integer, nb_types.SliceType):
    nb_iutils.lower_builtin(operator.getitem, PDArrayType, idx_type)(
        create_lowering_op(operator.getitem, binop=False))

    for x in list(ak_types.values()) + [pdarray_type]:
        nb_iutils.lower_builtin(operator.setitem, PDArrayType, idx_type, x)(
            create_lowering_op(operator.setitem, binop=False))


# -- automatic overloading of Arkouda APIs ----------------------------------

#
# pdarray methods
#
@nb_tmpl.infer_getattr
class PDArrayFieldResolver(nb_tmpl.AttributeTemplate):
    key = PDArrayType
    is_method = True    # TODO: does this matter for data fields?

    def generic_resolve(self, typ, attr):
        ft = typ.__dict__.get(attr, None)
        if ft is not None:
            return ft

        if not isinstance(typ, PDArrayType):
            return

        try:
            _f = getattr(ak.pdarray, attr)
            if inspect.isfunction(_f):
                class PDMethodTemplate(nb_tmpl.AbstractTemplate):
                    key = (self.key, attr)
                    method = PDArrayMethod(_f)

                    def generic(self, args, kws):
                        sig = self.method.get_call_type(self.context, args, kws)
                        return sig.as_method()

                bf = nb_types.BoundFunction(PDMethodTemplate, typ)
                typ.__dict__[attr] = bf

                return bf

        except AttributeError:
            pass


#
# pdarray creation
#
def _getf(mod, _fn):
    try:
        # some functions have a legacy version, so try on the arkouda
        # module first, in case the configuration has an override
        _f = getattr(ak, _fn)
    except AttributeError:
        _f = getattr(mod, _fn)
    return _f

for _fn in akcreate.__all__:
    _f = _getf(akcreate, _fn)
    if callable(_f):
        create_creator_overload(_f)

#
# operations
#
for mod in [akclass, aknum, aksetops]:
    for _fn in mod.__all__:
        _f = _getf(mod, _fn)
        if callable(_f):
            create_annotated_overload(_f)

#
# sorting
#
for mod in [aksorting]:
    for _fn in mod.__all__:
        _f = _getf(mod, _fn)
        if callable(_f):
            create_annotated_overload(_f)
