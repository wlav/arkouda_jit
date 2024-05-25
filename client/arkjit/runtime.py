""" Arkouda client-side JIT runtime support
"""

import numba as nb
import numba.core.imputils as nb_iutils
import numba.core.types as nb_types
import numba.core.typing as nb_typing
import numba.core.typing.templates as nb_tmpl
import numba.extending as nb_ext

from .numba_ext import (PDArrayType,
                        )

__all__ = [
    "cleanup_container",
]


def decref_pda(item):
    assert 0 and "dummy placeholder"

def cleanup_container(container):
    assert 0 and "dummy placeholder"


def lower_decref_pda(context, builder, sig, args):
    pyapi = context.get_python_api(builder)
    pyapi.decref(args[0])

def lower_cleanup_container(context, builder, sig, args):
    def cleanup_impl(cont):
        if not cont:
            return False

        for pda in cont:
            decref_pda(pda)
        return True

    return context.compile_internal(
        builder, cleanup_impl, sig, args, locals=dict(c=sig.return_type))


@nb.extending.type_callable(decref_pda)
def type_decref_pda(context):
    def typer(arr, kwds=None, reverse=None):
        return nb_typing.signature(nb_types.void, arr)
    return typer

templates = dict()
def cleanup_type(context, typ):
    global templates
    try:
        template = templates[typ]
    except KeyError:
        class CleanupTemplate(nb_typing.templates.ConcreteTemplate):
            cases = [nb_typing.signature(nb_types.bool_, typ)]
            key = cleanup_container

        template = CleanupTemplate
        templates[typ] = template

    f = nb_types.Function(template)

    # compulsory registrations
    f.get_call_type(context, (typ,), {})
    nb_iutils.lower_builtin(cleanup_container, typ)(lower_cleanup_container)
    nb_iutils.lower_builtin(decref_pda, typ.dtype)(lower_decref_pda)

    return f

