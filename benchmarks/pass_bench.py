from common import ArkoudaJITBench
from context import arkouda as ak

import arkjit
import pytest


preamble = "@pytest.mark.benchmark(group=group, warmup=True)"
configs = [('baseline', 'False',), ('numba', 'True',)]

all_benchmarks = []

ARRAY_SIZE = 1000000


#- common benchmark harness --------------------------------------------------
def basic_bench(benchmark, func, optimize, passes="all", *args, **kwds):
    call = func
    # optimization is done outside benchmark loop, so no warmup call needed
    if optimize: call = arkjit.optimize(passes=passes)(call)
    try:
        ArkoudaJITBench.setUpClass()
        with ArkoudaJITBench() as context:
            benchmark(call, *args, **kwds)
    finally:
        ArkoudaJITBench.tearDownClass()


#- group: common subexpression elemination -----------------------------------
def local_cse():
    A = ak.arange(ARRAY_SIZE)
    B = A*A + A*A
    return B

all_benchmarks.append(('common-subexpr-elemination', (
"""
def test_{0}_local_cse(benchmark, optimize={1}):
    basic_bench(benchmark, local_cse, optimize, passes=("cse",))
""",
)))


#- create all benchmarks in all groups ---------------------------------------
for group, benchmarks in all_benchmarks:
    for bench in benchmarks:
        for config in configs:
            exec(preamble+bench.format(*config))

