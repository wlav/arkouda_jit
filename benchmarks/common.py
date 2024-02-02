from base_test import ArkoudaTest


class ArkoudaJITBench(ArkoudaTest):
    """Base class for JIT benchmarks"""

    def __enter__(self):
        self.setUp()
        return self

    def __exit__(self, *exc):
        self.tearDown()
        return False

