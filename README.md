arkjit: Numba-based JIT for Arkouda
===================================

Client-side JIT and code lifter for Arkouda, based on Numba. Basic example:

```python
import arkouda as ak
import arkjit

import os

ak.connect(connect_url=os.getenv('ARKOUDA_URL'))

@arkjit.optimize()       # enable client-side JIT
def calc():
    A = ak.arange(10)
    B = A*A + A*A        # common sub-expression will be eliminated
    return B

try:
    B = calc()
    print(B)
finally:
    ak.disconnect()
    pass
```

----

Arkouda documentation: https://bears-r-us.github.io/arkouda/ <br>
Bug reports/feedback: https://github.com/wlav/arkouda_jit/issues
