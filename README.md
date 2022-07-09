# WeightedPCA
Simple Python class for performing a Weighted PCA (WPCA).

See [this blog plost](https://nicholastsmith.wordpress.com/2022/07/08/weighted-pca/) for more details on how the algorithm functions.

## Example
```python

import numpy as np
from WeightedPCA import WPCA

m, n = 8, 4
A = np.random.randint(0, 10, size=(m, n))
# Repeats (or weightings) of the rows of A
W = np.random.randint(1, 10, size=m)

wpca = WPCA().fit(A, W)
TA   = np.repeat(wpca.transform(A), W, axis=0)
```