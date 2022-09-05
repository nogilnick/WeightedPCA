# Weighted PCA
Python classes for performing a weighted principal component analysis (PCA).

## WPCA
A class for performing a weighted principal component analysis (WPCA) using the singular value decomposition (SVD).

See [this blog plost](https://nicholastsmith.wordpress.com/2022/07/08/weighted-pca/) for more details on how the algorithm functions.

### Example
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

## EigenPCA
A class for performing a weighted principal component analysis (WPCA) using the eigenvalue decomposition of either `XX*` or `X*X`. If not specified, the approach is chosen to the be the more efficient of the two. The method supports sparse matrices and is able to efficiently decompose <i>m</i>x<i>n</i> rectangular matrices (i.e. when `m << n` or `m >> n`)

### Example
```python

from scipy.sparse import random
from EigenPCA import EigenPCA

Z = random(100000, 1000, density=0.005, format='csc')

EP = EigenPCA(n_components=4)

X = EP.fit_transform(SM).A
```
