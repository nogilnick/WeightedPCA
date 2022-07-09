import numpy as np
from sklearn.decomposition import PCA
from WeightedPCA import WPCA

# Unit tests for transform with scaling
for i in range(9999):
   # Generate a random problem instance
   m, n = 8, 4
   A = np.random.randint(0, 10, size=(m, n))
   # Repeats (or weightings) of the rows of A
   W = np.random.randint(1, 10, size=m)
   # Row i of A is repeated W[i] times
   R = np.repeat(A, W, axis=0)

   # Using sklearn on the full matrix R
   Ra = R.mean(0)
   Rd = R.std(0)
   M  = (R - Ra) / Rd
   
   pca = PCA().fit(M)
   
   TA1 = pca.transform(M)
   # Now using WPCA on the matrix A with weights W
   wpca = WPCA().fit(A, W)
   
   # SVD is sign indeterminate; fix order of signs
   vt1 =  pca.components_[:, 0]
   vt2 = wpca.components_[0]
   for c, (i, j) in enumerate(zip(vt1, vt2)):
      if np.sign(i) != np.sign(j):
         wpca.components_[:, c] *= -1

   # Transform and then repeat rows according to W
   TA2 = np.repeat(wpca.transform(A), W, axis=0)
   assert(np.allclose(TA1, TA2))

# Unit tests for transform without scaling
for i in range(9999):
   # Generate a random problem instance
   m, n = 8, 4
   A = np.random.randint(0, 10, size=(m, n))
   # Repeats (or weightings) of the rows of A
   W = np.random.randint(1, 10, size=m)
   # Row i of A is repeated W[i] times
   R = np.repeat(A, W, axis=0)

   # Using sklearn on the full matrix R
   Ra = R.mean(0)
   M  = (R - Ra)
   
   pca = PCA().fit(M)
   
   TA1 = pca.transform(M)
   # Now using WPCA on the matrix A with weights W
   wpca = WPCA(scale=False).fit(A, W)
   
   # SVD is sign indeterminate; fix order of signs
   vt1 =  pca.components_[:, 0]
   vt2 = wpca.components_[0]
   for c, (i, j) in enumerate(zip(vt1, vt2)):
      if np.sign(i) != np.sign(j):
         wpca.components_[:, c] *= -1

   # Transform and then repeat rows according to W
   TA2 = np.repeat(wpca.transform(A), W, axis=0)
   assert(np.allclose(TA1, TA2))

print('All Tests Passed')
