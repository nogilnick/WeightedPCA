import numpy as np

# Code for performing a Weighted Principal Component Analysis (WPCA)
# That is, using a matrix X and integer weight vector W to implicitly
# perform PCA on a matrix B formed by taking each row X_i and
# repeating it W_i times
class WPCA:

   def __str__(self):
      return '{}({})'.format(self.__class__.__name__,
                             getattr(self, 'n_components_', ''))

   def __repr__(self):
      return self.__str__()

   def __init__(self, n_components=None, scale=True):
      self.n_components = n_components
      self.scale = scale

   def fit(self, X, sample_weight=None):
      m, n = X.shape
      W = (np.full((m, 1), 1 / m) if sample_weight is None else
           (sample_weight / sample_weight.sum()))
      if len(W.shape) == 1:
         W = W[:, None]

      if self.n_components is None:
         self.n_components_ = n
      else:
         self.n_components_ = min(self.n_components, n)

      self.mean_ = W.T @ X
      X          = X - self.mean_
      if self.scale:
         self.std_  = np.sqrt(W.T @ np.square(X))
         X         /= self.std_

      U, S, VT = np.linalg.svd(np.sqrt(W) * X, full_matrices=False)

      self.exp_var_ = np.square(S) / (m - 1)
      self.tot_var_ = np.sum(self.exp_var_ )
      self.exp_rat_ = self.exp_var_  / self.tot_var_

      # n_components_ can also be x in [0, 1] and is taken to mean: take the largest
      # component until the total explained variance is >= x
      if isinstance(self.n_components_, float):
         c_ratio = np.cumsum(self.exp_rat_)
         self.n_components_ = \
            np.searchsorted(c_ratio, self.n_components_, side='right') + 1

      self.components_ = VT[:self.n_components_].T

      return self

   def fit_transform(self, X, sample_weight=None):
      return self.fit(X, sample_weight=sample_weight).transform(X)

   def transform(self, X):
      X          = X - self.mean_
      if self.scale:
         X      /= self.std_

      return X @ self.components_