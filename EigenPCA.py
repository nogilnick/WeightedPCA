import numpy        as     np
from   scipy.sparse import spmatrix

class EigenPCA:
   
   def __init__(self, n_components=-1, scale=True, T=None):
      self.scale = scale
      self.nc    = n_components
      self.T     = T
   
   def fit(self, X, sample_weight=None):
      m, n = X.shape
      W = ((np.full((m, 1), 1 / m))
              if sample_weight is None else
           (sample_weight / sample_weight.sum()))
      if len(W.shape) == 1:
         W = W[:, None]

      if self.nc < 0:
         self.nc_ = min(m, n)
      else:
         self.nc_ = min(self.nc, m, n)
      
      self.mean_ = Xm = W.T @ X
      
      if self.scale:
         if isinstance(X, spmatrix):
            Xd = np.sqrt(W.T @ X.power(2) - np.square(W.T @ X))
            X  = X.multiply(1 / Xd)
         else:
            Xd = np.sqrt(W.T @ np.square(X) - np.square(W.T @ X))
            X = X * (1 / Xd)
         self.std_ = Xd
      
      Wr = np.sqrt(W)
      if (not self.T) and (m > n):   # Use (X^T)X
         if isinstance(X, spmatrix):
            Xw = X.multiply(Wr)
         else:
            Xw = X * Wr

         C = Xw.T @ Xw - Xm.T @ Xm
      else:                          # Use X(X^T)
         XXm = X @ Xm.T
         C   = X @ X.T - XXm - XXm.T + Xm @ Xm.T
         C   = Wr * C * Wr.T
      
      S, Q = np.linalg.eigh(C)
      S = S[::-1]                    # eigh sorts in ascending order
      Q = Q[:, ::-1]
      
      self.exp_var_ = S
      self.tot_var_ = np.sum(self.exp_var_ )
      self.exp_rat_ = self.exp_var_  / self.tot_var_
   
      # nc_ can also be x in [0, 1] and is taken to mean: take the largest
      # component until the total explained variance is >= x
      if isinstance(self.nc_, float):
         c_ratio = np.cumsum(self.exp_rat_)
         self.nc_ = \
            np.searchsorted(c_ratio, self.nc_, side='right') + 1
         
      Qnc = Q[:, :self.nc_]
      
      if (not self.T) and (m > n):
         C = Qnc     # eigh returns normalized eigenvectors
      else:
         # (A-u)@(A-u)^T = Z and  Z = (P @ D @ P^T) then
         # (A-u)^T @ P is eigenvector matrix for (A-u)^T @ (A-u)
         # Wr * (X - u).T @ Q == (Wr * X).T @ Q - np.outer(u, Wr.T @ Q)
         C  = (Wr * X).T @ Qnc - Xm.T * (Wr.T @ Qnc)
         C /= np.linalg.norm(C, ord=2, axis=0, keepdims=True)  # Need to re-normalize
      
      self.components_ = C
      
      return self

   def fit_transform(self, X, sample_weight=None):
      return self.fit(X, sample_weight=sample_weight).transform(X)

   def transform(self, X):
      if self.scale:
         if isinstance(X, spmatrix):
            X = X.multiply(1 / self.std_)
         else:
            X = X * (1 / self.std_)
            
      return X @ self.components_ - self.mean_ @ self.components_
