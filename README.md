# SparseTensor

Pure python/python3 library for supporting block sparse tensors with arbitrarily many U(1) symmetries

The library supports a bunch of operations on tensors (see sparsenumpy.py):
  tensor contraction (using tensordot)
  merging and splitting indices
  svd decomposition
  truncation
  qr decomposition
  vectorization of SparseTensor for use in sparse solvers (might not be very fast)
  
