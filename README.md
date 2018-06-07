# SparseTensor

Pure python/python3 library for supporting block sparse tensors with arbitrarily many U(1) symmetries

  The library supports a bunch of operations on tensors (see sparsenumpy.py):

  tensor contraction (using tensordot)

  merging and splitting indices

  svd decomposition
  
  truncation
  
  qr decomposition
  
  vectorization of SparseTensor for use in sparse solvers (might not be very fast)
  
testCaseTensor.py does some unittest checks 

The file tensordotprof.py runs a small profiling on the tensordot function. You can visualize the result using e.g. snakeviz
It shows that for tensors with many small blocks, the overhead of python verus numpy.tensordot is large. When the blocksize gets larger than 50, the overhead becomes small and for 80 its negligible
  
