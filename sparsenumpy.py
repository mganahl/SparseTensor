#!/usr/bin/env python
import numpy as np
import sys
import itertools
import warnings
import qutilities as utils
try:
    spt=sys.modules['SparseTensor']
except KeyError:
    import SparseTensor as spt

herm=lambda x:np.conj(np.transpose(x))


class Error(Exception):
    """Base class for exceptions in this module."""
    pass

class TensorKeyError(Error):
    def __init__(self,message='TensorKeyError'):
        self.message=message
    pass
class TensorSizeError(Error):
    def __init__(self,message='TensorKeyError'):
        self.message=message
    pass

def test(mssg):
    print(mssg)
    input()

def buildTensor(tensor):
    warnings.warn("buildTensor is deprecated; use SparseTensor.__tondarray__() instead",stacklevel=2)
    sizes=tuple([])
    
    blocks=[dict() for n in range(tensor._rank)]
    for n in range(tensor._rank):
        s=0
        start=0
        for k in sorted(tensor._Ds[n].keys()):
            blocks[n][k]=tuple([start,start+utils.prod(utils.flatten(tensor._Ds[n][k]))])
            start+=utils.prod(utils.flatten(tensor._Ds[n][k]))
            s+=utils.prod(utils.flatten(tensor._Ds[n][k]))
        sizes+=tuple([s])

    full=np.zeros(sizes,dtype=tensor._dtype)
    for k in tensor.__keys__():
        b=tuple([])
        for n in range(tensor._rank):
            b+=tuple([slice(blocks[n][k[n]][0],blocks[n][k[n]][1],1)])
        full[b]=tensor[k]
    return full


    

#D is a dict() containing key-shape pairs, both of which are possibly nested tuples that define the size of each block;
#D could be e.g. obtained from a tensor by D=tensor._Ds[n]; this would give an identity matrix that could be contracted
#with index "n" of tensor.
#note that the matrix is diagonal. see SparseTensor for an explanation of all other parameters;

def eye(D,qflow,mergelevel,dtype=float):
    Ds=[D.copy(),D.copy()]
    iden=spt.SparseTensor(keys=[],values=[],Ds=Ds,qflow=qflow,mergelevel=mergelevel,dtype=dtype)
    for k in D.keys():
        key=tuple([k,k])
        shape=tuple([D[k],D[k]])
        size=utils.prod(utils.flatten(D[k]))
        iden.__insertmergedblock__(key,shape,np.eye(size).astype(dtype))

    return iden

#vectorizes a block-symmetric tensor; 
#returns: vector of type ndarray containing the blocks of tensor in vectorized form.
#         shapes can be used to retrieve the block symmetric structure in tensorise
def vectorize(tensor):
    #determine the neccessary size:
    shapes={}
    if (len(tensor._shapes)==0):
        tensor.__shapes__()
    size=0
    for k in sorted(tensor.__keys__()):
        shapes[k]=(size,np.shape(tensor[k]))
        size+=utils.prod(tensor[k].shape)
    vec=np.zeros(size,dtype=tensor._dtype)
    for k in sorted(shapes.keys()):
        start=shapes[k][0]
        length=utils.prod(shapes[k][1])
        end=start+length
        vec[start:end]=np.reshape(tensor[k],length)
    return vec,list([shapes,tensor._qflow,tensor._mergelevel,tensor._shapes,tensor._ktoq])
    
#tensorize a vector into "shapes"; shape has to be obtained from "vector,shapes=vectorize(tensor)"
def tensorize(unpackdata,vector):
    shapes=unpackdata[0]
    qflow=unpackdata[1]
    mergelevel=unpackdata[2]
    tensorshapes=unpackdata[3]
    keytoq=unpackdata[4]
    if vector.dtype==np.float64:
        dtype=float
    elif vector.dtype==np.complex128:
        dtype=complex
    #determine the neccessary size:        
    k=list(shapes.keys())[0]
    Ds=[dict() for n in range(len(k))]
    tensor=spt.SparseTensor(keys=[],values=[],Ds=Ds,qflow=qflow,keytoq=keytoq,mergelevel=mergelevel,dtype=dtype)
    #there's no need to sort the keys here
    for k in shapes:
        start=shapes[k][0]
        end=start+utils.prod(shapes[k][1])

        tensor.__insertmergedblock__(k,tensorshapes[k],np.reshape(vector[start:end],shapes[k][1]))
    return tensor


"""
computes the svd decomosition of tensor
routine partitions the indices of the tensor into groupes defined by indices1 and indices2, and then sv-decomposes the resulting object. the user has to make
sure that indices1 and indices2 are such that the resulting object is a matrix, i.e. has rank 2.
then it performs an SVD of this matrix and returns U,S,V, where U and V are isometries,and S is a diagonal matrix containing the Schmidt values.
When computing M=U*S*V, using e.g. the routine sparsenumpy.tensordot, M might have
additional blocks which were not present in the before the SVD. These blocks should all have norm 0.0, and are the result of
sv-decomposing block-matrices of a block-diagonal structure, i.e. mat=[[A,0],[0,B]], where mat has a fixed central U(1) charge q. The user can use M.__squeeze__()
to remove these blocks again.

NOTE ON THE qflow CONVENTION: any tensor has an internal member _qflow which determines which legs are of inflowing and which are of outflowing character.
There is an ambiguity for labelling the new leg after the svd if the indices1 AND indices2 both contain legs of in- AND outflowing character.
on the other hand, if indices1 are all in- or outflowing, then the new leg is either out-or inflowing (and similar for indices2). The routine first checks
if indices1 are all of equal flow-type. If so, the flow of the new leg is determined from that. If not, it checks if indices2 are all of equal flow type. If yes, it determines
the flow of the new leg from them. If not, then there is an ambiguity in the value of _qflow in the new, svd-derived tensor-leg. In fact, the value of _qflow of this new tensor leg
can be chosen arbitrarily. 
"""
def svd(tensor,indices1,indices2,defflow=1):
    inds=sorted(indices1)+sorted(indices2)
    assert(sorted(inds)==list(range(tensor._rank)))
    #transpose the indices of tensor such that  indices1 are the the first (in ascending order) and indices2 are the second (in ascendin

    matrix=merge(tensor,indices1,indices2)

    assert(len(matrix._Ds)==2)

    #both indices are merged ones
    #matrix is now split apart using svd
    #we randomly assign the columns of the resulting matrix inflow character, and the rows outflow character. The total flow has to sum up to zero
    #if the tensor has the correct symmetry.
    #first, according to the chosen combination, find all blocks that have the same total quantum number inflow (and outflow):
    #qndict has as keys the total inflow of U(1) charge; as value, it has a list of tuples (k,sizes) of the keys of all blocks with this U(1) charge inflow and their size
    qndict=dict()
    usedict=(tensor._ktoq!=(list([None])*tensor._rank))

    flatktoq=utils.flatten(matrix._ktoq[0])
    flatflowx=utils.flatten(matrix._qflow[0])
    flatflowy=utils.flatten(matrix._qflow[1])
    
    #if all x-flows are inflowing, the resulting new index has to be outflowing to convserve the charge
    #(vice versa for all-outflowing x)
    if (list(map(np.sign,flatflowx))==([1]*len(flatflowx))) or (list(map(np.sign,flatflowx))==([-1]*len(flatflowx))):
        #print 'x-keys have the same flow direction'
        netflowx=flatflowx[0]
        #computeflowfromq=False
    elif (list(map(np.sign,flatflowy))==([1]*len(flatflowy))) or (list(map(np.sign,flatflowy))==([-1]*len(flatflowy))):
        #print 'y-keys have the same flow direction'            
        netflowx=-flatflowy[0]
        #computeflowfromq=False
    else:
        warnings.warn("in sparsenumpy.svd: flow-structure of merged tensor is non-uniform in the x-component. Flow direction of the new bond cannot be unambigously labelled; using the default value {0}.".format(defflow),stacklevel=2)
        netflowx=defflow
        
    for k in matrix.__keys__():
        #note: matrix has combined keys as row and column keys; they have to be utils.flattened to obtain the total U(1) charge.
        #kflat=[utils.flatten(utils.tupsignmult(k[0],matrix._qflow[0])),utils.flatten(utils.tupsignmult(k[1],matrix._qflow[1]))]
        if not usedict:
            kflat=utils.tupsignmult(utils.flatten(k[0]),utils.flatten(matrix._qflow[0]))
            
            q=sum(kflat[1::],kflat[0])
        elif usedict:
            flatkey=utils.flatten(k[0])
            #sum has to be used in a bit an awkward fashion; the second argument is the first element of the sequence to be added
            if flatktoq[0]==None:
                q=flatkey[0]*flatflowx[0]
            else:
                q=flatktoq[0][flatkey[0]]*flatflowx[0]
                
            for n in range(1,len(flatkey)):
                if flatktoq[n]==None:
                    q+=flatkey[n]*flatflowx[n]
                else:
                    q+=flatktoq[n][flatkey[n]]*flatflowx[n]


        #each leg of the matrix has in this case incoming and outgoing sub-indices; the new central index can then be randomly assigned a
        #flow direction. The covention I use is that if the total charge is positive, then the flow direction is positive as well
        #if (computeflowfromq==True) and (matrix._qsign(q)!=0):
        #    #netflowx has to be the same for all keys
        #    netflowx=matrix._qsign(q)
        #    computeflowfromq=False
        if q not in qndict:
            qndict[q]=[(k,matrix[k].shape)]
        elif q in qndict:
            qndict[q].append((k,matrix[k].shape))
            
    #in the case that there is only one entry in the matrix, and it has q=0, then the flowdirection of the x-leg of the matrix is chosen to be positive:
    #if computeflowfromq==True:
    #    netflowx=1
    #    computeflowfromq=False
            
    #for each key-value pair in qndict, we now can do a seperate svd (this could be parallelized).
    #for a given key "q" below we have a set of N different matrices that together form a large matrix
    #which can be sv-decomposed. We now have to build this huge matrix. First, we go through the list qndict[q]
    #and find all different x and y keys:
    #the total charge flow for every block in matrix has to be the same; take the first block to check what it is:

    UDs=[dict() for n in range(2)]
    VDs=[dict() for n in range(2)]
    SDs=[dict() for n in range(2)]
    Uqflow=tuple([matrix._qflow[0],-netflowx])
    Vqflow=tuple([netflowx,matrix._qflow[1]])
    Sqflow=tuple([netflowx,-netflowx])
    
    mergelevelU=tuple([matrix._mergelevel[0],'A'])
    mergelevelS=tuple(['A','A'])
    mergelevelV=tuple(['A',matrix._mergelevel[1]])

    keytoqU=tuple([matrix._ktoq[0],None])
    keytoqS=tuple([None,None])
    keytoqV=tuple([None,matrix._ktoq[1]])        

    #the flow-sign of the new index depends on wether kflat is positive or negative. If kflat is positive, it means the the net charge inflow i on the x-index of the tensor is
    #positive, and hence the new index has to have outflow character (and vice versa for a net charge outflow the on x-index).

    U=spt.SparseTensor(keys=[],values=[],Ds=UDs,qflow=Uqflow,keytoq=keytoqU,mergelevel=mergelevelU,dtype=tensor._dtype)
    S=spt.SparseTensor(keys=[],values=[],Ds=SDs,qflow=Sqflow,keytoq=keytoqS,mergelevel=mergelevelS,dtype=tensor._dtype)
    V=spt.SparseTensor(keys=[],values=[],Ds=VDs,qflow=Vqflow,keytoq=keytoqV,mergelevel=mergelevelV,dtype=tensor._dtype)

    for q in qndict:
        #note: k[0] is a tuple (a,b), where a is the x-key and b the y-key of a matrix-block with total charge q
        #      k[1] is the shape of this block
        startx=0
        starty=0
        xkeys=dict()
        ykeys=dict()
        for k in qndict[q]:
            #k[0][0] contains the x-key of the block. this key can be a single key-object, or a tuple of key-objects (a nested tuple, i.e. a merged index).
            #in the latter case, the nested structure has to be transferred to the sv-decomposed matrices, U,S,V.
            #the same goes for ykeys
            if k[0][0] not in xkeys:
                xkeys[k[0][0]]=(startx,startx+k[1][0])
                startx+=k[1][0]
            elif k[0][0] in xkeys:
                #this is a consistency check and could in principle be removed
                assert((xkeys[k[0][0]][1]-xkeys[k[0][0]][0])==k[1][0])
            if k[0][1] not in ykeys:
                ykeys[k[0][1]]=(starty,starty+k[1][1])
                starty+=k[1][1]
            elif k[0][1] in ykeys:
                #this is a consistency check and could in principle be removed
                assert((ykeys[k[0][1]][1]-ykeys[k[0][1]][0])==k[1][1])
                
        #now we have the total size, and we can build the matrix
        mat=np.zeros((startx,starty)).astype(tensor._dtype)

        #fill in the blocks
        for k in qndict[q]:
            mat[xkeys[k[0][0]][0]:xkeys[k[0][0]][1],ykeys[k[0][1]][0]:ykeys[k[0][1]][1]]=matrix[k[0]]
        #now for the svd:
        blocku,blocks,blockv=np.linalg.svd(mat,full_matrices=False)
        Skey=tuple([q*(netflowx),q*(netflowx)])        
        S[Skey]=np.diag(blocks)
        Dc=blocku.shape[1]

        for xkey in xkeys:
            Ukey=tuple([xkey,q*netflowx])
            shape=tuple([matrix._Ds[0][xkey],Dc])
            assert(utils.prod(utils.flatten(shape))==utils.prod(blocku[slice(xkeys[xkey][0],xkeys[xkey][1],1),:].shape))            
            U.__insertmergedblock__(Ukey,shape,blocku[slice(xkeys[xkey][0],xkeys[xkey][1],1),:])
        for ykey in ykeys:            
            Vkey=tuple([q*(netflowx),ykey])
            shape=tuple([Dc,matrix._Ds[1][ykey]])
            assert(utils.prod(utils.flatten(shape))==utils.prod(blockv[:,slice(ykeys[ykey][0],ykeys[ykey][1],1)].shape))
            V.__insertmergedblock__(Vkey,shape,blockv[:,slice(ykeys[ykey][0],ykeys[ykey][1],1)])


    Usplit=splitSingle(U,0)
    Vsplit=splitSingle(V,1)
    return Usplit,S,Vsplit


#computes the qr decomosition of tensor
#routine partitions the indices of the tensor into groupes defined by indices1 and indices2, and then sv-decomposes the resulting object. the user has to make
#sure that indices1 and indices2 are such that the resulting object is a matrix, i.e. has rank 2.
#the routine transposes the indices of tensor such that indices1 are the the first and indices2 are the second (the order within indices1 and indices2 remains as is).
#The routine returns Q,R, where Q is an isometries,and R is an upper triangular matrix. When computing M=Q*R, using e.g. the routine sparsenumpy.tensordot, M might have
#additional blocks which were not present in the first place before the QR. These blocks should all have norm 0.0, and are the result of
#qr-decomposing block-matrices of a block-diagonal structure, i.e. mat=[[A,0],[0,B]], where mat has a fixed central U(1) charge q. The user can use M.__squeeze__()
#to remove these blocks again.



def qr(tensor,indices1,indices2,defflow=1):
    inds=sorted(indices1)+sorted(indices2)
    assert(sorted(inds)==list(range(tensor._rank)))
    #transpose the indices of tensor such that  indices1 are the the first (in ascending order) and indices2 are the second (in ascendin
    matrix=merge(tensor,indices1,indices2)
    assert(len(matrix._Ds)==2)

    #both indices are merged ones
    #matrix is now split apart using svd
    #we randomly assign the columns of the resulting matrix inflow character, and the rows outflow character. The total flow has to sum up to zero
    #if the tensor has the correct symmetry.
    #first, according to the chosen combination, find all blocks that have the same total quantum number inflow (and outflow):
    #qndict has as keys the total inflow of U(1) charge; as value, it has a list of tuples (k,sizes) of the keys of all blocks with this U(1) charge inflow and their size
    qndict=dict()
    usedict=(tensor._ktoq!=(list([None])*tensor._rank))

    flatktoq=utils.flatten(matrix._ktoq[0])
    flatflowx=utils.flatten(matrix._qflow[0])
    flatflowy=utils.flatten(matrix._qflow[1])
    
    #if all x-flows are inflowing, the resulting new index has to be outflowing to convserve the charge
    #(vice versa for all-outflowing x)

    if (list(map(np.sign,flatflowx))==([1]*len(flatflowx))) or (list(map(np.sign,flatflowx))==([-1]*len(flatflowx))):
        #print 'x-keys have the same flow direction'
        netflowx=flatflowx[0]
        #computeflowfromq=False
    elif (list(map(np.sign,flatflowy))==([1]*len(flatflowy))) or (list(map(np.sign,flatflowy))==([-1]*len(flatflowy))):
        #print 'y-keys have the same flow direction'            
        netflowx=-flatflowy[0]
        #computeflowfromq=False
    else:
        warnings.warn("in sparsenumpy.qr: flow-structure of merged tensor is non-uniform in the x-component. Flow direction of the new bond cannot be unambigously labelled; using the default value {0}.".format(defflow),stacklevel=2)
        netflowx=defflow
        
    for k in matrix.__keys__():
        #note: matrix has combined keys as row and column keys; they have to be utils.flattened to obtain the total U(1) charge.
        #kflat=[utils.flatten(utils.tupsignmult(k[0],matrix._qflow[0])),utils.flatten(utils.tupsignmult(k[1],matrix._qflow[1]))]
        if not usedict:
            kflat=utils.tupsignmult(utils.flatten(k[0]),utils.flatten(matrix._qflow[0]))
            #sum has to be used in a bit an awkward fashion; the second argument is the first element of the sequence to be added
            q=sum(kflat[1::],kflat[0])
        elif usedict:
            flatkey=utils.flatten(k[0])
            #sum has to be used in a bit an awkward fashion; the second argument is the first element of the sequence to be added
            if flatktoq[0]==None:
                q=flatkey[0]*flatflowx[0]
            else:
                q=flatktoq[0][flatkey[0]]*flatflowx[0]
                
            for n in range(1,len(flatkey)):

                if flatktoq[n]==None:
                    q+=flatkey[n]*flatflowx[n]
                else:
                    q+=flatktoq[n][flatkey[n]]*flatflowx[n]


        #each leg of the matrix has in this case incoming and outgoing sub-indices; the new central index can then be randomly assigned a
        #flow direction. The covention I use is that if the total charge is positive, then the flow direction is positive as well
        if q not in qndict:
            qndict[q]=[(k,matrix[k].shape)]
        elif q in qndict:
            qndict[q].append((k,matrix[k].shape))


    #in the case that there is only one entry in the matrix, and it has q=0, then the flowdirection of the x-leg of the matrix is chosen to be positive
    #if computeflowfromq==True:
    #    netflowx=1
    #    computeflowfromq=False
    
    #for each key-value pair in qndict, we now can do a seperate svd (this could be parallelized).
    #for a given key "q" below we have a set of N different matrices that together form a large matrix
    #which can be sv-decomposed. We now have to build this huge matrix. First, we go through the list qndict[q]
    #and find all different x and y keys:
    #the total charge flow for every block in matrix has to be the same; take the first block to check what it is:

    QDs=[dict() for n in range(2)]
    RDs=[dict() for n in range(2)]

    Qqflow=tuple([matrix._qflow[0],-netflowx])
    Rqflow=tuple([netflowx,matrix._qflow[1]])
    
    mergelevelQ=tuple([matrix._mergelevel[0],'A'])
    mergelevelR=tuple(['A',matrix._mergelevel[1]])


    keytoqQ=tuple([matrix._ktoq[0],None])
    keytoqR=tuple([None,matrix._ktoq[1]])        
    
    #the flow-sign of the new index depends on wether kflat is positive or negative. If kflat is positive, it means the the net charge inflow i on the x-index of the tensor is
    #positive, and hence the new index has to have outflow character (and vice versa for a net charge outflow the on x-index).
    Q=spt.SparseTensor(keys=[],values=[],Ds=QDs,qflow=Qqflow,keytoq=keytoqQ,mergelevel=mergelevelQ,dtype=tensor._dtype)
    R=spt.SparseTensor(keys=[],values=[],Ds=RDs,qflow=Rqflow,keytoq=keytoqR,mergelevel=mergelevelR,dtype=tensor._dtype)
    for q in qndict:
        startx=0
        starty=0
        xkeys=dict()
        ykeys=dict()
        for k in qndict[q]:
            #note: k[0] is a tuple (a,b), where a is the x-key and b the y-key of a matrix-block with total charge q
            #      k[1] is the shape of this block
            #k[0][0] contains the x-key of the block. this key can be a single key-object, or a tuple of key-objects (a nested tuple, i.e. a merged index).
            #in the latter case, the nested structure has to be transferred to the sv-decomposed matrices, U,S,V.
            #the same goes for ykeys
            if k[0][0] not in xkeys:
                xkeys[k[0][0]]=(startx,startx+k[1][0])
                startx+=k[1][0]
            elif k[0][0] in xkeys:
                #this is a consistency check and could in principle be removed
                assert((xkeys[k[0][0]][1]-xkeys[k[0][0]][0])==k[1][0])
                
            if k[0][1] not in ykeys:
                ykeys[k[0][1]]=(starty,starty+k[1][1])
                starty+=k[1][1]
            elif k[0][1] in ykeys:
                #this is a consistency check and could in principle be removed
                assert((ykeys[k[0][1]][1]-ykeys[k[0][1]][0])==k[1][1])
                
        #now we have the total size, and we can build the matrix
        mat=np.zeros((startx,starty)).astype(tensor._dtype)

        #fill in the blocks
        for k in qndict[q]:
            mat[xkeys[k[0][0]][0]:xkeys[k[0][0]][1],ykeys[k[0][1]][0]:ykeys[k[0][1]][1]]=matrix[k[0]]

        #now for the svd:
        blockq,blockr=np.linalg.qr(mat)
        Dc=blockq.shape[1]

        for xkey in xkeys:
            Qkey=tuple([xkey,q*netflowx])
            #Qkey=tuple([xkey,abs(q)])            
            shape=tuple([matrix._Ds[0][xkey],Dc])
            assert(utils.prod(utils.flatten(shape))==utils.prod(blockq[slice(xkeys[xkey][0],xkeys[xkey][1],1),:].shape))            
            Q.__insertmergedblock__(Qkey,shape,blockq[slice(xkeys[xkey][0],xkeys[xkey][1],1),:])
        for ykey in ykeys:            
            #Rkey=tuple([abs(q),ykey])
            Rkey=tuple([q*netflowx,ykey])            
            shape=tuple([Dc,matrix._Ds[1][ykey]])
            assert(utils.prod(utils.flatten(shape))==utils.prod(blockr[:,slice(ykeys[ykey][0],ykeys[ykey][1],1)].shape))
            R.__insertmergedblock__(Rkey,shape,blockr[:,slice(ykeys[ykey][0],ykeys[ykey][1],1)])
    Qsplit=splitSingle(Q,0)
    Rsplit=splitSingle(R,1)    
    return Qsplit,Rsplit

    
def isdiag(mat):
    assert(len(mat._Ds)==2)
    for k in mat.__keys__():
        if k[0]!=k[1]:
            return False
    return True



#returns the norm of the diagonal of diagmat
def normdiag(diagmat):
    dtype=diagmat._dtype
    Z=dtype(0.0)
    for k in diagmat.__keys__():
        if(k[0]!=k[1]):
            continue
        vec=np.diag(diagmat[k])
        Z+=np.conj(vec).dot(vec)
    return np.sqrt(Z)

#returns elementwise sqrt of tensor
def sqrt(tensor):
    return tensor.__sqrt__()
def conj(tensor):
    return tensor.__conj__()
def real(tensor):
    return tensor.__real__()
def imag(tensor):
    return tensor.__imag__()
def zeros(tensor):
    zero=tensor.__copy__()
    zero.__zero__()    
    return zero
def trace(matrix):
    assert(len(matrix._Ds==2))
    t=matrix.dtype(0.0)
    for k in matrix.__keys__():
        if k[0]==k[1]:
            t+=np.trace(matrix[k])
    return t
def transpose(tensor,newinds):
    return tensor.__transpose__(newinds)






"""
computes the contraction of tensor1 with tensor2 over the indices inds. 
The mergelevel of the tensors is respected and returned accordingly, i.e.
the resulting tensor can be split over the uncontracted indices if they were merged indices in the first place.
"""
def tensordot(tensor1,tensor2,inds,ignore_qflow=False):
    #assert(tensor1._qsign==tensor2._qsign)
    if not ignore_qflow:
        for l in range(len(inds[0])):
            if tensor1._qflow[inds[0][l]]!=tensor2.__flippedflow__()[inds[1][l]]:
                warnings.warn('sparsenumpy.tensordot(tensor1,tensor2,inds): tensor1._qflow[{0}]==tensor2._qflow[{1}]'.format(inds[0][l],inds[1][l]),stacklevel=2)
    assert(tensor1._dtype==tensor2._dtype)
    t2=tensor2#.__copy__()
    ind1=inds[0]
    ind2=inds[1]
    cind1=sorted(list(set(range(len(tensor1._Ds))).difference(set(ind1))))
    cind2=sorted(list(set(range(len(tensor2._Ds))).difference(set(ind2))))

    assert(len(ind1)==len(ind2))
    Ds=[]
    qflow=tuple([])
    mergelevel=tuple([])
    keytoq=tuple([])
    for i1 in cind1:
        Ds.append(dict())
        qflow+=tuple([tensor1._qflow[i1]])
        mergelevel+=tuple([tensor1._mergelevel[i1]])
        keytoq+=tuple([tensor1._ktoq[i1]])        
    for i2 in cind2:
        Ds.append(dict())
        qflow+=tuple([tensor2._qflow[i2]])
        mergelevel+=tuple([tensor2._mergelevel[i2]])
        keytoq+=tuple([tensor2._ktoq[i2]])        
        
    result=spt.SparseTensor([],[],Ds,qflow,keytoq=keytoq,mergelevel=mergelevel,dtype=tensor1._dtype)
    for k1 in tensor1.__keys__():
        k1reduced=[k1[m] for m in ind1]
        #for k2 in t2._tensor.keys():
        if k1[ind1[0]] in t2._keys[ind2[0]]:
            for k2 in t2._keys[ind2[0]][k1[ind1[0]]]:
                k2reduced=[k2[n] for n in ind2]
                if k1reduced!=k2reduced:
                    continue
            
                newkey=tuple([k1[m] for m in cind1]+[k2[m] for m in cind2])
                newshape=tuple([tensor1._Ds[m][k1[m]] for m in cind1]+[t2._Ds[m][k2[m]] for m in cind2])            
                if newkey in result._tensor:
                    result._tensor[newkey]+=np.tensordot(tensor1[k1],t2[k2],inds)
                elif newkey not in result._tensor:
                    #try:
                    result.__insertmergedblock__(newkey,newshape,np.tensordot(tensor1[k1],t2[k2],inds))
                    #except ValueError:
                    #    print (tensor1[k1].shape,t2[k2].shape)
                    #    input(k1)

            
    return result


#return result

"""
merge a list of index-lists into a single index; the indices to be merged have to be increasing, consecutive numbers, e.g. indices=[[1,2,3],[5,6],[8,9]]
The function copies the data of tensor, i.e. modifying merged does not modify tensor
"""
def merge(tensor,*indices):
    merged=tensor.__copy__()
    cnt=0
    if len(indices[0])>0:
        if (indices[0]!=list(range(indices[0][0],indices[0][-1]+1))):
            sys.exit('sparsenumpy.merge(tensor,*indices): cannot merge {0}: it is not a list of consecutive, increasing numbers'.format(indices[0]))        
        merged=mergeSingle(merged,indices[0])
        cnt=len(indices[0])-1
    for n in range(1,len(indices)):
        inds=indices[n]
        if len(inds)>0:        
            if (inds!=list(range(inds[0],inds[-1]+1))):
                sys.exit('sparsenumpy.merge(tensor,*indices): cannot merge {0}: it is not a list of consecutive, increasing numbers'.format(inds))        
            merged=mergeSingle(merged,list(range(inds[0]-cnt,inds[-1]+1-cnt)))
            cnt+=(len(inds)-1)
    return merged

"""
merge a list "inds" into a single index; inds have to be consecutive numbers, either increasing or decreasing;
note that "inds" HAS to be a list. 
The function does not copy the tensor-blocks; it returns a new object with containing the identical tensors (i.e. the same memory)
as tensor; i.e. modifying result also modifies tensor!
"""
def mergeSingle(tensor,inds):
    if not isinstance(inds,list):
        inds=list(inds)
    if len(inds)==1:
        return tensor
    try: 
        if len(inds)>tensor._rank:
            raise TensorSizeError
        assert((inds==list(range(inds[0],inds[-1]+1))) or (inds==list(range(inds[0],inds[-1]-1,-1))))

        if (inds==list(range(inds[0],inds[-1]+1))):
            Ds=[]
            qflow=tuple([])
            keytoq=tuple([])
            mergelevel=tuple([])
            for n in range(inds[0]):
                mergelevel+=tuple([tensor._mergelevel[n]])
                qflow+=tuple([tensor._qflow[n]])
                keytoq+=tuple([tensor._ktoq[n]])
                Ds.append(dict())
                
            Ds.append(dict())
            locqflow=tuple([])
            lockeytoq=tuple([])
            tempmergelevel=tuple([])
            for n in inds:
                tempmergelevel+=tuple([tensor._mergelevel[n]])                
                locqflow+=tuple([tensor._qflow[n]])
                lockeytoq+=tuple([tensor._ktoq[n]])

            mergelevel+=tuple([tempmergelevel])                
            qflow+=tuple([locqflow])
            keytoq+=tuple([lockeytoq])
            for n in range(inds[-1]+1,tensor._rank):
                mergelevel+=tuple([tensor._mergelevel[n]])                
                qflow+=tuple([tensor._qflow[n]])
                keytoq+=tuple([tensor._ktoq[n]])
                Ds.append(dict())
                

                
            result=spt.SparseTensor([],[],Ds,qflow,keytoq=keytoq,mergelevel=mergelevel,dtype=tensor._dtype)

            for k in tensor.__keys__():
                newkey=tuple([])
                for n in range(0,inds[0]):
                    newkey+=tuple([k[n]])
                tempkey=tuple([])
                for m in inds:
                    tempkey+=tuple([k[m]])
                newkey+=tuple([tempkey])
                for n in range(inds[-1]+1,tensor._rank):
                    newkey+=tuple([k[n]])
                
                oldshape=tensor[k].shape
                combshape=tuple([])
                for n in inds:
                    combshape+=tuple([tensor._Ds[n][k[n]]])

                newshape=tuple(oldshape[0:inds[0]])+tuple([utils.prod(utils.flatten(combshape))])+tuple(oldshape[inds[-1]+1::])
                result._tensor[newkey]=np.reshape(tensor[k],newshape)
                result.__addkey__(newkey)
                for n in range(inds[0]):
                    result._Ds[n]=dict(tensor._Ds[n])
                result._Ds[inds[0]][newkey[inds[0]]]=combshape
                for n in range(inds[-1]+1,len(oldshape)):
                    result._Ds[inds[0]+n-inds[-1]]=dict(tensor._Ds[n])

            #this refreshes the ._shapes member of result, a dict() containing key-shape (both are nested tuples) pairs for each block that has key as keyvalue.
            result.__shapes__()
            return result


        if (inds==list(range(inds[0],inds[-1]-1,-1))):
            sys.exit('sparsenumpy.mergeSingle(tensor,indices): indices is not a list of consecutive, increasing numbers')
    except AssertionError:
        print ('ERROR in sparsennumpy.merge: inds are not consecutive numbers')
    except TensorSizeError:
        print ('ERROR in sparsenumpy.merge: rank of tensor is smaller than the number of indices to be merged (tensor._rank<len(inds))')
        sys.exit()

#runs through all inds and splits them by one level
def split(tensor,inds):
    num=0
    for ind in sorted(inds):
        index=ind+num        
        assert(tensor._mergelevel[index]!='A')
        num+=len(tensor._mergelevel[index])-1
        tensor=splitSingle(tensor,index)
    return tensor

#splits the index "index" of the tensor into its constituent indices.
#Note that combined keys are ALWAYS next to each other (due to the way they have to be merged in the function "merge" above)
def splitSingle(tensor,index):
    if tensor._mergelevel[index]=='A':
        return tensor.__copy__()
    if tensor._mergelevel[index]!='A':
        #check if combkey really is a combined key, i.e. it has to be a tuple()
        if(tensor._mergelevel[index]=='A'):
            raise TensorKeyError
        Ds=[]
        for n in range(tensor._rank+len(tensor._mergelevel[index])-1):
            Ds.append(dict())

        #define the quantum number flow for the split tensor
        qflow=tuple([])
        mergelevel=tuple([])
        keytoq=tuple([])
        for n in range(0,index):
            qflow+=tuple([tensor._qflow[n]])
            mergelevel+=tuple([tensor._mergelevel[n]])
            keytoq+=tuple([tensor._ktoq[n]])            
        for n in range(len(tensor._mergelevel[index])):
            qflow+=tuple([tensor._qflow[index][n]])
            mergelevel+=tuple([tensor._mergelevel[index][n]])
            keytoq+=tuple([tensor._ktoq[index][n]])
            
        for n in range(index+1,tensor._rank):
            qflow+=tuple([tensor._qflow[n]])
            mergelevel+=tuple([tensor._mergelevel[n]])
            keytoq+=tuple([tensor._ktoq[n]])            

        result=spt.SparseTensor(keys=[],values=[],Ds=Ds,qflow=qflow,keytoq=keytoq,mergelevel=mergelevel,dtype=tensor._dtype)

        for key in tensor.__keys__():
            #for each key k, the entry at "index" is now split up into its constituents
            newkey=tuple([])
            for n in range(0,index):
                newkey+=tuple([key[n]])
            for n in range(len(key[index])):
                newkey+=tuple([key[index][n]])
            for n in range(index+1,tensor._rank):
                newkey+=tuple([key[n]])

            
            oldshape=tensor[key].shape
            oldshape_combined=tensor._Ds[index][key[index]]
            newshape=tuple([])
            newshape_combined=tuple([])            
            for n in range(index):
                newshape+=tuple([oldshape[n]])
                newshape_combined+=tuple([tensor._Ds[n][key[n]]])

            newshape_combined+=oldshape_combined            
            for n in range(len(oldshape_combined)):
                newshape+=tuple([utils.prod(utils.flatten(oldshape_combined[n]))])

            for n in range(index+1,len(oldshape)):
                newshape+=tuple([oldshape[n]])
                newshape_combined+=tuple([tensor._Ds[n][key[n]]])                
            result.__insertmergedblock__(newkey,newshape_combined,np.reshape(tensor[key],newshape))
        return result

#take an SV-decomposed tensor U,S,V and truncate all eigenvalues smaller than eps from S
#normalizes S
#the routine keeps at least the largest eigevalue in S
def truncate(U,S,V,eps):
    #first normalize S
    S/=S.__norm__()
    mx=S.__blockmax__()
    maxkey=list(mx.keys())[0]
    maxval=mx[maxkey]
    for k in mx.keys():
        if mx[k]>maxval:
            maxkey=k
            maxval=mx[k]
    
    assert(S._rank==2)
    assert(S.__dim__(0)==S.__dim__(1))
    remove=set()
    keys=list(S._tensor.keys())
    for k in keys:
        diag=np.diag(S[k])
        dold=np.copy(diag)
        diag=diag[np.nonzero(diag>=eps)]
        S.__remove__(k)
        if len(diag)>0:
            mat=np.diag(diag)
            S[k]=mat
        else:
            remove.add(k[0])
    shapes=dict()
    for k in S.__keys__():
        shapes[k]=S[k].shape


    #now we need to truncate U and V
    if len(S._tensor.keys())==0:
        mat=np.ones((1,1)).astype(S._dtype)
        S[maxkey]=mat
    S/=S.__norm__()        

    keys=list(U._tensor.keys())
    for uk in keys:
        if uk[-1] in remove and uk[-1]!=maxkey[0]:
            U.__remove__(uk)

    keys=list(V._tensor.keys())
    for vk in keys:
        if vk[0] in remove and vk[0]!=maxkey[1]:
            V.__remove__(vk)

    for sk in shapes:
        #get a list of all blocks with a quantum number sk on the last index (that's the index that has to be connected with S)
        for uk in U._keys[-1][sk[0]]:
            mat=U[uk]
            b=[]
            for n in range(len(mat.shape)-1):
                b.append(slice(0,mat.shape[n],1))
            b.append(slice(0,shapes[sk][0]))
            slices=tuple(b)
            U._Ds[-1][uk[-1]]=shapes[sk][0]
            U._tensor[uk]=mat[slices]

        for vk in V._keys[0][sk[1]]:
            mat=V[vk]
            b=[]
            b.append(slice(0,shapes[sk][0]))            
            for n in range(1,len(mat.shape)):
                b.append(slice(0,mat.shape[n],1))
            slices=tuple(b)
            V._Ds[0][vk[0]]=shapes[sk][1]
            V._tensor[vk]=mat[slices]

    

