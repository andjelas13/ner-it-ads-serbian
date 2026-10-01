"""Leakage-safe direct CSR builder; avoids DictVectorizer peak allocations."""
from collections import Counter
import numpy as np
from scipy.sparse import csr_matrix

def vocabulary(features, min_df=1):
    counts=Counter(k for d in features for k,v in d.items() if v)
    names=sorted(k for k,n in counts.items() if n>=min_df)
    return {name:i for i,name in enumerate(names)}

def to_csr(features, vocab):
    # First pass permits exact NumPy preallocation; no list of (row,col,value).
    row_nnz=np.fromiter((sum(1 for k,v in d.items() if v and k in vocab) for d in features),dtype=np.int64,count=len(features))
    indptr=np.empty(len(features)+1,dtype=np.int64); indptr[0]=0; np.cumsum(row_nnz,out=indptr[1:])
    total=int(indptr[-1]); indices=np.empty(total,dtype=np.int32); data=np.empty(total,dtype=np.float32); pos=0
    for d in features:
        for k,v in d.items():
            if v and k in vocab:
                indices[pos]=vocab[k]; data[pos]=float(v) if isinstance(v,(int,float)) else 1.; pos+=1
    return csr_matrix((data,indices,indptr),shape=(len(features),len(vocab)),dtype=np.float32)
