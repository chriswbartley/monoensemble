#cython: boundscheck=False
#cython: cdivision=True
#cython: wraparound=False
#cython: nonecheck=False
# Author: Chris Bartley
#
# License: BSD 3 clause

# NOTE: for some strange reason all the arrays coming from numpy are order='F'
# (Fortran) COLUMN major rather than row major. I've gone with it.
cimport cython

from libc.stdlib cimport free
from libc.string cimport memset
from libcpp cimport bool
import numpy as np
cimport numpy as np
np.import_array()

from scipy.sparse import issparse
from scipy.sparse import csr_matrix

ctypedef np.npy_float32 DTYPE_t          # Type of X
ctypedef np.npy_float64 DOUBLE_t         # Type of y, sample_weight
ctypedef np.npy_intp SIZE_t              # Type for indices and counters
ctypedef np.npy_int32 INT32_t            # Signed 32 bit integer
ctypedef np.npy_uint32 UINT32_t          # Unsigned 32 bit integer

ctypedef np.int32_t int32
ctypedef np.float64_t float64
ctypedef np.float_t float
ctypedef np.uint8_t uint8

# no namespace lookup for numpy dtype and array creation
from numpy import zeros as np_zeros
from numpy import ones as np_ones
from numpy import bool as np_bool
from numpy import float32 as np_float32
from numpy import float64 as np_float64
from libc.math cimport log, exp

# constant to mark tree leafs
cdef SIZE_t TREE_LEAF = -1
cdef float64 RULE_LOWER_CONST=-1e9
cdef float64 RULE_UPPER_CONST=1e9
cdef int32 MIN_NODE_SIZE_FOR_SORTING_=5
cdef _log_logistic_sigmoid_(int32 n_samples, 
                            float64 *X,
                            float64 * out):
    cdef int32 i=0
    for i in range(n_samples):
        if X[i]>0.:
            out[i]=-log(1.+exp(-X[i]))
        else:
            out[i]=X[i]-log(1.+exp(X[i]))
    return


def _log_logistic_sigmoid(int32 n_samples,  
                           np.ndarray[DOUBLE_t, ndim=1] X):
    out=np.empty_like(X)
    _log_logistic_sigmoid_( n_samples, 
                            <float64*> (<np.ndarray> X).data  ,
                            <float64*> (<np.ndarray> out).data  
                            )
    return out

cdef _custom_dot_(float64 *X,
                  float64 *w,
                  int32 n_rows, 
                  int32 n_cols, 
                  float64 *out):
    cdef int32 i=0
    cdef int32 j=0
    cdef float64 ttl=0.
    for i in range(n_rows):
        ttl=0.
        for j in range(n_cols):
            ttl=ttl+X[i*n_cols+j]*w[j]
        out[i]=ttl
    return

def _custom_dot(np.ndarray[float64, ndim=2] X,
                np.ndarray[float64, ndim=1] w):
    out=np.zeros(X.shape[0],dtype=np.float64)
    _custom_dot_(
                            <float64*> (<np.ndarray> X).data  ,
                            <float64*> (<np.ndarray> w).data  ,
                            X.shape[0],
                            X.shape[1],
                            <float64*> (<np.ndarray> out).data  
                            )
    return out

cdef _custom_dot_multiply_(float64 *X,
                  float64 *w,
                  float64 *y,
                  float64 c,
                  int32 n_rows, 
                  int32 n_cols, 
                  float64 *out):
    cdef int32 i=0
    cdef int32 j=0
    cdef float64 ttl=0.
    for i in range(n_rows):
        ttl=0.
        for j in range(n_cols):
            ttl=ttl+X[i*n_cols+j]*w[j]
        out[i]=y[i]*(ttl+c)
    return

def _custom_dot_multiply(np.ndarray[float64, ndim=2] X,
                np.ndarray[float64, ndim=1] w,
                np.ndarray[float64, ndim=1] y,
                float64 c):
    out=np.zeros(X.shape[0],dtype=np.float64)
    _custom_dot_multiply_(
                            <float64*> (<np.ndarray> X).data  ,
                            <float64*> (<np.ndarray> w).data  ,
                            <float64*> (<np.ndarray> y).data  ,
                            c,
                            X.shape[0],
                            X.shape[1],
                            <float64*> (<np.ndarray> out).data  
                            )
    return out


cdef void _apply_rules_with_map_and_feat_cache(float64 *X,
                       float64 *rule_lower_corners,
                           float64 *rule_upper_corners,
                           int32 n_samples,
                          int32 n_features,
                          int32 n_rules,
                          int32 *X_leaf_node_ids,
                          int32 *node_rule_map,
                          int32 *out_rule_feats_upper,
                          int32 *out_rule_feats_lower,
                          int32 *out):
    """   """
    cdef int32 res
    cdef int32 rule_start
    cdef int32 rule_end
    cdef int32 i
    cdef int32 j
    cdef int32 r
    cdef int32 j_test
    cdef int32 leaf_id
    cdef int32 base_rule_id
    cdef int32 i_r
    cdef int32 cont
    cdef int32 f_to_check
    cdef int32 if_to_check
    for i in range(n_samples):
        leaf_id=X_leaf_node_ids[i]
        base_rule_id=node_rule_map[leaf_id * n_rules] # first column gives base leaf, this is the corresponding rule
        out[i * n_rules + base_rule_id]=1 
        i_r=1
        if i_r>=n_rules:
            cont=0
        else:
            cont=1 if node_rule_map[leaf_id * n_rules +i_r]!=-99 else 0
        while cont==1:
            r=node_rule_map[leaf_id * n_rules +i_r]
            res=1 
            # check lower rules
            if_to_check=0
            j=out_rule_feats_lower[leaf_id * n_rules * n_features + r*n_features + if_to_check]
            while j!=-99 and res!=0:
                if X[j * n_samples + i] <= rule_lower_corners[j * n_rules +  r]:
                    res=0
                if_to_check=if_to_check+1
                j=out_rule_feats_lower[leaf_id * n_rules * n_features + r*n_features + if_to_check]
            if res==1:     
                # check upper rules
                if_to_check=0
                j=out_rule_feats_upper[leaf_id * n_rules * n_features + r*n_features + if_to_check]
                while j!=-99 and res!=0:
                    if X[j * n_samples + i] > rule_upper_corners[j * n_rules +  r]:
                        res=0
                    if_to_check=if_to_check+1
                    j=out_rule_feats_upper[leaf_id * n_rules * n_features + r*n_features + if_to_check]
            out[i * n_rules + r]=res 
            i_r=i_r+1
            if i_r>=n_rules:
                cont=0
            else:
                cont=1 if node_rule_map[leaf_id * n_rules +i_r]!=-99 else 0
            


cdef void _apply_rules_with_map_sparse(float64 *X,
                       object rule_lower_corners,
                           object rule_upper_corners,
                           int32 n_samples,
                          int32 n_features,
                          int32 n_rules,
                          int32 *X_leaf_node_ids,
                          int32 *node_rule_map,
                          int32 *out):
    """   """

    cdef float64* lower_data = <float64*>(<np.ndarray> rule_lower_corners.data).data
    cdef INT32_t* lower_indices = <INT32_t*>(<np.ndarray> rule_lower_corners.indices).data
    cdef INT32_t* lower_indptr = <INT32_t*>(<np.ndarray> rule_lower_corners.indptr).data
    cdef float64* upper_data = <float64*>(<np.ndarray> rule_upper_corners.data).data
    cdef INT32_t* upper_indices = <INT32_t*>(<np.ndarray> rule_upper_corners.indices).data
    cdef INT32_t* upper_indptr = <INT32_t*>(<np.ndarray> rule_upper_corners.indptr).data
    cdef int32 res
    cdef int32 rule_start
    cdef int32 rule_end
    cdef int32 i
    cdef int32 j
    cdef int32 r
    cdef int32 j_test
    cdef int32 leaf_id
    cdef int32 base_rule_id
    cdef int32 i_r
    cdef int32 cont

    for i in range(n_samples):
        leaf_id=X_leaf_node_ids[i]
        base_rule_id=node_rule_map[leaf_id * n_rules] # first column gives base leaf, this is the corresponding rule
        out[i * n_rules + base_rule_id]=1 
        i_r=1
        if i_r>=n_rules:
            cont=0
        else:
            cont=1 if node_rule_map[leaf_id * n_rules +i_r]!=-99 else 0
        while cont==1:
            r=node_rule_map[leaf_id * n_rules +i_r]
            # check lower rules
            rule_start=lower_indptr[r]
            rule_end=lower_indptr[r+1]
            res=1 
            for j_test in range(rule_start,rule_end):
                j= lower_indices[j_test]
                if X[j * n_samples + i] <= (lower_data[j_test]+RULE_LOWER_CONST):
                    res=0
            if res==1:     
                rule_start=upper_indptr[r]
                rule_end=upper_indptr[r+1]
                for j_test in range(rule_start,rule_end):
                    j= upper_indices[j_test]
                    if X[j * n_samples + i] > (upper_data[j_test]+RULE_UPPER_CONST):
                        res=0                
            out[i * n_rules + r]=res 
            i_r=i_r+1
            if i_r>=n_rules:
                cont=0
            else:
                cont=1 if node_rule_map[leaf_id * n_rules +i_r]!=-99 else 0

cdef int32 _search_sorted(float64 *arr, int32 arr_start_idx,int32 arr_len,int32 stride, float64 val):
    cdef int32 first
    cdef int32 last 
    cdef int32 found
    cdef int32 midpoint
    first = 0
    last=arr_len-1
    found = 0
    # right sided by default
    while first<=last and  found !=1:
        midpoint = (first + last)//2
        if midpoint<(arr_len-1) :
            if arr[arr_start_idx+midpoint*stride] <= val and arr[arr_start_idx+(midpoint+1)*stride] > val:
                found=1
        else: # midpt at the end
            if arr[arr_start_idx+midpoint*stride] <= val:
                found=1
        # otherwise move mid point
        if val < arr[arr_start_idx+midpoint*stride]:
            last = midpoint-1
        else:
            first = midpoint+1
            
    return midpoint+1

@cython.boundscheck(False)       
cdef void _update_sorted_datapoint_posns(
                           float64 *sorted_feats,
                          int32 *sorted_indxs,
                          int32 *sorted_datapoint_posns,
                          int32 n_samples,
                          int32 n_features):
    cdef int32 i
    cdef int32 k
    cdef int32 j
    for j in range(n_features):
        i=0
        for k in range(n_samples):
            sorted_datapoint_posns[sorted_indxs[k+j*n_samples]+j*n_samples]=i
            i=i+1 

@cython.boundscheck(False) 
cdef _traverse_node_c(int32 node_id,
                    int32 num_feats,
                       int32 *children_left,
                       int32 *children_right,
                       int32 *features,
                       float64 *thresholds, 
                       int32 *out_leaf_ids,
                       float64 *out_rule_upper_corners,
                       float64 *out_rule_lower_corners):
    cdef int32 feature
    cdef float64 threshold
    cdef int32 left_node_id
    cdef int32 right_node_id
    cdef int32 j
    # recurse on children 
    if children_left[node_id] != -1: #TREE_LEAF:
        feature = features[node_id]
        threshold = thresholds[node_id]
        left_node_id = children_left[node_id]
        right_node_id = children_right[node_id]
        # update limit arrays
        for j in range(num_feats):
            out_rule_upper_corners[left_node_id*num_feats+j] = out_rule_upper_corners[node_id*num_feats+j]
            out_rule_lower_corners[left_node_id*num_feats+j] = out_rule_lower_corners[node_id*num_feats+j]
            out_rule_upper_corners[ right_node_id*num_feats+j] = out_rule_upper_corners[node_id*num_feats+j]
            out_rule_lower_corners[right_node_id*num_feats+j] = out_rule_lower_corners[node_id*num_feats+j]
        out_rule_upper_corners[left_node_id*num_feats+feature] = threshold
        out_rule_lower_corners[right_node_id*num_feats+feature] = threshold
        # recurse
        _traverse_node_c(left_node_id, num_feats,
                       children_left,children_right,features,thresholds, out_leaf_ids,out_rule_upper_corners,out_rule_lower_corners)  # "<="
        _traverse_node_c(right_node_id,num_feats,
                       children_left,children_right,features,thresholds, out_leaf_ids,out_rule_upper_corners,out_rule_lower_corners)  # ">"
    else:  # a leaf node
        out_leaf_ids[node_id] = node_id
        if node_id == 0:# the base node (0) is the only node!
            pass
            #print('Warning: Tree only has one node! (i.e. the root node)')

cdef float64 _calc_deriv(float64 *y,
               float64 *f,
               float64 *sample_weight,
               int32 n):
    cdef float64 ttl=0.
    cdef int32 i =0
    for i in range(n):
        if sample_weight[i]>0:
            ttl=ttl-2*sample_weight[i]*y[i]/(exp(2*y[i]*f[i])+1)
    return ttl

cdef float64 _calc_deriv_2(float64 *y,
               float64 *f,
               float64 *sample_weight,
               int32 n):
    cdef float64 ttl=0.
    cdef int32 i =0
    for i in range(n):
        if sample_weight[i]>0:
            ttl=ttl+sample_weight[i]*4*y[i]**2*exp(2*y[i]*f[i])/(1+exp(2*y[i]*f[i]))**2
    return ttl

def calc_newton_step_c(np.ndarray[float64, ndim=1] y,
                     np.ndarray[float64, ndim=1] f,
                     np.ndarray[float64, ndim=1] sample_weight):
    d=_calc_deriv(<float64*> (<np.ndarray> y).data,
                  <float64*> (<np.ndarray> f).data,
                  <float64*> (<np.ndarray> sample_weight).data,
                  len(y))
    d2=_calc_deriv_2(<float64*> (<np.ndarray> y).data,
                  <float64*> (<np.ndarray> f).data,
                  <float64*> (<np.ndarray> sample_weight).data,
                  len(y))
    return d/d2


def extract_rules_from_tree_c(np.ndarray[int32, ndim=1] children_left,
                              np.ndarray[int32, ndim=1] children_right,
                              np.ndarray[int32, ndim=1] features,
                              np.ndarray[float64, ndim=1] thresholds, 
                              int32 num_feats, 
                              np.ndarray[int32, ndim=1] out_leaf_ids,
                              np.ndarray[float64, ndim=2] out_rule_upper_corners,
                              np.ndarray[float64, ndim=2] out_rule_lower_corners):
    _traverse_node_c(np.int32(0),
                     num_feats,
                     <int32*> (<np.ndarray> children_left).data ,
                     <int32*> (<np.ndarray> children_right).data ,
                     <int32*> (<np.ndarray> features).data ,
                     <float64*> (<np.ndarray> thresholds).data,
                     <int32*> (<np.ndarray> out_leaf_ids).data ,
                     <float64*> (<np.ndarray> out_rule_upper_corners).data,
                     <float64*> (<np.ndarray> out_rule_lower_corners).data
                     )
  
            
@cython.boundscheck(False)       
cdef void _apply_rules_set_based(float64 *X,
                       float64 *rule_lower_corners,
                           float64 *rule_upper_corners,
                           float64 *sorted_feats,
                          int32 *sorted_indxs,
                          int32 *sorted_datapoint_posns,
                           Py_ssize_t n_samples,
                          Py_ssize_t n_features,
                          Py_ssize_t n_rules,
                          int32 *out):
    """   """
    cdef int32 res
    cdef int32 rule_start
    cdef int32 rule_end
    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef Py_ssize_t r
    cdef int32 j_test
    cdef int32 i_f
    cdef int32 i_ff
    cdef int32 insert_pos
    cdef int32 dirn
    cdef np.ndarray[np.int32_t, ndim=1] viable_set = np.empty(n_samples, dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=1] feat_sets=np.zeros([n_features*2*4],dtype=np.int32)
    cdef int32 viable_set_size
    cdef int32 viable_set_size_this
    cdef int32 i_viable
    cdef int32 min_viable_size
    cdef int32  min_viable_index
    
    # apply each rule
    for r in range(n_rules):
        i_f=0
        for j in range(n_features):
            if rule_lower_corners[j * n_rules + r]!=RULE_LOWER_CONST: 
                insert_pos=_search_sorted(sorted_feats,j*n_samples, n_samples,1,rule_lower_corners[j * n_rules + r]) 
                feat_sets[0*2*n_features+ i_f]=j
                feat_sets[1*2*n_features+ i_f]=-1
                feat_sets[2*2*n_features+ i_f]=insert_pos
                feat_sets[3*2*n_features+ i_f]=n_samples-insert_pos
                i_f=i_f+1
            if rule_upper_corners[j * n_rules + r]!=RULE_UPPER_CONST: 
                insert_pos=_search_sorted(sorted_feats,j*n_samples, n_samples,1,rule_upper_corners[j * n_rules + r])
                feat_sets[0*2*n_features+ i_f]=j
                feat_sets[1*2*n_features+ i_f]=1
                feat_sets[2*2*n_features+ i_f]=insert_pos
                feat_sets[3*2*n_features+ i_f]=insert_pos
                i_f=i_f+1
        if i_f==0:
            for i in range(n_samples):
                out[r +  i*n_rules]=1 
        else:
            min_viable_size=100000
            min_viable_index=-1
            for i_ff in range(i_f):
                if feat_sets[3*2*n_features+ i_ff]<min_viable_size:
                    min_viable_size=feat_sets[3*2*n_features+ i_ff]
                    min_viable_index=i_ff
            i_ff=min_viable_index # start with minimum because the size of the first set is an upper bound on complexity for all subsequent interscetion operations
            j=feat_sets[0*2*n_features+ i_ff]
            insert_pos=feat_sets[2*2*n_features+ i_ff]
            dirn=feat_sets[1*2*n_features+ i_ff]
            viable_set_size=feat_sets[3*2*n_features+ i_ff]
            if viable_set_size>0:
                if dirn==-1:
                    for i in range(viable_set_size):
                        viable_set[i]=sorted_indxs[j*n_samples + (insert_pos+i)  ] 
                else:
                    for i in range(viable_set_size):
                        viable_set[i]=sorted_indxs[j*n_samples + (i) ] 
                    
                for i_ff in range(0,i_f):
                    if i_ff !=min_viable_index  and viable_set_size>0:
                        j=feat_sets[0*2*n_features+ i_ff]
                        insert_pos=feat_sets[2*2*n_features+ i_ff]
                        dirn=feat_sets[1*2*n_features+ i_ff]
                        viable_set_size_this=feat_sets[3*2*n_features+ i_ff]
                        if dirn==-1:
                            i_viable=0
                            for i in range(viable_set_size):
                                if  sorted_datapoint_posns[viable_set[i] + j*n_samples]>=insert_pos: 
                                    viable_set[i_viable]=viable_set[i]
                                    i_viable=i_viable+1
                            viable_set_size=i_viable
                        else:
                            i_viable=0
                            for i in range(viable_set_size):
                                if  sorted_datapoint_posns[viable_set[i] + j*n_samples ]<insert_pos: 
                                    viable_set[i_viable]=viable_set[i]
                                    i_viable=i_viable+1
                            viable_set_size=i_viable
            
                if viable_set_size>0:
                    for i in range(viable_set_size) :
                        out[viable_set[i]*n_rules + r]=1 
                    
cdef _traverse_node_with_rule_sorted_c(int32 node_id,
                       int32 num_feats,
                       int32 num_rules,
                       int32 num_samples,
                       int32 *children_left,
                       int32 *children_right,
                       int32 *features,
                       float64 *thresholds, 
                       int32 *node_members,
                       int32 *node_members_count,
                       int32 *node_members_start,
                       int32 rule_id,
                       float64 *rule_upper_corners,
                       float64 *rule_lower_corners,
                       int32 *rule_upper_feats_engaged,
                       int32 rule_upper_feats_engaged_count,
                       int32 *rule_lower_feats_engaged,
                       int32 rule_lower_feats_engaged_count,
                       float64 *X,
                       float64 *X_by_node_sorted,
                       int32 *X_by_node_sorted_idx,
                       int32 *X_by_node_sorted_idx_posns, 
                       int32 *out_rule_mask):
    cdef int32 feature
    cdef float64 threshold
    cdef int32 left_node_id
    cdef int32 right_node_id
    cdef int32 n_samples_in_node
    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef Py_ssize_t r
    cdef int32 j_
    cdef int32 j_test
    cdef int32 i_f
    cdef int32 i_ff
    cdef int32 insert_pos
    cdef int32 dirn
    cdef int32 num_pts
    cdef np.ndarray[np.int32_t, ndim=1] viable_set 
    cdef np.ndarray[np.int32_t, ndim=1] feat_sets 
    cdef float64 rule_lower_corners1[200]# num_feats
    cdef float64 rule_upper_corners1[200]# num_feats
    cdef float64 lower_bound=0.
    cdef float64 upper_bound=0.
    cdef int32 viable_set_size
    cdef int32 viable_set_size_this
    cdef int32 i_viable
    cdef int32 min_viable_size
    cdef int32  min_viable_index
    # recurse on children 
    if children_left[node_id] != -1:
        feature = features[node_id]
        threshold = thresholds[node_id]
        left_node_id = children_left[node_id]
        right_node_id = children_right[node_id]
        # check if rule goes right
        if rule_upper_corners[feature]>threshold:
            for j in range(num_feats):
                rule_upper_corners1[j]=rule_upper_corners[j]
                rule_lower_corners1[j]=rule_lower_corners[j]
            #rule_upper_changed=0
            if rule_lower_corners1[feature]<=threshold: # rule lower bound no longer needed
                rule_lower_corners1[feature]=RULE_LOWER_CONST
                rule_upper_changed=1
            _traverse_node_with_rule_sorted_c(right_node_id, num_feats,num_rules,num_samples,
                   children_left,children_right,features,thresholds, node_members,node_members_count,node_members_start,rule_id,
                   rule_upper_corners1,
                   rule_lower_corners1,
                   rule_upper_feats_engaged,
                   rule_upper_feats_engaged_count,
                   rule_lower_feats_engaged,
                   rule_lower_feats_engaged_count,
                   X,
                   X_by_node_sorted,
                   X_by_node_sorted_idx,
                   X_by_node_sorted_idx_posns,
                   out_rule_mask)  
        # check if rule goes left
        if rule_lower_corners[feature]<threshold:

            for j in range(num_feats):
                rule_upper_corners1[j]=rule_upper_corners[j]
                rule_lower_corners1[j]=rule_lower_corners[j]
            if rule_upper_corners1[feature]>=threshold: # rule lower bound no longer needed
                rule_upper_corners1[feature]=RULE_UPPER_CONST
            _traverse_node_with_rule_sorted_c(left_node_id, num_feats,num_rules,num_samples,
                   children_left,children_right,features,thresholds, node_members,node_members_count,node_members_start,rule_id,
                   rule_upper_corners1,
                   rule_lower_corners1,
                   rule_upper_feats_engaged,
                   rule_upper_feats_engaged_count,
                   rule_lower_feats_engaged,
                   rule_lower_feats_engaged_count,
                   X,
                   X_by_node_sorted,
                   X_by_node_sorted_idx,
                   X_by_node_sorted_idx_posns,
                   out_rule_mask)  
    else:  # a leaf node - check remaining rule
        n_samples_in_node=node_members_count[node_id]
        if n_samples_in_node>=MIN_NODE_SIZE_FOR_SORTING_:
            # SORTED FEATURE WAY OF DOING IT - 
            viable_set = np.empty(n_samples_in_node, dtype=np.int32)
            feat_sets=np.zeros([num_feats*2*4],dtype=np.int32)
            # apply each feature limit and store set descriptors
            i_f=0
            for j_ in range(rule_lower_feats_engaged_count): 
                j=rule_lower_feats_engaged[j_]
                if rule_lower_corners[j ]!=RULE_LOWER_CONST: 
                    insert_pos=_search_sorted(X_by_node_sorted,node_members_start[node_id]*num_feats, node_members_count[node_id],num_feats,rule_lower_corners[j ]) #, side='right'
                    feat_sets[0*2*num_feats+ i_f]=j
                    feat_sets[1*2*num_feats+ i_f]=-1
                    feat_sets[2*2*num_feats+ i_f]=insert_pos
                    feat_sets[3*2*num_feats+ i_f]=node_members_count[node_id]-insert_pos
                    i_f=i_f+1
            for j_ in range(rule_upper_feats_engaged_count): 
                j=rule_upper_feats_engaged[j_]
                if rule_upper_corners[j ]!=RULE_UPPER_CONST: 
                    insert_pos=_search_sorted(X_by_node_sorted,node_members_start[node_id]*num_feats, node_members_count[node_id],num_feats,rule_upper_corners[j ]) #, side='right'
                    feat_sets[0*2*num_feats+ i_f]=j
                    feat_sets[1*2*num_feats+ i_f]=+1
                    feat_sets[2*2*num_feats+ i_f]=insert_pos
                    feat_sets[3*2*num_feats+ i_f]=insert_pos
                    i_f=i_f+1
            
            if i_f==0: # if no rules found, add all node members (shortcut exit)
                for i in range(n_samples_in_node) :
                    out_rule_mask[node_members[node_id*num_samples+i]*num_rules + rule_id]=1 
            else: # check for intersections:
                # intersect sets to build viable set
                min_viable_size=100000
                min_viable_index=-1
                min_viable_feat=-1
                for i_ff in range(i_f):
                    if feat_sets[3*2*num_feats+ i_ff]<min_viable_size:
                        min_viable_size=feat_sets[3*2*num_feats+ i_ff]
                        min_viable_index=i_ff
                        min_viable_feat=feat_sets[0*2*num_feats+ i_ff]
                i_ff=min_viable_index # start with minimum because the size of the first set is an upper bound on complexity for all subsequent interscetion operations
                j=feat_sets[0*2*num_feats+ i_ff]
                insert_pos=feat_sets[2*2*num_feats+ i_ff]
                dirn=feat_sets[1*2*num_feats+ i_ff]
                viable_set_size=feat_sets[3*2*num_feats+ i_ff]
                if viable_set_size>0:
                    if dirn==-1:
                        for i in range(viable_set_size):
                            viable_set[i]=X_by_node_sorted_idx[(node_members_start[node_id]+insert_pos+i)*num_feats+min_viable_feat ]
                    else:
                        for i in range(viable_set_size):
                            viable_set[i]=X_by_node_sorted_idx[(node_members_start[node_id]+i)*num_feats+min_viable_feat ]
                        
                    for i_ff in range(0,i_f):
                        if i_ff !=min_viable_index and viable_set_size>0:
                            j=feat_sets[0*2*num_feats+ i_ff]
                            insert_pos=feat_sets[2*2*num_feats+ i_ff]
                            dirn=feat_sets[1*2*num_feats+ i_ff]
                            viable_set_size_this=feat_sets[3*2*num_feats+ i_ff]
                            if dirn==-1:
                                i_viable=0
                                for i in range(viable_set_size):
                                    if  X_by_node_sorted_idx_posns[viable_set[i]*num_feats+j]-node_members_start[node_id]>=insert_pos: # + j*n_samples]>=insert_pos: # viable_set[i]*n_features + j 
                                        viable_set[i_viable]=viable_set[i]
                                        i_viable=i_viable+1
                                viable_set_size=i_viable
                            else:
                                i_viable=0
                                for i in range(viable_set_size):
                                    if  X_by_node_sorted_idx_posns[viable_set[i]*num_feats+j]-node_members_start[node_id]<insert_pos: #+ j*n_samples ]<insert_pos: # viable_set[i] + j*n_samples
                                        viable_set[i_viable]=viable_set[i]
                                        i_viable=i_viable+1
                                viable_set_size=i_viable
                
                if viable_set_size>0:
                    for i in range(viable_set_size) :
                        out_rule_mask[viable_set[i]*num_rules+ rule_id]=1 
        else:
            # BASIC WAY OF CALCULATING
            num_pts=node_members_count[node_id]
            for i in range(num_pts):
                out_rule_mask[node_members[node_id*num_samples+i]*num_rules+rule_id]=1
            for j_ in range(rule_lower_feats_engaged_count):
                j=rule_lower_feats_engaged[j_]
                lower_bound=rule_lower_corners[j]
                if lower_bound!=RULE_LOWER_CONST:
                    for i in range(num_pts):
                        if X[node_members[node_id*num_samples+i]*num_feats+j]<=lower_bound:
                            out_rule_mask[node_members[node_id*num_samples+i]*num_rules+rule_id]=0
            for j_ in range(rule_upper_feats_engaged_count): 
                j=rule_upper_feats_engaged[j_]
                upper_bound=rule_upper_corners[j]
                if upper_bound!=RULE_UPPER_CONST:
                    for i in range(num_pts):
                        if X[node_members[node_id*num_samples+i]*num_feats+j]>upper_bound:
                            out_rule_mask[node_members[node_id*num_samples+i]*num_rules+rule_id]=0

def apply_rules_from_tree_sorted_c(np.ndarray[float64, ndim=2] X,
                            np.ndarray[float64, ndim=2] X_by_node_sorted,
                            np.ndarray[int32, ndim=2] X_by_node_sorted_idx,
                            np.ndarray[int32, ndim=2] X_by_node_sorted_idx_posns,
                            np.ndarray[int32, ndim=1] children_left,
                            np.ndarray[int32, ndim=1] children_right,
                            np.ndarray[int32, ndim=1] features,
                            np.ndarray[float64, ndim=1] thresholds,
                            np.ndarray[int32, ndim=2] node_members,
                            np.ndarray[int32, ndim=1] node_members_count, 
                            np.ndarray[int32, ndim=1] node_members_start, 
                            int32 num_feats,
                            np.ndarray[float64, ndim=2] rule_upper_corners,
                            np.ndarray[float64, ndim=2] rule_lower_corners,
                            np.ndarray[int32, ndim=2] out_rule_mask):
    
    cdef np.ndarray[int32, ndim=1] rule_upper_feats_engaged = np.zeros([num_feats],dtype=np.int32,order='C')
    cdef np.ndarray[int32, ndim=1] rule_lower_feats_engaged = np.zeros([num_feats],dtype=np.int32,order='C')
    cdef int32 rule_id
    cdef int32 j
    cdef int32 j_
    cdef int32 upper_feats_cnt
    cdef int32 lower_feats_cnt
    cdef int32 num_rules =rule_upper_corners.shape[0]
    for rule_id in range(num_rules): 
        j_=0
        for j in range(num_feats):
            if rule_upper_corners[rule_id,j]!=RULE_UPPER_CONST:
                rule_upper_feats_engaged[j_]=j
                j_=j_+1
        upper_feats_cnt=j_
        j_=0
        for j in range(num_feats):
            if rule_lower_corners[rule_id,j]!=RULE_LOWER_CONST:
                rule_lower_feats_engaged[j_]=j
                j_=j_+1
        lower_feats_cnt=j_
        _traverse_node_with_rule_sorted_c(<int32> 0,
                       <int32> num_feats,
                       <int32> num_rules,
                       <int32> X.shape[0],
                       <int32*> (<np.ndarray> children_left).data ,
                       <int32*> (<np.ndarray> children_right).data ,
                       <int32*> (<np.ndarray> features).data ,
                       <float64*> (<np.ndarray> thresholds).data , 
                       <int32*> (<np.ndarray> node_members).data ,
                       <int32*> (<np.ndarray> node_members_count).data ,
                       <int32*> (<np.ndarray> node_members_start).data ,
                       <int32> rule_id,
                       <float64*> (<np.ndarray> rule_upper_corners[rule_id,:].copy()).data ,
                       <float64*> (<np.ndarray> rule_lower_corners[rule_id,:].copy()).data ,
                       <int32*> (<np.ndarray> rule_upper_feats_engaged).data ,
                       <int32> upper_feats_cnt,
                       <int32*> (<np.ndarray> rule_lower_feats_engaged).data ,
                       <int32> lower_feats_cnt,
                       <float64*> (<np.ndarray> X).data ,
                       <float64*> (<np.ndarray> X_by_node_sorted).data ,
                       <int32*> (<np.ndarray> X_by_node_sorted_idx).data ,
                       <int32*> (<np.ndarray> X_by_node_sorted_idx_posns).data ,
                       <int32*> (<np.ndarray> out_rule_mask).data )
    

cdef void _apply_rules_sparse(float64 *X,
                       object rule_lower_corners,
                           object rule_upper_corners,
                           Py_ssize_t n_samples,
                          Py_ssize_t n_features,
                          Py_ssize_t n_rules,
                          int32 *out):
    """   """
    cdef float64* lower_data = <float64*>(<np.ndarray> rule_lower_corners.data).data
    cdef INT32_t* lower_indices = <INT32_t*>(<np.ndarray> rule_lower_corners.indices).data
    cdef INT32_t* lower_indptr = <INT32_t*>(<np.ndarray> rule_lower_corners.indptr).data
    cdef float64* upper_data = <float64*>(<np.ndarray> rule_upper_corners.data).data
    cdef INT32_t* upper_indices = <INT32_t*>(<np.ndarray> rule_upper_corners.indices).data
    cdef INT32_t* upper_indptr = <INT32_t*>(<np.ndarray> rule_upper_corners.indptr).data
    cdef int32 res
    cdef int32 rule_start
    cdef int32 rule_end
    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef Py_ssize_t r
    cdef int32 j_test
    for i in range(n_samples):
        for r in range(n_rules):
            # check lower rules
            rule_start=lower_indptr[r]
            rule_end=lower_indptr[r+1]
            res=1 
            for j_test in range(rule_start,rule_end):
                j= lower_indices[j_test]
                if X[j * n_samples + i] <= (lower_data[j_test]+RULE_LOWER_CONST):
                    res=0
            if res==1:     
                rule_start=upper_indptr[r]
                rule_end=upper_indptr[r+1]
                for j_test in range(rule_start,rule_end):
                    j= upper_indices[j_test]
                    if X[j * n_samples + i] > (upper_data[j_test]+RULE_UPPER_CONST):
                        res=0                
            out[i * n_rules + r]=res 


cdef void _get_node_map_and_rules_sparse(int32 *leaf_ids, 
                          float64 *leaf_values,
                          float64 *leaf_lower_corners,
                          float64 *leaf_upper_corners,
                          object rule_lower_corners,
                          object rule_upper_corners,
                          Py_ssize_t n_leaves,
                          Py_ssize_t n_features,
                          Py_ssize_t n_rules,
                          int32 *out,
                          int32 *out_rule_feats_upper,
                          int32 *out_rule_feats_lower):
    """   """
    cdef float64* lower_data = <float64*>(<np.ndarray> rule_lower_corners.data).data
    cdef INT32_t* lower_indices = <INT32_t*>(<np.ndarray> rule_lower_corners.indices).data
    cdef INT32_t* lower_indptr = <INT32_t*>(<np.ndarray> rule_lower_corners.indptr).data
    cdef float64* upper_data = <float64*>(<np.ndarray> rule_upper_corners.data).data
    cdef INT32_t* upper_indices = <INT32_t*>(<np.ndarray> rule_upper_corners.indices).data
    cdef INT32_t* upper_indptr = <INT32_t*>(<np.ndarray> rule_upper_corners.indptr).data
    cdef int32 res
    cdef int32 rule_start
    cdef int32 rule_end
    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef Py_ssize_t r
    cdef int32 j_test
    cdef int32 leaf_id =0
    cdef int32 r_to_add
    cdef int32 f_upper_to_add
    cdef int32 f_lower_to_add
    for i in range(n_leaves):
        leaf_id=leaf_ids[i]
        out[leaf_id * n_rules +0]=i # base rule always applies
        r_to_add=1
        for r in range(n_rules):
            if r!=i:
                # check lower rules
                rule_start=lower_indptr[r]
                rule_end=lower_indptr[r+1]
                res=1 
                for j_test in range(rule_start,rule_end):
                    j= lower_indices[j_test]
                    if leaf_upper_corners[j * n_leaves + i] <= (lower_data[j_test]+RULE_LOWER_CONST):
                        res=0                        
                if res==1:     
                    rule_start=upper_indptr[r]
                    rule_end=upper_indptr[r+1]
                    for j_test in range(rule_start,rule_end):
                        j= upper_indices[j_test]
                        if leaf_lower_corners[j * n_leaves + i] >= (upper_data[j_test]+RULE_UPPER_CONST):
                            res=0   
                if res==1: # rule does overlap
                    out[leaf_id * n_rules + r_to_add]=r 
                    r_to_add=r_to_add+1
                    # now examine rule features to see which are reqd to assess
                    rule_start=upper_indptr[r]
                    rule_end=upper_indptr[r+1]
                    f_upper_to_add=0
                    for j_test in range(rule_start,rule_end):
                        j= upper_indices[j_test]
                        if leaf_upper_corners[j * n_leaves + i] > (upper_data[j_test]+RULE_UPPER_CONST):
                            out_rule_feats_upper[leaf_id * n_rules * n_features + r*n_features+f_upper_to_add]=j
                            f_upper_to_add=f_upper_to_add+1
                    rule_start=lower_indptr[r]
                    rule_end=lower_indptr[r+1]
                    f_lower_to_add=0
                    for j_test in range(rule_start,rule_end):
                        j= lower_indices[j_test]
                        if leaf_lower_corners[j * n_leaves + i] < (lower_data[j_test]+RULE_LOWER_CONST):
                            out_rule_feats_lower[leaf_id * n_rules * n_features + r*n_features + f_lower_to_add]=j
                            f_lower_to_add=f_lower_to_add+1                    


cdef void _get_node_map_sparse(int32 *leaf_ids, 
                          float64 *leaf_values,
                          float64 *leaf_lower_corners,
                          float64 *leaf_upper_corners,
                          object rule_lower_corners,
                          object rule_upper_corners,
                          Py_ssize_t n_leaves,
                          Py_ssize_t n_features,
                          Py_ssize_t n_rules,
                          int32 *out):
    """   """
    cdef float64* lower_data = <float64*>(<np.ndarray> rule_lower_corners.data).data
    cdef INT32_t* lower_indices = <INT32_t*>(<np.ndarray> rule_lower_corners.indices).data
    cdef INT32_t* lower_indptr = <INT32_t*>(<np.ndarray> rule_lower_corners.indptr).data
    cdef float64* upper_data = <float64*>(<np.ndarray> rule_upper_corners.data).data
    cdef INT32_t* upper_indices = <INT32_t*>(<np.ndarray> rule_upper_corners.indices).data
    cdef INT32_t* upper_indptr = <INT32_t*>(<np.ndarray> rule_upper_corners.indptr).data
    cdef int32 res
    cdef int32 rule_start
    cdef int32 rule_end
    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef Py_ssize_t r
    cdef int32 j_test
    cdef int32 leaf_id =0
    cdef int32 r_to_add =0
    for i in range(n_leaves):
        leaf_id=leaf_ids[i]
        out[leaf_id * n_rules +0]=i # base rule always applies
        r_to_add=1
        for r in range(n_rules):
            if r!=i:
                # check lower rules
                rule_start=lower_indptr[r]
                rule_end=lower_indptr[r+1]
                res=1 
                for j_test in range(rule_start,rule_end):
                    j= lower_indices[j_test]
                    if leaf_upper_corners[j * n_leaves + i] <= (lower_data[j_test]+RULE_LOWER_CONST):
                        res=0                        
                if res==1:     
                    rule_start=upper_indptr[r]
                    rule_end=upper_indptr[r+1]
                    for j_test in range(rule_start,rule_end):
                        j= upper_indices[j_test]
                        if leaf_lower_corners[j * n_leaves + i] >= (upper_data[j_test]+RULE_UPPER_CONST):
                            res=0   
                if res==1:
                    out[leaf_id * n_rules + r_to_add]=r 
                    r_to_add=r_to_add+1

         
@cython.boundscheck(False)
cdef void _apply_rules(float64 *X,
                       float64 *rule_lower_corners,
                           float64 *rule_upper_corners,
                           Py_ssize_t n_samples,
                          Py_ssize_t n_features,
                          Py_ssize_t n_rules,
                          int32 *out):
    cdef int32 res 
    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef Py_ssize_t r
    for i in range(n_samples):
        for r in range(n_rules):
            res=1
            j=0
            while res==1 and j<n_features:
                if X[j * n_samples + i] > rule_upper_corners[j * n_rules + r]:
                    res=0
                if X[j * n_samples + i] <= rule_lower_corners[j * n_rules +  r]:
                    res=0
                j=j+1
            out[i * n_rules + r]=res 



def get_node_map_and_rule_feats_c(np.ndarray[int32, ndim=1] leaf_ids, 
                      np.ndarray[float64, ndim=1] leaf_values,
                      np.ndarray[float64, ndim=2] leaf_lower_corners,
                      np.ndarray[float64, ndim=2] leaf_upper_corners,
                      object rule_lower_corners, 
                      object rule_upper_corners,
                      int32 n_rules,
                      np.ndarray[int32, ndim=2]  out,
                      np.ndarray[int32, ndim=3]  out_rules_upper,
                      np.ndarray[int32, ndim=3]  out_rules_lower):
    _get_node_map_and_rules_sparse(
         <int32*> (<np.ndarray> leaf_ids).data, 
         <float64*> (<np.ndarray> leaf_values).data, 
         <float64*> (<np.ndarray> leaf_lower_corners).data,
         <float64*> (<np.ndarray> leaf_upper_corners).data,
         rule_lower_corners, 
         rule_upper_corners,
         leaf_lower_corners.shape[0],
         leaf_lower_corners.shape[1],
         n_rules, 
         <int32*> (<np.ndarray> out).data,
         <int32*> (<np.ndarray> out_rules_upper).data,
         <int32*> (<np.ndarray> out_rules_lower).data)
            
def get_node_map_c   (np.ndarray[int32, ndim=1] leaf_ids, 
                      np.ndarray[float64, ndim=1] leaf_values,
                      np.ndarray[float64, ndim=2] leaf_lower_corners,
                      np.ndarray[float64, ndim=2] leaf_upper_corners,
                      object rule_lower_corners, 
                      object rule_upper_corners,
                      np.ndarray[int32, ndim=2]  out):
        if issparse(rule_lower_corners):
            _get_node_map_sparse(
                 <int32*> (<np.ndarray> leaf_ids).data, 
                 <float64*> (<np.ndarray> leaf_values).data, 
                 <float64*> (<np.ndarray> leaf_lower_corners).data,
                 <float64*> (<np.ndarray> leaf_upper_corners).data,
                 rule_lower_corners, 
                 rule_upper_corners,
                 leaf_lower_corners.shape[0],
                 leaf_lower_corners.shape[1],
                 rule_lower_corners.shape[0],
                 <int32*> (<np.ndarray> out).data)

@cython.boundscheck(False) 
cdef _update_rule_coefs(int32 *rule_mask,
                      float64 *y_pred,
                      int32 *y,
                      float64 *sample_weight,
                      float64 lidstone_alpha,
                      Py_ssize_t n_samples,
                      Py_ssize_t n_rules,
                      float64 *out):
    cdef int32 res
    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef Py_ssize_t r
    cdef float64 sum_swt_one
    cdef float64 sum_swt_pred
    cdef float64 sum_swt_ttl
    cdef float64 prob1
    cdef float64 prob1_pred
    
    for r in range(n_rules):
        sum_swt_one=0.
        sum_swt_pred=0.
        sum_swt_ttl=0.
        for i in range(n_samples):
            if rule_mask[i * n_rules + r]==1:
                sum_swt_ttl=sum_swt_ttl+sample_weight[i]
                if y[i]==1:
                    sum_swt_one=sum_swt_one+sample_weight[i]
                sum_swt_pred=sum_swt_pred+sample_weight[i]/(1.+exp(-y_pred[i]))
        prob1=(sum_swt_one+lidstone_alpha)/(sum_swt_ttl+2*lidstone_alpha)
        prob1_pred=(sum_swt_pred+lidstone_alpha)/(sum_swt_ttl+2*lidstone_alpha)
        out[r]   =  log(prob1*(1-prob1_pred)/((1-prob1)*prob1_pred)) 


    
def update_rule_coefs(object rule_mask,
                      object  y_pred,
                      object y,
                      object sample_weight,
                      object lidstone_alpha,
                      np.ndarray[float64, ndim=1] out):
    _update_rule_coefs(<int32*> (<np.ndarray> rule_mask).data,
           <float64*> (<np.ndarray> y_pred).data,
           <int32*> (<np.ndarray> y).data,
           <float64*> (<np.ndarray> sample_weight).data,
           <float64> lidstone_alpha,
           rule_mask.shape[0],
           rule_mask.shape[1],
           <float64*> (<np.ndarray> out).data)

@cython.boundscheck(False) 
cdef _update_rule_coefs_newton_step(int32 *rule_mask,
                      float64 *residual,
                      int32 *y,
                      float64 *sample_weight,
                      Py_ssize_t n_samples,
                      Py_ssize_t n_rules,
                      float64 *out):
    cdef int32 res
    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef Py_ssize_t r
    cdef float64 sum_numerator
    cdef float64 sum_denominator
    cdef float64 y_
    cdef float64 coef_

    for r in range(n_rules):
        sum_numerator=0.
        sum_denominator=0.
        for i in range(n_samples):
            if rule_mask[i * n_rules + r]==1:
                sum_numerator=sum_numerator+sample_weight[i]*residual[i]
                y_=<float64>(y[i])
                sum_denominator=sum_denominator+sample_weight[i]*(y_-residual[i])*(1.0-y_+residual[i])
        if abs(sum_denominator) < 1e-150:
            coef_ = 0.0
        else:
            coef_ = sum_numerator / sum_denominator
        out[r]   =   coef_ 
                    
def update_rule_coefs_newton_step(object rule_mask,
                      object  residual,
                      object y,
                      object sample_weight,
                      np.ndarray[float64, ndim=1] out):
    _update_rule_coefs_newton_step(<int32*> (<np.ndarray> rule_mask).data,
           <float64*> (<np.ndarray> residual).data,
           <int32*> (<np.ndarray> y).data,
           <float64*> (<np.ndarray> sample_weight).data,
           rule_mask.shape[0],
           rule_mask.shape[1],
           <float64*> (<np.ndarray> out).data)
            
def apply_rules_rule_feat_cache_c(np.ndarray[float64, ndim=2] X,object rule_lower_corners, object rule_upper_corners,
                   object  X_leaf_node_ids,
                   object node_rule_map,
                   object node_rule_feat_upper,
                   object node_rule_feat_lower,
                   np.ndarray[int32, ndim=2] out):
    _apply_rules_with_map_and_feat_cache(<float64*> (<np.ndarray> X).data, 
                  <float64*> (<np.ndarray> rule_lower_corners).data, 
                  <float64*> (<np.ndarray> rule_upper_corners).data,
                 <int32> X.shape[0],
                 <int32> X.shape[1],
                 <int32> rule_lower_corners.shape[0],
                 <int32*> (<np.ndarray> X_leaf_node_ids).data ,
                 <int32*> (<np.ndarray> node_rule_map).data ,
                 <int32*> (<np.ndarray> node_rule_feat_upper).data ,
                 <int32*> (<np.ndarray> node_rule_feat_lower).data ,
                 <int32*> (<np.ndarray> out).data)
def apply_rules_c(np.ndarray[float64, ndim=2] X,object rule_lower_corners, object rule_upper_corners,
                   object  X_leaf_node_ids,
                   object node_rule_map,
                   np.ndarray[int32, ndim=2] out):
    if issparse(rule_lower_corners):
        if node_rule_map is None:
            _apply_rules_sparse(<float64*> (<np.ndarray> X).data, 
                  rule_lower_corners, 
                  rule_upper_corners,
                 X.shape[0],
                 X.shape[1],
                 rule_lower_corners.shape[0],
                 <int32*> (<np.ndarray> out).data)
        else:
            _apply_rules_with_map_sparse(<float64*> (<np.ndarray> X).data, 
                  rule_lower_corners, 
                  rule_upper_corners,
                 <int32> X.shape[0],
                 <int32> X.shape[1],
                 <int32> rule_lower_corners.shape[0],
                 <int32*> (<np.ndarray> X_leaf_node_ids).data ,
                 <int32*> (<np.ndarray> node_rule_map).data ,
                 <int32*> (<np.ndarray> out).data)
    else:
        _apply_rules(<float64*> (<np.ndarray> X).data, 
                 <float64*> (<np.ndarray> rule_lower_corners).data, 
                 <float64*> (<np.ndarray> rule_upper_corners).data,
                 X.shape[0],
                 X.shape[1],
                 rule_lower_corners.shape[0],
                 <int32*> (<np.ndarray> out).data)

def apply_rules_set_based_c(np.ndarray[float64, ndim=2] X,
                np.ndarray[float64, ndim=2] sorted_feats,
                np.ndarray[int32, ndim=2] sorted_indxs,
                np.ndarray[int32, ndim=2] sorted_datapoint_posns,
                object rule_lower_corners,
                object rule_upper_corners,
                np.ndarray[int32, ndim=2] out):

    if issparse(rule_lower_corners):
        pass # DENSE NOT IMPLEMENTED

    else:
        _update_sorted_datapoint_posns(
                           <float64*> (<np.ndarray> sorted_feats).data,
                          <int32*> (<np.ndarray> sorted_indxs).data ,
                          <int32*> (<np.ndarray> sorted_datapoint_posns).data ,
                         X.shape[0],
                         X.shape[1])
        _apply_rules_set_based(<float64*> (<np.ndarray> X).data, 
              <float64*> (<np.ndarray> rule_lower_corners).data , 
              <float64*> (<np.ndarray> rule_upper_corners).data ,
              <float64*> (<np.ndarray> sorted_feats).data,
              <int32*> (<np.ndarray> sorted_indxs).data ,
              <int32*> (<np.ndarray> sorted_datapoint_posns).data ,
             X.shape[0],
             X.shape[1],
             rule_lower_corners.shape[0],
             <int32*> (<np.ndarray> out).data)        
    
    
   

def _random_sample_mask(np.npy_intp n_total_samples,
                        np.npy_intp n_total_in_bag, random_state):
     """Create a random sample mask where ``n_total_in_bag`` elements are set.

     Parameters
     ----------
     n_total_samples : int
         The length of the resulting mask.

     n_total_in_bag : int
         The number of elements in the sample mask which are set to 1.

     random_state : np.RandomState
         A numpy ``RandomState`` object.

     Returns
     -------
     sample_mask : np.ndarray, shape=[n_total_samples]
         An ndarray where ``n_total_in_bag`` elements are set to ``True``
         the others are ``False``.
     """
     cdef np.ndarray[float64, ndim=1, mode="c"] rand = \
          random_state.rand(n_total_samples)
     cdef np.ndarray[uint8, ndim=1, mode="c", cast=True] sample_mask = \
          np_zeros((n_total_samples,), dtype=np_bool)

     cdef np.npy_intp n_bagged = 0
     cdef np.npy_intp i = 0

     for i in range(n_total_samples):
         if rand[i] * (n_total_samples - i) < (n_total_in_bag - n_bagged):
             sample_mask[i] = 1
             n_bagged += 1

     return sample_mask
    
    


