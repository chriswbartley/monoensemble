# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

# Authors: Gilles Louppe <g.louppe@gmail.com>
#          Peter Prettenhofer <peter.prettenhofer@gmail.com>
#          Brian Holt <bdholt1@gmail.com>
#          Noel Dawe <noel@dawe.me>
#          Satrajit Gosh <satrajit.ghosh@gmail.com>
#          Lars Buitinck
#          Arnaud Joly <arnaud.v.joly@gmail.com>
#          Joel Nothman <joel.nothman@gmail.com>
#          Fares Hedayati <fares.hedayati@gmail.com>
#          Jacob Schreiber <jmschreiber91@gmail.com>
#
# License: BSD 3 clause

#from _criterion cimport Criterion

from libc.stdlib cimport free
from libc.stdlib cimport qsort
from libc.string cimport memcpy
from libc.string cimport memset
from libc.stdlib cimport malloc
from libc.stdlib cimport realloc
from libc.math cimport log as ln

import numpy as np
cimport numpy as np
np.import_array()

from scipy.sparse import csc_matrix

#from _utils cimport log
#from _utils cimport rand_int
#from _utils cimport rand_uniform
#from _utils cimport RAND_R_MAX
#from _utils cimport safe_realloc

cdef double INFINITY = np.inf

ctypedef np.npy_float32 DTYPE_t          # Type of X
ctypedef np.npy_float64 DOUBLE_t         # Type of y, sample_weight
ctypedef np.npy_intp SIZE_t              # Type for indices and counters
ctypedef np.npy_int32 INT32_t            # Signed 32 bit integer
ctypedef np.npy_uint32 UINT32_t          # Unsigned 32 bit integer

# A record stored in the WeightedPQueue
cdef struct WeightedPQueueRecord:
    DOUBLE_t data
    DOUBLE_t weight 
    
cdef struct Node:
    # Base storage structure for the nodes in a Tree object

    SIZE_t left_child                    # id of the left child of the node
    SIZE_t right_child                   # id of the right child of the node
    SIZE_t feature                       # Feature used for splitting the node
    DOUBLE_t threshold                   # Threshold value at the node
    DOUBLE_t impurity                    # Impurity of the node (i.e., the value of the criterion)
    SIZE_t n_node_samples                # Number of samples at the node
    DOUBLE_t weighted_n_node_samples     # Weighted number of samples at the node

cdef struct SplitRecord:
    # Data to track sample split
    SIZE_t feature         # Which feature to split on.
    SIZE_t pos             # Split samples array at the given position,
                           # i.e. count of samples below threshold for feature.
                           # pos is >= end if the node is a leaf.
    double threshold       # Threshold to split at.
    double improvement     # Impurity improvement given parent node.
    double impurity_left   # Impurity of the left split.
    double impurity_right  # Impurity of the right split.

cdef struct Cell:
    # Base storage stucture for cells in a QuadTree object

    # Tree structure
    SIZE_t parent              # Parent cell of this cell
    SIZE_t[8] children         # Array pointing to childrens of this cell
    
    # Cell description
    SIZE_t cell_id             # Id of the cell in the cells array in the Tree
    SIZE_t point_index         # Index of the point at this cell (only defined
                               # in non empty leaf)
    bint is_leaf               # Does this cell have children?
    DTYPE_t squared_max_width  # Squared value of the maximum width w
    SIZE_t depth               # Depth of the cell in the tree
    SIZE_t cumulative_size     # Number of points included in the subtree with
                               # this cell as a root.

    # Internal constants
    DTYPE_t[3] center          # Store the center for quick split of cells
    DTYPE_t[3] barycenter      # Keep track of the center of mass of the cell

    # Cell boundaries
    DTYPE_t[3] min_bounds      # Inferior boundaries of this cell (inclusive)
    DTYPE_t[3] max_bounds      # Superior boundaries of this cell (exclusive)

# =============================================================================
# Stack data structure
# =============================================================================

# A record on the stack for depth-first tree growing
cdef struct StackRecord:
    SIZE_t start
    SIZE_t end
    SIZE_t depth
    SIZE_t parent
    bint is_left
    double impurity
    SIZE_t n_constant_features

#cdef class Stack:
#    cdef SIZE_t capacity
#    cdef SIZE_t top
#    cdef StackRecord* stack_
#
#    cdef bint is_empty(self) nogil
#    cdef int push(self, SIZE_t start, SIZE_t end, SIZE_t depth, SIZE_t parent,
#                  bint is_left, double impurity,
#                  SIZE_t n_constant_features) nogil except -1
#    cdef int pop(self, StackRecord* res) nogil


# =============================================================================
# PriorityHeap data structure
# =============================================================================

# A record on the frontier for best-first tree growing
cdef struct PriorityHeapRecord:
    SIZE_t node_id
    SIZE_t start
    SIZE_t end
    SIZE_t pos
    SIZE_t depth
    bint is_leaf
    double impurity
    double impurity_left
    double impurity_right
    double improvement

#cdef class PriorityHeap:
#    cdef SIZE_t capacity
#    cdef SIZE_t heap_ptr
#    cdef PriorityHeapRecord* heap_
#
#    cdef bint is_empty(self) nogil
#    cdef void heapify_up(self, PriorityHeapRecord* heap, SIZE_t pos) nogil
#    cdef void heapify_down(self, PriorityHeapRecord* heap, SIZE_t pos, SIZE_t heap_length) nogil
#    cdef int push(self, SIZE_t node_id, SIZE_t start, SIZE_t end, SIZE_t pos,
#                  SIZE_t depth, bint is_leaf, double improvement,
#                  double impurity, double impurity_left,
#                  double impurity_right) nogil except -1
#    cdef int pop(self, PriorityHeapRecord* res) nogil


# safe_realloc(&p, n) resizes the allocation of p to n * sizeof(*p) bytes or
# raises a MemoryError. It never calls free, since that's __dealloc__'s job.
#   cdef DTYPE_t *p = NULL
#   safe_realloc(&p, n)
# is equivalent to p = malloc(n * sizeof(*p)) with error checking.
ctypedef fused realloc_ptr:
    # Add pointer types here as needed.
    (DTYPE_t*)
    (SIZE_t*)
    (unsigned char*)
    (WeightedPQueueRecord*)
    (DOUBLE_t*)
    (DOUBLE_t**)
    (Node*)
    (Cell*)
    (Node**)
    (StackRecord*)
    (PriorityHeapRecord*)
    
# Mitigate precision differences between 32 bit and 64 bit
cdef DTYPE_t FEATURE_THRESHOLD = 1e-7

# Constant to switch between algorithm non zero value extract algorithm
# in SparseSplitter
cdef DTYPE_t EXTRACT_NNZ_SWITCH = 0.1
cdef enum:
    # Max value for our rand_r replacement (near the bottom).
    # We don't use RAND_MAX because it's different across platforms and
    # particularly tiny on Windows/MSVC.
    RAND_R_MAX = 0x7FFFFFFF

cdef realloc_ptr safe_realloc(realloc_ptr* p, size_t nelems) nogil except *:
    # sizeof(realloc_ptr[0]) would be more like idiomatic C, but causes Cython
    # 0.20.1 to crash.
    cdef size_t nbytes = nelems * sizeof(p[0][0])
    if nbytes / sizeof(p[0][0]) != nelems:
        # Overflow in the multiplication
        with gil:
            raise MemoryError("could not allocate (%d * %d) bytes"
                              % (nelems, sizeof(p[0][0])))
    cdef realloc_ptr tmp = <realloc_ptr>realloc(p[0], nbytes)
    if tmp == NULL:
        with gil:
            raise MemoryError("could not allocate %d bytes" % nbytes)

    p[0] = tmp
    return tmp  # for convenience

# rand_r replacement using a 32bit XorShift generator
# See http://www.jstatsoft.org/v08/i14/paper for details
cdef inline UINT32_t our_rand_r(UINT32_t* seed) nogil:
    seed[0] ^= <UINT32_t>(seed[0] << 13)
    seed[0] ^= <UINT32_t>(seed[0] >> 17)
    seed[0] ^= <UINT32_t>(seed[0] << 5)

    return seed[0] % (<UINT32_t>RAND_R_MAX + 1)
    
cdef inline SIZE_t rand_int(SIZE_t low, SIZE_t high,
                            UINT32_t* random_state) nogil:
    """Generate a random integer in [low; end)."""
    return low + our_rand_r(random_state) % (high - low)


cdef inline double rand_uniform(double low, double high,
                                UINT32_t* random_state) nogil:
    """Generate a random double in [low; high)."""
    return ((high - low) * <double> our_rand_r(random_state) /
            <double> RAND_R_MAX) + low


cdef inline double log(double x) nogil:
    return ln(x) / ln(2.0)


cdef inline void _init_split(SplitRecord* self, SIZE_t start_pos) nogil:
    self.impurity_left = INFINITY
    self.impurity_right = INFINITY
    self.pos = start_pos
    self.feature = 0
    self.threshold = 0.
    self.improvement = -INFINITY

cdef class Criterion:
    # The criterion computes the impurity of a node and the reduction of
    # impurity of a split on that node. It also computes the output statistics
    # such as the mean in regression and class probabilities in classification.

    # Internal structures
    cdef DOUBLE_t* y                     # Values of y
    cdef SIZE_t y_stride                 # Stride in y (since n_outputs >= 1)
    cdef DOUBLE_t* sample_weight         # Sample weights

    cdef SIZE_t* samples                 # Sample indices in X, y
    cdef SIZE_t start                    # samples[start:pos] are the samples in the left node
    cdef SIZE_t pos                      # samples[pos:end] are the samples in the right node
    cdef SIZE_t end

    cdef SIZE_t n_outputs                # Number of outputs
    cdef SIZE_t n_samples                # Number of samples
    cdef SIZE_t n_node_samples           # Number of samples in the node (end-start)
    cdef double weighted_n_samples       # Weighted number of samples (in total)
    cdef double weighted_n_node_samples  # Weighted number of samples in the node
    cdef double weighted_n_left          # Weighted number of samples in the left node
    cdef double weighted_n_right         # Weighted number of samples in the right node

    cdef double* sum_total          # For classification criteria, the sum of the
                                    # weighted count of each label. For regression,
                                    # the sum of w*y. sum_total[k] is equal to
                                    # sum_{i=start}^{end-1} w[samples[i]]*y[samples[i], k],
                                    # where k is output index.
    cdef double* sum_left           # Same as above, but for the left side of the split
    cdef double* sum_right          # same as above, but for the right side of the split

    # The criterion object is maintained such that left and right collected
    # statistics correspond to samples[start:pos] and samples[pos:end].

    # Methods
    cdef int init(self, DOUBLE_t* y, SIZE_t y_stride, DOUBLE_t* sample_weight,
                  double weighted_n_samples, SIZE_t* samples, SIZE_t start,
                  SIZE_t end) nogil except -1:
        pass
    cdef int reset(self) nogil except -1:
        pass
    cdef int reverse_reset(self) nogil except -1:
        pass
    cdef int update(self, SIZE_t new_pos) nogil except -1:
        pass
    cdef double node_impurity(self) nogil:
        pass
    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil:
        pass
    cdef void node_value(self, double* dest) nogil:
        pass
    cdef double impurity_improvement(self, double impurity) nogil:
        pass
    cdef double proxy_impurity_improvement(self) nogil:
        pass

#cdef class BestSplitterMT:
#    # The splitter searches in the input space for a feature and a threshold
#    # to split the samples samples[start:end].
#    #
#    # The impurity computations are delegated to a criterion object.
#
#    # Internal structures
#    cdef public Criterion criterion      # Impurity criterion
#    cdef public SIZE_t max_features      # Number of features to test
#    cdef public SIZE_t min_samples_leaf  # Min samples in a leaf
#    cdef public double min_weight_leaf   # Minimum weight in a leaf
#
#    cdef object random_state             # Random state
#    cdef UINT32_t rand_r_state           # sklearn_rand_r random number state
#
#    cdef SIZE_t* samples                 # Sample indices in X, y
#    cdef SIZE_t n_samples                # X.shape[0]
#    cdef double weighted_n_samples       # Weighted number of samples
#    cdef SIZE_t* features                # Feature indices in X
#    cdef SIZE_t* constant_features       # Constant features indices
#    cdef SIZE_t n_features               # X.shape[1]
#    cdef DTYPE_t* feature_values         # temp. array holding feature values
#
#    cdef SIZE_t start                    # Start position for the current node
#    cdef SIZE_t end                      # End position for the current node
#
#    cdef bint presort                    # Whether to use presorting, only
#                                         # allowed on dense data
#
#    cdef DOUBLE_t* y
#    cdef SIZE_t y_stride
#    cdef DOUBLE_t* sample_weight
#
#    cdef DTYPE_t* X
#    cdef SIZE_t X_sample_stride
#    cdef SIZE_t X_feature_stride
#
#    cdef np.ndarray X_idx_sorted
#    cdef INT32_t* X_idx_sorted_ptr
#    cdef SIZE_t X_idx_sorted_stride
#    cdef SIZE_t n_total_samples
#    cdef SIZE_t* sample_mask
#    
#    # The samples vector `samples` is maintained by the Splitter object such
#    # that the samples contained in a node are contiguous. With this setting,
#    # `node_split` reorganizes the node samples `samples[start:end]` in two
#    # subsets `samples[start:pos]` and `samples[pos:end]`.
#
#    # The 1-d  `features` array of size n_features contains the features
#    # indices and allows fast sampling without replacement of features.
#
#    # The 1-d `constant_features` array of size n_features holds in
#    # `constant_features[:n_constant_features]` the feature ids with
#    # constant values for all the samples that reached a specific node.
#    # The value `n_constant_features` is given by the parent node to its
#    # child nodes.  The content of the range `[n_constant_features:]` is left
#    # undefined, but preallocated for performance reasons
#    # This allows optimization with depth-based tree building.
#
#    # Methods
#    cdef int init(self, object X, np.ndarray y,
#                  DOUBLE_t* sample_weight,
#                  np.ndarray X_idx_sorted=NULL) except -1
#
#    cdef int node_reset(self, SIZE_t start, SIZE_t end,
#                        double* weighted_n_node_samples) nogil except -1
#
#    cdef int node_split(self,
#                        double impurity,   # Impurity of the node
#                        SplitRecord* split,
#                        SIZE_t* n_constant_features) nogil except -1
#
#    cdef void node_value(self, double* dest) nogil
#
#    cdef double node_impurity(self) nogil
        
cdef class BestSplitterMT: 
        # Internal structures
    cdef public Criterion criterion      # Impurity criterion
    cdef public SIZE_t max_features      # Number of features to test
    cdef public SIZE_t min_samples_leaf  # Min samples in a leaf
    cdef public double min_weight_leaf   # Minimum weight in a leaf

    cdef object random_state             # Random state
    cdef UINT32_t rand_r_state           # sklearn_rand_r random number state

    cdef SIZE_t* samples                 # Sample indices in X, y
    cdef SIZE_t n_samples                # X.shape[0]
    cdef double weighted_n_samples       # Weighted number of samples
    cdef SIZE_t* features                # Feature indices in X
    cdef SIZE_t* constant_features       # Constant features indices
    cdef SIZE_t n_features               # X.shape[1]
    cdef DTYPE_t* feature_values         # temp. array holding feature values

    cdef SIZE_t start                    # Start position for the current node
    cdef SIZE_t end                      # End position for the current node

    cdef bint presort                    # Whether to use presorting, only
                                         # allowed on dense data

    cdef DOUBLE_t* y
    cdef SIZE_t y_stride
    cdef DOUBLE_t* sample_weight

    cdef DTYPE_t* X
    cdef SIZE_t X_sample_stride
    cdef SIZE_t X_feature_stride

    cdef np.ndarray X_idx_sorted
    cdef INT32_t* X_idx_sorted_ptr
    cdef SIZE_t X_idx_sorted_stride
    cdef SIZE_t n_total_samples
    cdef SIZE_t* sample_mask
    """Abstract splitter class.

    Splitters are called by tree builders to find the best splits on both
    sparse and dense data, one split at a time.
    """
    def __reduce__(self):
        return (BestSplitterMT, (self.criterion,
                               self.max_features,
                               self.min_samples_leaf,
                               self.min_weight_leaf,
                               self.random_state,
                               self.presort), self.__getstate__())
    def __cinit__(self, object criterion, SIZE_t max_features,
                  SIZE_t min_samples_leaf, double min_weight_leaf,
                  object random_state, bint presort):
        """
        Parameters
        ----------
        criterion : Criterion
            The criterion to measure the quality of a split.

        max_features : SIZE_t
            The maximal number of randomly selected features which can be
            considered for a split.

        min_samples_leaf : SIZE_t
            The minimal number of samples each leaf can have, where splits
            which would result in having less samples in a leaf are not
            considered.

        min_weight_leaf : double
            The minimal weight each leaf can have, where the weight is the sum
            of the weights of each sample in it.

        random_state : object
            The user inputted random state to be used for pseudo-randomness
        """

        self.criterion = criterion

        self.samples = NULL
        self.n_samples = 0
        self.features = NULL
        self.n_features = 0
        self.feature_values = NULL

        self.y = NULL
        self.y_stride = 0
        self.sample_weight = NULL

        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_leaf = min_weight_leaf
        self.random_state = random_state
        self.presort = presort

        self.X = NULL
        self.X_sample_stride = 0
        self.X_feature_stride = 0
        self.X_idx_sorted_ptr = NULL
        self.X_idx_sorted_stride = 0
        self.sample_mask = NULL
        self.presort = presort
        
    def __dealloc__(self):
        """Destructor."""

        free(self.samples)
        free(self.features)
        free(self.constant_features)
        free(self.feature_values)
        if self.presort == 1:
            free(self.sample_mask)



            
    def __getstate__(self):
        return {}

    def __setstate__(self, d):
        pass

    cdef int init(self,
                   object X,
                   np.ndarray[DOUBLE_t, ndim=2, mode="c"] y,
                   DOUBLE_t* sample_weight,
                   np.ndarray X_idx_sorted=None) except -1:
        """Initialize the splitter.

        Take in the input data X, the target Y, and optional sample weights.

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.

        Parameters
        ----------
        X : object
            This contains the inputs. Usually it is a 2d numpy array.

        y : numpy.ndarray, dtype=DOUBLE_t
            This is the vector of targets, or true labels, for the samples

        sample_weight : numpy.ndarray, dtype=DOUBLE_t (optional)
            The weights of the samples, where higher weighted samples are fit
            closer than lower weight samples. If not provided, all samples
            are assumed to have uniform weight.
        """

        self.rand_r_state = self.random_state.randint(0, RAND_R_MAX)
        cdef SIZE_t n_samples = X.shape[0]

        # Create a new array which will be used to store nonzero
        # samples from the feature of interest
        cdef SIZE_t* samples = safe_realloc(&self.samples, n_samples)

        cdef SIZE_t i, j
        cdef double weighted_n_samples = 0.0
        j = 0

        for i in range(n_samples):
            # Only work with positively weighted samples
            if sample_weight == NULL or sample_weight[i] != 0.0:
                samples[j] = i
                j += 1

            if sample_weight != NULL:
                weighted_n_samples += sample_weight[i]
            else:
                weighted_n_samples += 1.0

        # Number of samples is number of positively weighted samples
        self.n_samples = j
        self.weighted_n_samples = weighted_n_samples

        cdef SIZE_t n_features = X.shape[1]
        cdef SIZE_t* features = safe_realloc(&self.features, n_features)

        for i in range(n_features):
            features[i] = i

        self.n_features = n_features

        safe_realloc(&self.feature_values, n_samples)
        safe_realloc(&self.constant_features, n_features)

        self.y = <DOUBLE_t*> y.data
        self.y_stride = <SIZE_t> y.strides[0] / <SIZE_t> y.itemsize

        self.sample_weight = sample_weight
        
        #BDS
        cdef np.ndarray X_ndarray = X

        self.X = <DTYPE_t*> X_ndarray.data
        self.X_sample_stride = <SIZE_t> X.strides[0] / <SIZE_t> X.itemsize
        self.X_feature_stride = <SIZE_t> X.strides[1] / <SIZE_t> X.itemsize

        if self.presort == 1:
            self.X_idx_sorted = X_idx_sorted
            self.X_idx_sorted_ptr = <INT32_t*> self.X_idx_sorted.data
            self.X_idx_sorted_stride = (<SIZE_t> self.X_idx_sorted.strides[1] /
                                        <SIZE_t> self.X_idx_sorted.itemsize)

            self.n_total_samples = X.shape[0]
            safe_realloc(&self.sample_mask, self.n_total_samples)
            memset(self.sample_mask, 0, self.n_total_samples*sizeof(SIZE_t))
        return 0



    cdef int node_reset(self, SIZE_t start, SIZE_t end,
                        double* weighted_n_node_samples) nogil except -1:
        """Reset splitter on node samples[start:end].

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.

        Parameters
        ----------
        start : SIZE_t
            The index of the first sample to consider
        end : SIZE_t
            The index of the last sample to consider
        weighted_n_node_samples : numpy.ndarray, dtype=double pointer
            The total weight of those samples
        """

        self.start = start
        self.end = end

        self.criterion.init(self.y,
                            self.y_stride,
                            self.sample_weight,
                            self.weighted_n_samples,
                            self.samples,
                            start,
                            end)

        weighted_n_node_samples[0] = self.criterion.weighted_n_node_samples
        return 0

    cdef int node_split(self, double impurity, SplitRecord* split,
                        SIZE_t* n_constant_features) nogil except -1:
        """Find the best split on node samples[start:end]

        Returns -1 in case of failure to allocate memory (and raise MemoryError)
        or 0 otherwise.
        """
        # Find the best split
        cdef SIZE_t* samples = self.samples
        cdef SIZE_t start = self.start
        cdef SIZE_t end = self.end

        cdef SIZE_t* features = self.features
        cdef SIZE_t* constant_features = self.constant_features
        cdef SIZE_t n_features = self.n_features

        cdef DTYPE_t* X = self.X
        cdef DTYPE_t* Xf = self.feature_values
        cdef SIZE_t X_sample_stride = self.X_sample_stride
        cdef SIZE_t X_feature_stride = self.X_feature_stride
        cdef SIZE_t max_features = self.max_features
        cdef SIZE_t min_samples_leaf = self.min_samples_leaf
        cdef double min_weight_leaf = self.min_weight_leaf
        cdef UINT32_t* random_state = &self.rand_r_state

        cdef INT32_t* X_idx_sorted = self.X_idx_sorted_ptr
        cdef SIZE_t* sample_mask = self.sample_mask

        cdef SplitRecord best, current
        cdef double current_proxy_improvement = -INFINITY
        cdef double best_proxy_improvement = -INFINITY

        cdef SIZE_t f_i = n_features
        cdef SIZE_t f_j
        cdef SIZE_t tmp
        cdef SIZE_t p
        cdef SIZE_t feature_idx_offset
        cdef SIZE_t feature_offset
        cdef SIZE_t i
        cdef SIZE_t j

        cdef SIZE_t n_visited_features = 0
        # Number of features discovered to be constant during the split search
        cdef SIZE_t n_found_constants = 0
        # Number of features known to be constant and drawn without replacement
        cdef SIZE_t n_drawn_constants = 0
        cdef SIZE_t n_known_constants = n_constant_features[0]
        # n_total_constants = n_known_constants + n_found_constants
        cdef SIZE_t n_total_constants = n_known_constants
        cdef DTYPE_t current_feature_value
        cdef SIZE_t partition_end

        _init_split(&best, end)

        if self.presort == 1:
            for p in range(start, end):
                sample_mask[samples[p]] = 1

        # Sample up to max_features without replacement using a
        # Fisher-Yates-based algorithm (using the local variables `f_i` and
        # `f_j` to compute a permutation of the `features` array).
        #
        # Skip the CPU intensive evaluation of the impurity criterion for
        # features that were already detected as constant (hence not suitable
        # for good splitting) by ancestor nodes and save the information on
        # newly discovered constant features to spare computation on descendant
        # nodes.
        while (f_i > n_total_constants and  # Stop early if remaining features
                                            # are constant
                (n_visited_features < max_features or
                 # At least one drawn features must be non constant
                 n_visited_features <= n_found_constants + n_drawn_constants)):

            n_visited_features += 1

            # Loop invariant: elements of features in
            # - [:n_drawn_constant[ holds drawn and known constant features;
            # - [n_drawn_constant:n_known_constant[ holds known constant
            #   features that haven't been drawn yet;
            # - [n_known_constant:n_total_constant[ holds newly found constant
            #   features;
            # - [n_total_constant:f_i[ holds features that haven't been drawn
            #   yet and aren't constant apriori.
            # - [f_i:n_features[ holds features that have been drawn
            #   and aren't constant.

            # Draw a feature at random
            f_j = rand_int(n_drawn_constants, f_i - n_found_constants,
                           random_state)

            if f_j < n_known_constants:
                # f_j in the interval [n_drawn_constants, n_known_constants[
                tmp = features[f_j]
                features[f_j] = features[n_drawn_constants]
                features[n_drawn_constants] = tmp

                n_drawn_constants += 1

            else:
                # f_j in the interval [n_known_constants, f_i - n_found_constants[
                f_j += n_found_constants
                # f_j in the interval [n_total_constants, f_i[
                current.feature = features[f_j]
                feature_offset = self.X_feature_stride * current.feature

                # Sort samples along that feature; either by utilizing
                # presorting, or by copying the values into an array and
                # sorting the array in a manner which utilizes the cache more
                # effectively.
                if self.presort == 1:
                    p = start
                    feature_idx_offset = self.X_idx_sorted_stride * current.feature

                    for i in range(self.n_total_samples): 
                        j = X_idx_sorted[i + feature_idx_offset]
                        if sample_mask[j] == 1:
                            samples[p] = j
                            Xf[p] = X[self.X_sample_stride * j + feature_offset]
                            p += 1
                else:
                    for i in range(start, end):
                        Xf[i] = X[self.X_sample_stride * samples[i] + feature_offset]

                    sort(Xf + start, samples + start, end - start)

                if Xf[end - 1] <= Xf[start] + FEATURE_THRESHOLD:
                    features[f_j] = features[n_total_constants]
                    features[n_total_constants] = current.feature

                    n_found_constants += 1
                    n_total_constants += 1

                else:
                    f_i -= 1
                    features[f_i], features[f_j] = features[f_j], features[f_i]

                    # Evaluate all splits
                    self.criterion.reset()
                    p = start

                    while p < end:
                        while (p + 1 < end and
                               Xf[p + 1] <= Xf[p] + FEATURE_THRESHOLD):
                            p += 1

                        # (p + 1 >= end) or (X[samples[p + 1], current.feature] >
                        #                    X[samples[p], current.feature])
                        p += 1
                        # (p >= end) or (X[samples[p], current.feature] >
                        #                X[samples[p - 1], current.feature])

                        if p < end:
                            current.pos = p

                            # Reject if min_samples_leaf is not guaranteed
                            if (((current.pos - start) < min_samples_leaf) or
                                    ((end - current.pos) < min_samples_leaf)):
                                continue

                            self.criterion.update(current.pos)

                            # Reject if min_weight_leaf is not satisfied
                            if ((self.criterion.weighted_n_left < min_weight_leaf) or
                                    (self.criterion.weighted_n_right < min_weight_leaf)):
                                continue

                            current_proxy_improvement = self.criterion.proxy_impurity_improvement()

                            if current_proxy_improvement > best_proxy_improvement:
                                best_proxy_improvement = current_proxy_improvement
                                current.threshold = (Xf[p - 1] + Xf[p]) / 2.0

                                if current.threshold == Xf[p]:
                                    current.threshold = Xf[p - 1]

                                best = current  # copy

        # Reorganize into samples[start:best.pos] + samples[best.pos:end]
        if best.pos < end:
            feature_offset = X_feature_stride * best.feature
            partition_end = end
            p = start

            while p < partition_end:
                if X[X_sample_stride * samples[p] + feature_offset] <= best.threshold:
                    p += 1

                else:
                    partition_end -= 1

                    tmp = samples[partition_end]
                    samples[partition_end] = samples[p]
                    samples[p] = tmp

            self.criterion.reset()
            self.criterion.update(best.pos)
            best.improvement = self.criterion.impurity_improvement(impurity)
            self.criterion.children_impurity(&best.impurity_left,
                                             &best.impurity_right)

        # Reset sample mask
        if self.presort == 1:
            for p in range(start, end):
                sample_mask[samples[p]] = 0

        # Respect invariant for constant features: the original order of
        # element in features[:n_known_constants] must be preserved for sibling
        # and child nodes
        memcpy(features, constant_features, sizeof(SIZE_t) * n_known_constants)

        # Copy newly found constant features
        memcpy(constant_features + n_known_constants,
               features + n_known_constants,
               sizeof(SIZE_t) * n_found_constants)

        # Return values
        split[0] = best
        n_constant_features[0] = n_total_constants
        return 0

    cdef void node_value(self, double* dest) nogil:
        """Copy the value of node samples[start:end] into dest."""

        self.criterion.node_value(dest)

    cdef double node_impurity(self) nogil:
        """Return the impurity of the current node."""

        return self.criterion.node_impurity()






# Sort n-element arrays pointed to by Xf and samples, simultaneously,
# by the values in Xf. Algorithm: Introsort (Musser, SP&E, 1997).
cdef inline void sort(DTYPE_t* Xf, SIZE_t* samples, SIZE_t n) nogil:
    cdef int maxd = 2 * <int>log(n)
    introsort(Xf, samples, n, maxd)


cdef inline void swap(DTYPE_t* Xf, SIZE_t* samples,
        SIZE_t i, SIZE_t j) nogil:
    # Helper for sort
    Xf[i], Xf[j] = Xf[j], Xf[i]
    samples[i], samples[j] = samples[j], samples[i]


cdef inline DTYPE_t median3(DTYPE_t* Xf, SIZE_t n) nogil:
    # Median of three pivot selection, after Bentley and McIlroy (1993).
    # Engineering a sort function. SP&E. Requires 8/3 comparisons on average.
    cdef DTYPE_t a = Xf[0], b = Xf[n / 2], c = Xf[n - 1]
    if a < b:
        if b < c:
            return b
        elif a < c:
            return c
        else:
            return a
    elif b < c:
        if a < c:
            return a
        else:
            return c
    else:
        return b


# Introsort with median of 3 pivot selection and 3-way partition function
# (robust to repeated elements, e.g. lots of zero features).
cdef void introsort(DTYPE_t* Xf, SIZE_t *samples,
                    SIZE_t n, int maxd) nogil:
    cdef DTYPE_t pivot
    cdef SIZE_t i, l, r

    while n > 1:
        if maxd <= 0:   # max depth limit exceeded ("gone quadratic")
            heapsort(Xf, samples, n)
            return
        maxd -= 1

        pivot = median3(Xf, n)

        # Three-way partition.
        i = l = 0
        r = n
        while i < r:
            if Xf[i] < pivot:
                swap(Xf, samples, i, l)
                i += 1
                l += 1
            elif Xf[i] > pivot:
                r -= 1
                swap(Xf, samples, i, r)
            else:
                i += 1

        introsort(Xf, samples, l, maxd)
        Xf += r
        samples += r
        n -= r


cdef inline void sift_down(DTYPE_t* Xf, SIZE_t* samples,
                           SIZE_t start, SIZE_t end) nogil:
    # Restore heap order in Xf[start:end] by moving the max element to start.
    cdef SIZE_t child, maxind, root

    root = start
    while True:
        child = root * 2 + 1

        # find max of root, left child, right child
        maxind = root
        if child < end and Xf[maxind] < Xf[child]:
            maxind = child
        if child + 1 < end and Xf[maxind] < Xf[child + 1]:
            maxind = child + 1

        if maxind == root:
            break
        else:
            swap(Xf, samples, root, maxind)
            root = maxind


cdef void heapsort(DTYPE_t* Xf, SIZE_t* samples, SIZE_t n) nogil:
    cdef SIZE_t start, end

    # heapify
    start = (n - 2) / 2
    end = n
    while True:
        sift_down(Xf, samples, start, end)
        if start == 0:
            break
        start -= 1

    # sort by shrinking the heap, putting the max element immediately after it
    end = n - 1
    while end > 0:
        swap(Xf, samples, 0, end)
        sift_down(Xf, samples, 0, end)
        end = end - 1

#
#cdef class RandomSplitter(BaseDenseSplitter):
#    """Splitter for finding the best random split."""
#    def __reduce__(self):
#        return (RandomSplitter, (self.criterion,
#                                 self.max_features,
#                                 self.min_samples_leaf,
#                                 self.min_weight_leaf,
#                                 self.random_state,
#                                 self.presort), self.__getstate__())
#
#    cdef int node_split(self, double impurity, SplitRecord* split,
#                        SIZE_t* n_constant_features) nogil except -1:
#        """Find the best random split on node samples[start:end]
#
#        Returns -1 in case of failure to allocate memory (and raise MemoryError)
#        or 0 otherwise.
#        """
#        # Draw random splits and pick the best
#        cdef SIZE_t* samples = self.samples
#        cdef SIZE_t start = self.start
#        cdef SIZE_t end = self.end
#
#        cdef SIZE_t* features = self.features
#        cdef SIZE_t* constant_features = self.constant_features
#        cdef SIZE_t n_features = self.n_features
#
#        cdef DTYPE_t* X = self.X
#        cdef DTYPE_t* Xf = self.feature_values
#        cdef SIZE_t X_sample_stride = self.X_sample_stride
#        cdef SIZE_t X_feature_stride = self.X_feature_stride
#        cdef SIZE_t max_features = self.max_features
#        cdef SIZE_t min_samples_leaf = self.min_samples_leaf
#        cdef double min_weight_leaf = self.min_weight_leaf
#        cdef UINT32_t* random_state = &self.rand_r_state
#
#        cdef SplitRecord best, current
#        cdef double current_proxy_improvement = - INFINITY
#        cdef double best_proxy_improvement = - INFINITY
#
#        cdef SIZE_t f_i = n_features
#        cdef SIZE_t f_j
#        cdef SIZE_t p
#        cdef SIZE_t tmp
#        cdef SIZE_t feature_stride
#        # Number of features discovered to be constant during the split search
#        cdef SIZE_t n_found_constants = 0
#        # Number of features known to be constant and drawn without replacement
#        cdef SIZE_t n_drawn_constants = 0
#        cdef SIZE_t n_known_constants = n_constant_features[0]
#        # n_total_constants = n_known_constants + n_found_constants
#        cdef SIZE_t n_total_constants = n_known_constants
#        cdef SIZE_t n_visited_features = 0
#        cdef DTYPE_t min_feature_value
#        cdef DTYPE_t max_feature_value
#        cdef DTYPE_t current_feature_value
#        cdef SIZE_t partition_end
#
#        _init_split(&best, end)
#
#        # Sample up to max_features without replacement using a
#        # Fisher-Yates-based algorithm (using the local variables `f_i` and
#        # `f_j` to compute a permutation of the `features` array).
#        #
#        # Skip the CPU intensive evaluation of the impurity criterion for
#        # features that were already detected as constant (hence not suitable
#        # for good splitting) by ancestor nodes and save the information on
#        # newly discovered constant features to spare computation on descendant
#        # nodes.
#        while (f_i > n_total_constants and  # Stop early if remaining features
#                                            # are constant
#                (n_visited_features < max_features or
#                 # At least one drawn features must be non constant
#                 n_visited_features <= n_found_constants + n_drawn_constants)):
#            n_visited_features += 1
#
#            # Loop invariant: elements of features in
#            # - [:n_drawn_constant[ holds drawn and known constant features;
#            # - [n_drawn_constant:n_known_constant[ holds known constant
#            #   features that haven't been drawn yet;
#            # - [n_known_constant:n_total_constant[ holds newly found constant
#            #   features;
#            # - [n_total_constant:f_i[ holds features that haven't been drawn
#            #   yet and aren't constant apriori.
#            # - [f_i:n_features[ holds features that have been drawn
#            #   and aren't constant.
#
#            # Draw a feature at random
#            f_j = rand_int(n_drawn_constants, f_i - n_found_constants,
#                           random_state)
#
#            if f_j < n_known_constants:
#                # f_j in the interval [n_drawn_constants, n_known_constants[
#                tmp = features[f_j]
#                features[f_j] = features[n_drawn_constants]
#                features[n_drawn_constants] = tmp
#
#                n_drawn_constants += 1
#
#            else:
#                # f_j in the interval [n_known_constants, f_i - n_found_constants[
#                f_j += n_found_constants
#                # f_j in the interval [n_total_constants, f_i[
#
#                current.feature = features[f_j]
#                feature_stride = X_feature_stride * current.feature
#
#                # Find min, max
#                min_feature_value = X[X_sample_stride * samples[start] + feature_stride]
#                max_feature_value = min_feature_value
#                Xf[start] = min_feature_value
#
#                for p in range(start + 1, end):
#                    current_feature_value = X[X_sample_stride * samples[p] + feature_stride]
#                    Xf[p] = current_feature_value
#
#                    if current_feature_value < min_feature_value:
#                        min_feature_value = current_feature_value
#                    elif current_feature_value > max_feature_value:
#                        max_feature_value = current_feature_value
#
#                if max_feature_value <= min_feature_value + FEATURE_THRESHOLD:
#                    features[f_j] = features[n_total_constants]
#                    features[n_total_constants] = current.feature
#
#                    n_found_constants += 1
#                    n_total_constants += 1
#
#                else:
#                    f_i -= 1
#                    features[f_i], features[f_j] = features[f_j], features[f_i]
#
#                    # Draw a random threshold
#                    current.threshold = rand_uniform(min_feature_value,
#                                                     max_feature_value,
#                                                     random_state)
#
#                    if current.threshold == max_feature_value:
#                        current.threshold = min_feature_value
#
#                    # Partition
#                    partition_end = end
#                    p = start
#                    while p < partition_end:
#                        current_feature_value = Xf[p]
#                        if current_feature_value <= current.threshold:
#                            p += 1
#                        else:
#                            partition_end -= 1
#
#                            Xf[p] = Xf[partition_end]
#                            Xf[partition_end] = current_feature_value
#
#                            tmp = samples[partition_end]
#                            samples[partition_end] = samples[p]
#                            samples[p] = tmp
#
#                    current.pos = partition_end
#
#                    # Reject if min_samples_leaf is not guaranteed
#                    if (((current.pos - start) < min_samples_leaf) or
#                            ((end - current.pos) < min_samples_leaf)):
#                        continue
#
#                    # Evaluate split
#                    self.criterion.reset()
#                    self.criterion.update(current.pos)
#
#                    # Reject if min_weight_leaf is not satisfied
#                    if ((self.criterion.weighted_n_left < min_weight_leaf) or
#                            (self.criterion.weighted_n_right < min_weight_leaf)):
#                        continue
#
#                    current_proxy_improvement = self.criterion.proxy_impurity_improvement()
#
#                    if current_proxy_improvement > best_proxy_improvement:
#                        best_proxy_improvement = current_proxy_improvement
#                        best = current  # copy
#
#        # Reorganize into samples[start:best.pos] + samples[best.pos:end]
#        feature_stride = X_feature_stride * best.feature
#        if best.pos < end:
#            if current.feature != best.feature:
#                partition_end = end
#                p = start
#
#                while p < partition_end:
#                    if X[X_sample_stride * samples[p] + feature_stride] <= best.threshold:
#                        p += 1
#
#                    else:
#                        partition_end -= 1
#
#                        tmp = samples[partition_end]
#                        samples[partition_end] = samples[p]
#                        samples[p] = tmp
#
#
#            self.criterion.reset()
#            self.criterion.update(best.pos)
#            best.improvement = self.criterion.impurity_improvement(impurity)
#            self.criterion.children_impurity(&best.impurity_left,
#                                             &best.impurity_right)
#
#        # Respect invariant for constant features: the original order of
#        # element in features[:n_known_constants] must be preserved for sibling
#        # and child nodes
#        memcpy(features, constant_features, sizeof(SIZE_t) * n_known_constants)
#
#        # Copy newly found constant features
#        memcpy(constant_features + n_known_constants,
#               features + n_known_constants,
#               sizeof(SIZE_t) * n_found_constants)
#
#        # Return values
#        split[0] = best
#        n_constant_features[0] = n_total_constants
#        return 0
#
#
#cdef class BaseSparseSplitter(Splitter):
#    # The sparse splitter works only with csc sparse matrix format
#    cdef DTYPE_t* X_data
#    cdef INT32_t* X_indices
#    cdef INT32_t* X_indptr
#
#    cdef SIZE_t n_total_samples
#
#    cdef SIZE_t* index_to_samples
#    cdef SIZE_t* sorted_samples
#
#    def __cinit__(self, Criterion criterion, SIZE_t max_features,
#                  SIZE_t min_samples_leaf, double min_weight_leaf,
#                  object random_state, bint presort):
#        # Parent __cinit__ is automatically called
#
#        self.X_data = NULL
#        self.X_indices = NULL
#        self.X_indptr = NULL
#
#        self.n_total_samples = 0
#
#        self.index_to_samples = NULL
#        self.sorted_samples = NULL
#
#    def __dealloc__(self):
#        """Deallocate memory."""
#        free(self.index_to_samples)
#        free(self.sorted_samples)
#
#    cdef int init(self,
#                  object X,
#                  np.ndarray[DOUBLE_t, ndim=2, mode="c"] y,
#                  DOUBLE_t* sample_weight,
#                  np.ndarray X_idx_sorted=None) except -1:
#        """Initialize the splitter
#
#        Returns -1 in case of failure to allocate memory (and raise MemoryError)
#        or 0 otherwise.
#        """
#        # Call parent init
#        Splitter.init(self, X, y, sample_weight)
#
#        if not isinstance(X, csc_matrix):
#            raise ValueError("X should be in csc format")
#
#        cdef SIZE_t* samples = self.samples
#        cdef SIZE_t n_samples = self.n_samples
#
#        # Initialize X
#        cdef np.ndarray[dtype=DTYPE_t, ndim=1] data = X.data
#        cdef np.ndarray[dtype=INT32_t, ndim=1] indices = X.indices
#        cdef np.ndarray[dtype=INT32_t, ndim=1] indptr = X.indptr
#        cdef SIZE_t n_total_samples = X.shape[0]
#
#        self.X_data = <DTYPE_t*> data.data
#        self.X_indices = <INT32_t*> indices.data
#        self.X_indptr = <INT32_t*> indptr.data
#        self.n_total_samples = n_total_samples
#
#        # Initialize auxiliary array used to perform split
#        safe_realloc(&self.index_to_samples, n_total_samples)
#        safe_realloc(&self.sorted_samples, n_samples)
#
#        cdef SIZE_t* index_to_samples = self.index_to_samples
#        cdef SIZE_t p
#        for p in range(n_total_samples):
#            index_to_samples[p] = -1
#
#        for p in range(n_samples):
#            index_to_samples[samples[p]] = p
#        return 0
#
#    cdef inline SIZE_t _partition(self, double threshold,
#                                  SIZE_t end_negative, SIZE_t start_positive,
#                                  SIZE_t zero_pos) nogil:
#        """Partition samples[start:end] based on threshold."""
#
#        cdef double value
#        cdef SIZE_t partition_end
#        cdef SIZE_t p
#
#        cdef DTYPE_t* Xf = self.feature_values
#        cdef SIZE_t* samples = self.samples
#        cdef SIZE_t* index_to_samples = self.index_to_samples
#
#        if threshold < 0.:
#            p = self.start
#            partition_end = end_negative
#        elif threshold > 0.:
#            p = start_positive
#            partition_end = self.end
#        else:
#            # Data are already split
#            return zero_pos
#
#        while p < partition_end:
#            value = Xf[p]
#
#            if value <= threshold:
#                p += 1
#
#            else:
#                partition_end -= 1
#
#                Xf[p] = Xf[partition_end]
#                Xf[partition_end] = value
#                sparse_swap(index_to_samples, samples, p, partition_end)
#
#        return partition_end
#
#    cdef inline void extract_nnz(self, SIZE_t feature,
#                                 SIZE_t* end_negative, SIZE_t* start_positive,
#                                 bint* is_samples_sorted) nogil:
#        """Extract and partition values for a given feature.
#
#        The extracted values are partitioned between negative values
#        Xf[start:end_negative[0]] and positive values Xf[start_positive[0]:end].
#        The samples and index_to_samples are modified according to this
#        partition.
#
#        The extraction corresponds to the intersection between the arrays
#        X_indices[indptr_start:indptr_end] and samples[start:end].
#        This is done efficiently using either an index_to_samples based approach
#        or binary search based approach.
#
#        Parameters
#        ----------
#        feature : SIZE_t,
#            Index of the feature we want to extract non zero value.
#
#
#        end_negative, start_positive : SIZE_t*, SIZE_t*,
#            Return extracted non zero values in self.samples[start:end] where
#            negative values are in self.feature_values[start:end_negative[0]]
#            and positive values are in
#            self.feature_values[start_positive[0]:end].
#
#        is_samples_sorted : bint*,
#            If is_samples_sorted, then self.sorted_samples[start:end] will be
#            the sorted version of self.samples[start:end].
#
#        """
#        cdef SIZE_t indptr_start = self.X_indptr[feature],
#        cdef SIZE_t indptr_end = self.X_indptr[feature + 1]
#        cdef SIZE_t n_indices = <SIZE_t>(indptr_end - indptr_start)
#        cdef SIZE_t n_samples = self.end - self.start
#
#        # Use binary search if n_samples * log(n_indices) <
#        # n_indices and index_to_samples approach otherwise.
#        # O(n_samples * log(n_indices)) is the running time of binary
#        # search and O(n_indices) is the running time of index_to_samples
#        # approach.
#        if ((1 - is_samples_sorted[0]) * n_samples * log(n_samples) +
#                n_samples * log(n_indices) < EXTRACT_NNZ_SWITCH * n_indices):
#            extract_nnz_binary_search(self.X_indices, self.X_data,
#                                      indptr_start, indptr_end,
#                                      self.samples, self.start, self.end,
#                                      self.index_to_samples,
#                                      self.feature_values,
#                                      end_negative, start_positive,
#                                      self.sorted_samples, is_samples_sorted)
#
#        # Using an index to samples  technique to extract non zero values
#        # index_to_samples is a mapping from X_indices to samples
#        else:
#            extract_nnz_index_to_samples(self.X_indices, self.X_data,
#                                         indptr_start, indptr_end,
#                                         self.samples, self.start, self.end,
#                                         self.index_to_samples,
#                                         self.feature_values,
#                                         end_negative, start_positive)
#
#
#cdef int compare_SIZE_t(const void* a, const void* b) nogil:
#    """Comparison function for sort."""
#    return <int>((<SIZE_t*>a)[0] - (<SIZE_t*>b)[0])
#
#
#cdef inline void binary_search(INT32_t* sorted_array,
#                               INT32_t start, INT32_t end,
#                               SIZE_t value, SIZE_t* index,
#                               INT32_t* new_start) nogil:
#    """Return the index of value in the sorted array.
#
#    If not found, return -1. new_start is the last pivot + 1
#    """
#    cdef INT32_t pivot
#    index[0] = -1
#    while start < end:
#        pivot = start + (end - start) / 2
#
#        if sorted_array[pivot] == value:
#            index[0] = pivot
#            start = pivot + 1
#            break
#
#        if sorted_array[pivot] < value:
#            start = pivot + 1
#        else:
#            end = pivot
#    new_start[0] = start
#
#
#cdef inline void extract_nnz_index_to_samples(INT32_t* X_indices,
#                                              DTYPE_t* X_data,
#                                              INT32_t indptr_start,
#                                              INT32_t indptr_end,
#                                              SIZE_t* samples,
#                                              SIZE_t start,
#                                              SIZE_t end,
#                                              SIZE_t* index_to_samples,
#                                              DTYPE_t* Xf,
#                                              SIZE_t* end_negative,
#                                              SIZE_t* start_positive) nogil:
#    """Extract and partition values for a feature using index_to_samples.
#
#    Complexity is O(indptr_end - indptr_start).
#    """
#    cdef INT32_t k
#    cdef SIZE_t index
#    cdef SIZE_t end_negative_ = start
#    cdef SIZE_t start_positive_ = end
#
#    for k in range(indptr_start, indptr_end):
#        if start <= index_to_samples[X_indices[k]] < end:
#            if X_data[k] > 0:
#                start_positive_ -= 1
#                Xf[start_positive_] = X_data[k]
#                index = index_to_samples[X_indices[k]]
#                sparse_swap(index_to_samples, samples, index, start_positive_)
#
#
#            elif X_data[k] < 0:
#                Xf[end_negative_] = X_data[k]
#                index = index_to_samples[X_indices[k]]
#                sparse_swap(index_to_samples, samples, index, end_negative_)
#                end_negative_ += 1
#
#    # Returned values
#    end_negative[0] = end_negative_
#    start_positive[0] = start_positive_
#
#
#cdef inline void extract_nnz_binary_search(INT32_t* X_indices,
#                                           DTYPE_t* X_data,
#                                           INT32_t indptr_start,
#                                           INT32_t indptr_end,
#                                           SIZE_t* samples,
#                                           SIZE_t start,
#                                           SIZE_t end,
#                                           SIZE_t* index_to_samples,
#                                           DTYPE_t* Xf,
#                                           SIZE_t* end_negative,
#                                           SIZE_t* start_positive,
#                                           SIZE_t* sorted_samples,
#                                           bint* is_samples_sorted) nogil:
#    """Extract and partition values for a given feature using binary search.
#
#    If n_samples = end - start and n_indices = indptr_end - indptr_start,
#    the complexity is
#
#        O((1 - is_samples_sorted[0]) * n_samples * log(n_samples) +
#          n_samples * log(n_indices)).
#    """
#    cdef SIZE_t n_samples
#
#    if not is_samples_sorted[0]:
#        n_samples = end - start
#        memcpy(sorted_samples + start, samples + start,
#               n_samples * sizeof(SIZE_t))
#        qsort(sorted_samples + start, n_samples, sizeof(SIZE_t),
#              compare_SIZE_t)
#        is_samples_sorted[0] = 1
#
#    while (indptr_start < indptr_end and
#           sorted_samples[start] > X_indices[indptr_start]):
#        indptr_start += 1
#
#    while (indptr_start < indptr_end and
#           sorted_samples[end - 1] < X_indices[indptr_end - 1]):
#        indptr_end -= 1
#
#    cdef SIZE_t p = start
#    cdef SIZE_t index
#    cdef SIZE_t k
#    cdef SIZE_t end_negative_ = start
#    cdef SIZE_t start_positive_ = end
#
#    while (p < end and indptr_start < indptr_end):
#        # Find index of sorted_samples[p] in X_indices
#        binary_search(X_indices, indptr_start, indptr_end,
#                      sorted_samples[p], &k, &indptr_start)
#
#        if k != -1:
#             # If k != -1, we have found a non zero value
#
#            if X_data[k] > 0:
#                start_positive_ -= 1
#                Xf[start_positive_] = X_data[k]
#                index = index_to_samples[X_indices[k]]
#                sparse_swap(index_to_samples, samples, index, start_positive_)
#
#
#            elif X_data[k] < 0:
#                Xf[end_negative_] = X_data[k]
#                index = index_to_samples[X_indices[k]]
#                sparse_swap(index_to_samples, samples, index, end_negative_)
#                end_negative_ += 1
#        p += 1
#
#    # Returned values
#    end_negative[0] = end_negative_
#    start_positive[0] = start_positive_
#
#
#cdef inline void sparse_swap(SIZE_t* index_to_samples, SIZE_t* samples,
#                             SIZE_t pos_1, SIZE_t pos_2) nogil:
#    """Swap sample pos_1 and pos_2 preserving sparse invariant."""
#    samples[pos_1], samples[pos_2] =  samples[pos_2], samples[pos_1]
#    index_to_samples[samples[pos_1]] = pos_1
#    index_to_samples[samples[pos_2]] = pos_2
#
#
#cdef class BestSparseSplitter(BaseSparseSplitter):
#    """Splitter for finding the best split, using the sparse data."""
#
#    def __reduce__(self):
#        return (BestSparseSplitter, (self.criterion,
#                                     self.max_features,
#                                     self.min_samples_leaf,
#                                     self.min_weight_leaf,
#                                     self.random_state,
#                                     self.presort), self.__getstate__())
#
#    cdef int node_split(self, double impurity, SplitRecord* split,
#                        SIZE_t* n_constant_features) nogil except -1:
#        """Find the best split on node samples[start:end], using sparse features
#
#        Returns -1 in case of failure to allocate memory (and raise MemoryError)
#        or 0 otherwise.
#        """
#        # Find the best split
#        cdef SIZE_t* samples = self.samples
#        cdef SIZE_t start = self.start
#        cdef SIZE_t end = self.end
#
#        cdef INT32_t* X_indices = self.X_indices
#        cdef INT32_t* X_indptr = self.X_indptr
#        cdef DTYPE_t* X_data = self.X_data
#
#        cdef SIZE_t* features = self.features
#        cdef SIZE_t* constant_features = self.constant_features
#        cdef SIZE_t n_features = self.n_features
#
#        cdef DTYPE_t* Xf = self.feature_values
#        cdef SIZE_t* sorted_samples = self.sorted_samples
#        cdef SIZE_t* index_to_samples = self.index_to_samples
#        cdef SIZE_t max_features = self.max_features
#        cdef SIZE_t min_samples_leaf = self.min_samples_leaf
#        cdef double min_weight_leaf = self.min_weight_leaf
#        cdef UINT32_t* random_state = &self.rand_r_state
#
#        cdef SplitRecord best, current
#        _init_split(&best, end)
#        cdef double current_proxy_improvement = - INFINITY
#        cdef double best_proxy_improvement = - INFINITY
#
#        cdef SIZE_t f_i = n_features
#        cdef SIZE_t f_j, p, tmp
#        cdef SIZE_t n_visited_features = 0
#        # Number of features discovered to be constant during the split search
#        cdef SIZE_t n_found_constants = 0
#        # Number of features known to be constant and drawn without replacement
#        cdef SIZE_t n_drawn_constants = 0
#        cdef SIZE_t n_known_constants = n_constant_features[0]
#        # n_total_constants = n_known_constants + n_found_constants
#        cdef SIZE_t n_total_constants = n_known_constants
#        cdef DTYPE_t current_feature_value
#
#        cdef SIZE_t p_next
#        cdef SIZE_t p_prev
#        cdef bint is_samples_sorted = 0  # indicate is sorted_samples is
#                                         # inititialized
#
#        # We assume implicitely that end_positive = end and
#        # start_negative = start
#        cdef SIZE_t start_positive
#        cdef SIZE_t end_negative
#
#        # Sample up to max_features without replacement using a
#        # Fisher-Yates-based algorithm (using the local variables `f_i` and
#        # `f_j` to compute a permutation of the `features` array).
#        #
#        # Skip the CPU intensive evaluation of the impurity criterion for
#        # features that were already detected as constant (hence not suitable
#        # for good splitting) by ancestor nodes and save the information on
#        # newly discovered constant features to spare computation on descendant
#        # nodes.
#        while (f_i > n_total_constants and  # Stop early if remaining features
#                                            # are constant
#                (n_visited_features < max_features or
#                 # At least one drawn features must be non constant
#                 n_visited_features <= n_found_constants + n_drawn_constants)):
#
#            n_visited_features += 1
#
#            # Loop invariant: elements of features in
#            # - [:n_drawn_constant[ holds drawn and known constant features;
#            # - [n_drawn_constant:n_known_constant[ holds known constant
#            #   features that haven't been drawn yet;
#            # - [n_known_constant:n_total_constant[ holds newly found constant
#            #   features;
#            # - [n_total_constant:f_i[ holds features that haven't been drawn
#            #   yet and aren't constant apriori.
#            # - [f_i:n_features[ holds features that have been drawn
#            #   and aren't constant.
#
#            # Draw a feature at random
#            f_j = rand_int(n_drawn_constants, f_i - n_found_constants,
#                           random_state)
#
#            if f_j < n_known_constants:
#                # f_j in the interval [n_drawn_constants, n_known_constants[
#                tmp = features[f_j]
#                features[f_j] = features[n_drawn_constants]
#                features[n_drawn_constants] = tmp
#
#                n_drawn_constants += 1
#
#            else:
#                # f_j in the interval [n_known_constants, f_i - n_found_constants[
#                f_j += n_found_constants
#                # f_j in the interval [n_total_constants, f_i[
#
#                current.feature = features[f_j]
#                self.extract_nnz(current.feature,
#                                 &end_negative, &start_positive,
#                                 &is_samples_sorted)
#
#                # Sort the positive and negative parts of `Xf`
#                sort(Xf + start, samples + start, end_negative - start)
#                sort(Xf + start_positive, samples + start_positive,
#                     end - start_positive)
#
#                # Update index_to_samples to take into account the sort
#                for p in range(start, end_negative):
#                    index_to_samples[samples[p]] = p
#                for p in range(start_positive, end):
#                    index_to_samples[samples[p]] = p
#
#                # Add one or two zeros in Xf, if there is any
#                if end_negative < start_positive:
#                    start_positive -= 1
#                    Xf[start_positive] = 0.
#
#                    if end_negative != start_positive:
#                        Xf[end_negative] = 0.
#                        end_negative += 1
#
#                if Xf[end - 1] <= Xf[start] + FEATURE_THRESHOLD:
#                    features[f_j] = features[n_total_constants]
#                    features[n_total_constants] = current.feature
#
#                    n_found_constants += 1
#                    n_total_constants += 1
#
#                else:
#                    f_i -= 1
#                    features[f_i], features[f_j] = features[f_j], features[f_i]
#
#                    # Evaluate all splits
#                    self.criterion.reset()
#                    p = start
#
#                    while p < end:
#                        if p + 1 != end_negative:
#                            p_next = p + 1
#                        else:
#                            p_next = start_positive
#
#                        while (p_next < end and
#                               Xf[p_next] <= Xf[p] + FEATURE_THRESHOLD):
#                            p = p_next
#                            if p + 1 != end_negative:
#                                p_next = p + 1
#                            else:
#                                p_next = start_positive
#
#
#                        # (p_next >= end) or (X[samples[p_next], current.feature] >
#                        #                     X[samples[p], current.feature])
#                        p_prev = p
#                        p = p_next
#                        # (p >= end) or (X[samples[p], current.feature] >
#                        #                X[samples[p_prev], current.feature])
#
#
#                        if p < end:
#                            current.pos = p
#
#                            # Reject if min_samples_leaf is not guaranteed
#                            if (((current.pos - start) < min_samples_leaf) or
#                                    ((end - current.pos) < min_samples_leaf)):
#                                continue
#
#                            self.criterion.update(current.pos)
#
#                            # Reject if min_weight_leaf is not satisfied
#                            if ((self.criterion.weighted_n_left < min_weight_leaf) or
#                                    (self.criterion.weighted_n_right < min_weight_leaf)):
#                                continue
#
#                            current_proxy_improvement = self.criterion.proxy_impurity_improvement()
#
#                            if current_proxy_improvement > best_proxy_improvement:
#                                best_proxy_improvement = current_proxy_improvement
#
#                                current.threshold = (Xf[p_prev] + Xf[p]) / 2.0
#                                if current.threshold == Xf[p]:
#                                    current.threshold = Xf[p_prev]
#
#                                best = current
#
#        # Reorganize into samples[start:best.pos] + samples[best.pos:end]
#        if best.pos < end:
#            self.extract_nnz(best.feature, &end_negative, &start_positive,
#                             &is_samples_sorted)
#
#            self._partition(best.threshold, end_negative, start_positive,
#                            best.pos)
#
#            self.criterion.reset()
#            self.criterion.update(best.pos)
#            best.improvement = self.criterion.impurity_improvement(impurity)
#            self.criterion.children_impurity(&best.impurity_left,
#                                             &best.impurity_right)
#
#        # Respect invariant for constant features: the original order of
#        # element in features[:n_known_constants] must be preserved for sibling
#        # and child nodes
#        memcpy(features, constant_features, sizeof(SIZE_t) * n_known_constants)
#
#        # Copy newly found constant features
#        memcpy(constant_features + n_known_constants,
#               features + n_known_constants,
#               sizeof(SIZE_t) * n_found_constants)
#
#        # Return values
#        split[0] = best
#        n_constant_features[0] = n_total_constants
#        return 0
#
#
#cdef class RandomSparseSplitter(BaseSparseSplitter):
#    """Splitter for finding a random split, using the sparse data."""
#
#    def __reduce__(self):
#        return (RandomSparseSplitter, (self.criterion,
#                                       self.max_features,
#                                       self.min_samples_leaf,
#                                       self.min_weight_leaf,
#                                       self.random_state,
#                                       self.presort), self.__getstate__())
#
#    cdef int node_split(self, double impurity, SplitRecord* split,
#                        SIZE_t* n_constant_features) nogil except -1:
#        """Find a random split on node samples[start:end], using sparse features
#
#        Returns -1 in case of failure to allocate memory (and raise MemoryError)
#        or 0 otherwise.
#        """
#        # Find the best split
#        cdef SIZE_t* samples = self.samples
#        cdef SIZE_t start = self.start
#        cdef SIZE_t end = self.end
#
#        cdef INT32_t* X_indices = self.X_indices
#        cdef INT32_t* X_indptr = self.X_indptr
#        cdef DTYPE_t* X_data = self.X_data
#
#        cdef SIZE_t* features = self.features
#        cdef SIZE_t* constant_features = self.constant_features
#        cdef SIZE_t n_features = self.n_features
#
#        cdef DTYPE_t* Xf = self.feature_values
#        cdef SIZE_t* sorted_samples = self.sorted_samples
#        cdef SIZE_t* index_to_samples = self.index_to_samples
#        cdef SIZE_t max_features = self.max_features
#        cdef SIZE_t min_samples_leaf = self.min_samples_leaf
#        cdef double min_weight_leaf = self.min_weight_leaf
#        cdef UINT32_t* random_state = &self.rand_r_state
#
#        cdef SplitRecord best, current
#        _init_split(&best, end)
#        cdef double current_proxy_improvement = - INFINITY
#        cdef double best_proxy_improvement = - INFINITY
#
#        cdef DTYPE_t current_feature_value
#
#        cdef SIZE_t f_i = n_features
#        cdef SIZE_t f_j, p, tmp
#        cdef SIZE_t n_visited_features = 0
#        # Number of features discovered to be constant during the split search
#        cdef SIZE_t n_found_constants = 0
#        # Number of features known to be constant and drawn without replacement
#        cdef SIZE_t n_drawn_constants = 0
#        cdef SIZE_t n_known_constants = n_constant_features[0]
#        # n_total_constants = n_known_constants + n_found_constants
#        cdef SIZE_t n_total_constants = n_known_constants
#        cdef SIZE_t partition_end
#
#        cdef DTYPE_t min_feature_value
#        cdef DTYPE_t max_feature_value
#
#        cdef bint is_samples_sorted = 0  # indicate that sorted_samples is
#                                         # inititialized
#
#        # We assume implicitely that end_positive = end and
#        # start_negative = start
#        cdef SIZE_t start_positive
#        cdef SIZE_t end_negative
#
#        # Sample up to max_features without replacement using a
#        # Fisher-Yates-based algorithm (using the local variables `f_i` and
#        # `f_j` to compute a permutation of the `features` array).
#        #
#        # Skip the CPU intensive evaluation of the impurity criterion for
#        # features that were already detected as constant (hence not suitable
#        # for good splitting) by ancestor nodes and save the information on
#        # newly discovered constant features to spare computation on descendant
#        # nodes.
#        while (f_i > n_total_constants and  # Stop early if remaining features
#                                            # are constant
#                (n_visited_features < max_features or
#                 # At least one drawn features must be non constant
#                 n_visited_features <= n_found_constants + n_drawn_constants)):
#
#            n_visited_features += 1
#
#            # Loop invariant: elements of features in
#            # - [:n_drawn_constant[ holds drawn and known constant features;
#            # - [n_drawn_constant:n_known_constant[ holds known constant
#            #   features that haven't been drawn yet;
#            # - [n_known_constant:n_total_constant[ holds newly found constant
#            #   features;
#            # - [n_total_constant:f_i[ holds features that haven't been drawn
#            #   yet and aren't constant apriori.
#            # - [f_i:n_features[ holds features that have been drawn
#            #   and aren't constant.
#
#            # Draw a feature at random
#            f_j = rand_int(n_drawn_constants, f_i - n_found_constants,
#                           random_state)
#
#            if f_j < n_known_constants:
#                # f_j in the interval [n_drawn_constants, n_known_constants[
#                tmp = features[f_j]
#                features[f_j] = features[n_drawn_constants]
#                features[n_drawn_constants] = tmp
#
#                n_drawn_constants += 1
#
#            else:
#                # f_j in the interval [n_known_constants, f_i - n_found_constants[
#                f_j += n_found_constants
#                # f_j in the interval [n_total_constants, f_i[
#
#                current.feature = features[f_j]
#
#                self.extract_nnz(current.feature,
#                                 &end_negative, &start_positive,
#                                 &is_samples_sorted)
#
#                # Add one or two zeros in Xf, if there is any
#                if end_negative < start_positive:
#                    start_positive -= 1
#                    Xf[start_positive] = 0.
#
#                    if end_negative != start_positive:
#                        Xf[end_negative] = 0.
#                        end_negative += 1
#
#                # Find min, max in Xf[start:end_negative]
#                min_feature_value = Xf[start]
#                max_feature_value = min_feature_value
#
#                for p in range(start, end_negative):
#                    current_feature_value = Xf[p]
#
#                    if current_feature_value < min_feature_value:
#                        min_feature_value = current_feature_value
#                    elif current_feature_value > max_feature_value:
#                        max_feature_value = current_feature_value
#
#                # Update min, max given Xf[start_positive:end]
#                for p in range(start_positive, end):
#                    current_feature_value = Xf[p]
#
#                    if current_feature_value < min_feature_value:
#                        min_feature_value = current_feature_value
#                    elif current_feature_value > max_feature_value:
#                        max_feature_value = current_feature_value
#
#                if max_feature_value <= min_feature_value + FEATURE_THRESHOLD:
#                    features[f_j] = features[n_total_constants]
#                    features[n_total_constants] = current.feature
#
#                    n_found_constants += 1
#                    n_total_constants += 1
#
#                else:
#                    f_i -= 1
#                    features[f_i], features[f_j] = features[f_j], features[f_i]
#
#                    # Draw a random threshold
#                    current.threshold = rand_uniform(min_feature_value,
#                                                     max_feature_value,
#                                                     random_state)
#
#                    if current.threshold == max_feature_value:
#                        current.threshold = min_feature_value
#
#                    # Partition
#                    current.pos = self._partition(current.threshold,
#                                                  end_negative,
#                                                  start_positive,
#                                                  start_positive +
#                                                  (Xf[start_positive] == 0.))
#
#                    # Reject if min_samples_leaf is not guaranteed
#                    if (((current.pos - start) < min_samples_leaf) or
#                            ((end - current.pos) < min_samples_leaf)):
#                        continue
#
#                    # Evaluate split
#                    self.criterion.reset()
#                    self.criterion.update(current.pos)
#
#                    # Reject if min_weight_leaf is not satisfied
#                    if ((self.criterion.weighted_n_left < min_weight_leaf) or
#                            (self.criterion.weighted_n_right < min_weight_leaf)):
#                        continue
#
#                    current_proxy_improvement = self.criterion.proxy_impurity_improvement()
#
#                    if current_proxy_improvement > best_proxy_improvement:
#                        best_proxy_improvement = current_proxy_improvement
#                        current.improvement = self.criterion.impurity_improvement(impurity)
#
#                        self.criterion.children_impurity(&current.impurity_left,
#                                                         &current.impurity_right)
#                        best = current
#
#        # Reorganize into samples[start:best.pos] + samples[best.pos:end]
#        if best.pos < end:
#            if current.feature != best.feature:
#                self.extract_nnz(best.feature, &end_negative, &start_positive,
#                                 &is_samples_sorted)
#
#                self._partition(best.threshold, end_negative, start_positive,
#                                best.pos)
#
#            self.criterion.reset()
#            self.criterion.update(best.pos)
#            best.improvement = self.criterion.impurity_improvement(impurity)
#            self.criterion.children_impurity(&best.impurity_left,
#                                             &best.impurity_right)
#
#        # Respect invariant for constant features: the original order of
#        # element in features[:n_known_constants] must be preserved for sibling
#        # and child nodes
#        memcpy(features, constant_features, sizeof(SIZE_t) * n_known_constants)
#
#        # Copy newly found constant features
#        memcpy(constant_features + n_known_constants,
#               features + n_known_constants,
#               sizeof(SIZE_t) * n_found_constants)
#
#        # Return values
#        split[0] = best
#        n_constant_features[0] = n_total_constants
#        return 0
