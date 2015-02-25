import numpy
import mutation_model

from numpy import linalg as LA

def factor_power_two(val):
    power   = 0
    factors = []
    while val != 0:
        if (val & 1) != 0:
            factors.append(power)
        val   >>= 1
        power  += 1
    return factors

def factor(val):
    if val <= 128:
        return val
    else:
        return val & 127, factor_power_two(val & (~127))

# Class designed to accelerate repeated calcuations of transition matrices
# Optimization strategy involves sequentially calculating the matrix for 
# 1-128 generations and for several powers of two. These results are 
# memoized and used to calculate all subsequent transition probabilities
class MATRIX_OPTIMIZER:
    def __init__(self, per_gen_matrix, min_n):
        self.per_gen_matrix = per_gen_matrix
        self.N              = per_gen_matrix.shape[0]
        self.min_n          = min_n

    def precompute_results(self):
        # Compute transition matrices for 0-128 generations
        memoized_matrices = []
        cur_matrix = numpy.matrix(numpy.identity(self.N)) # Crazy bad things happen if this is an array instead of a matrix, but code still runs
        for i in xrange(0, 129):
            memoized_matrices.append(cur_matrix)
            cur_matrix = numpy.dot(cur_matrix, self.per_gen_matrix)
        self.memoized_matrices = memoized_matrices
            
        # Compute powers of 2 up to 2^13
        powers_of_two = []
        cur_matrix    = self.per_gen_matrix
        for i in xrange(1, 20):
            powers_of_two.append(cur_matrix)
            cur_matrix = numpy.dot(cur_matrix, cur_matrix)
        self.powers_of_two = powers_of_two

    def get_transition_matrix(self, num_generations):
        if num_generations <= 128:
            res = self.memoized_matrices[num_generations]
        else:
            v1, powers = factor(num_generations)
            if len(powers) == 0:
                res = self.memoized_matrices[v1]
            else:
                res = numpy.dot(self.memoized_matrices[v1], reduce(lambda x,y: numpy.dot(x,y), map(lambda x: self.powers_of_two[x], powers)))
        return numpy.clip(res, 1e-10, 1.0)

    def get_forward_str_probs(self, start_allele, num_generations):
        vec = numpy.zeros((self.N, 1))
        vec[start_allele-self.min_n] = 1
        return numpy.array(self.get_transition_matrix(num_generations).dot(vec).transpose())[0]


def main():
    allele_range = 3
    start_allele = 0
    mu           = 0.0001
    beta         = 0.5
    p_geom       = 1.0
    mut_model    = mutation_model.OUGeomSTRMutationModel(p_geom, mu, beta, allele_range)
    optimizer    = MATRIX_OPTIMIZER(mut_model.trans_matrix, mut_model.min_n)
    optimizer.precompute_results()
    other_model  = mutation_model.OUGeomSTRMutationModel(p_geom, mu, beta, allele_range)
    
    numpy.set_printoptions(precision=4, linewidth=150)
    for i in [128, 156, 240]:
        print(i)
        print(optimizer.get_transition_matrix(i))
        other_model.compute_forward_matrix(i, allow_bigger_err=True)
        print(other_model.forward_matrix)
        print(other_model.get_forward_str_probs(0, i))
        print(optimizer.get_forward_str_probs(0, i))



if __name__ == "__main__":
    main()
