import collections
import numpy
from scipy.misc  import logsumexp
from scipy.stats import geom

min_iter        = 40
max_iter        = 1000
min_eff_cov     = 1
CONVERGED       = 0
COVERAGE_LIMIT  = 1
ITERATION_LIMIT = 2

def run_EM(sample_read_counts, debug=False):
    valid_read_counts = sample_read_counts.values()
    allele_set        = set(reduce(lambda x,y:x+y, map(lambda x: x.keys(), valid_read_counts)))
    allele_sizes      = sorted(list(allele_set)) # Size of allele for each index
    allele_indices    = dict(map(reversed, enumerate(allele_sizes)))
    eff_coverage      = 0  # Effective number of reads informative for stutter inference
    read_counts       = [] # Array of dictionaries, where key = allele index and count = # such reads for a sample
    max_stutter       = 0
    for i in xrange(len(valid_read_counts)):
        sorted_sizes  = sorted(valid_read_counts[i].keys())
        max_stutter   = max(max_stutter, sorted_sizes[-1]-sorted_sizes[0])
        count_dict    = dict([(allele_indices[x[0]], x[1]) for x in valid_read_counts[i].items()])
        eff_coverage += sum(valid_read_counts[i].values())-1
        read_counts.append(count_dict)        
    num_stutters = 1 + 2*min(5, max_stutter) # Number of stutter options considered [-n, -n+1, ..., 0, ..., n-1, n]

    # Check that we have sufficient reads to perform the inference
    if eff_coverage < min_eff_cov:
        return COVERAGE_LIMIT, eff_coverage, None, None, None, None

    # Initialize parameters
    nalleles         = len(allele_sizes)
    nsamples         = len(read_counts)
    log_gt_priors    = init_log_gt_priors(read_counts, nalleles)
    down, up, p_geom = init_stutter_params(read_counts, allele_sizes)
    print("INIT: P_GEOM=%f, DOWN=%f, UP=%f"%(p_geom, down, up))
    if debug:
        print(numpy.exp(log_gt_priors))

    # Construct read count array
    read_counts_array = numpy.zeros((nsamples, nalleles))
    for sample_index,counts in enumerate(read_counts):
        for read_index,count in counts.items():
            read_counts_array[sample_index][read_index] += count

    # Perform EM iterative procedure until convergence
    converged = False
    prev_LL   = -100000000.0
    niter     = 0
    while niter < min_iter or (not converged and niter < max_iter):
        # Recalculate posteriors
        log_gt_posteriors = recalc_log_gt_posteriors(log_gt_priors, down, up, p_geom, read_counts_array, nalleles, allele_sizes)
        if debug:
            print("POSTERIORS:")
            print(numpy.exp(log_gt_posteriors))

        # Reestimate parameters
        log_gt_priors    = recalc_log_pop_priors(log_gt_posteriors)
        down, up, p_geom = recalc_stutter_params(log_gt_posteriors, read_counts, nalleles, allele_sizes)
        if debug:
            print("POP PRIORS: %s"%(str(numpy.exp(log_gt_priors))))

        if down < 0 or down > 1 or up < 0 or up > 1 or p_geom < 0 or p_geom > 1:
            exit("ERROR: Invalid paramter(s) during EM iteration: DOWN=%f, UP=%f, P_GEOM=%f"%(down, up, p_geom))
            
        # Test for convergence
        if niter % 4 == 0:
            new_LL    = calc_log_likelihood(log_gt_priors, down, up, p_geom, read_counts_array, nalleles, allele_sizes)
            print("ITERATION #%d, EM LL=%f, P_GEOM=%f, DOWN=%f, UP=%f"%(niter+1, new_LL, p_geom, down, up))
            converged = (-(new_LL-prev_LL)/prev_LL < 0.0001) and (new_LL-prev_LL < 0.0001)
            prev_LL   = new_LL
        niter    += 1

    print("PRIORS = %s"%(numpy.exp(log_gt_priors)))

    # Return optimized values or placeholders if EM failed
    if niter == max_iter:
        return ITERATION_LIMIT, eff_coverage, down, up, p_geom, new_LL
    else:
        return CONVERGED, eff_coverage, down, up, p_geom, new_LL

def init_log_gt_priors(read_counts, nalleles):
    gt_counts = numpy.zeros(nalleles) + 1.0 # Pseudocount
    for counts in read_counts:
        num_reads = sum(counts.values())
        for allele_index,count in counts.items():
            gt_counts[allele_index] += 1.0*count/num_reads
    return numpy.log(1.0*gt_counts/gt_counts.sum())

def init_stutter_params(read_counts, allele_sizes):
    dir_counts = numpy.array([1.0, 1.0, 1.0]) # Pseudocounts
    diff_sum = 3.0                            # Step sizes of 1 and 2, so that p_geom < 1 
    for counts in read_counts:
        num_reads = sum(counts.values())
        for allele_index,count in counts.items():
            posterior = 1.0*count/num_reads
            for read_index,read_count in counts.items():
                dir_counts[numpy.sign(read_index-allele_index)+1] += posterior*read_count
                diff_sum += read_count*posterior*abs(allele_sizes[read_index]-allele_sizes[allele_index])
    tot_dir_count = sum(dir_counts)
    return dir_counts[0]/tot_dir_count, dir_counts[2]/tot_dir_count, 1.0*(dir_counts[0]+dir_counts[2])/diff_sum

def recalc_log_pop_priors(log_gt_posteriors):
    log_counts = logsumexp(log_gt_posteriors, axis=0)
    return log_counts - logsumexp(log_counts)

def recalc_stutter_params(log_gt_posteriors, read_counts, nalleles, allele_sizes):
    nsamples   = log_gt_posteriors.shape[0]
    log_counts = [[0], [0], [0]]   # Pseudocounts
    log_diffs  = [0, numpy.log(2)] # Step sizes of 1 and 2, so that p_geom < 1 
    for i in xrange(nsamples):
        for j in xrange(nalleles):
            log_post = log_gt_posteriors[i][j]
            for read_index,count in read_counts[i].items():
                log_count = numpy.log(count)
                diff      = allele_sizes[read_index] - allele_sizes[j] 
                if diff != 0:
                    log_diffs.append(log_count + log_post + numpy.log(abs(diff)))
                log_counts[numpy.sign(diff)+1].append(log_post + log_count)
    log_tot_counts = map(logsumexp, log_counts)
    p_hat          = numpy.exp(logsumexp([log_tot_counts[0], log_tot_counts[2]]) - logsumexp(log_diffs))
    log_freqs      = log_tot_counts - logsumexp(log_tot_counts)
    return numpy.exp(log_freqs[0]), numpy.exp(log_freqs[2]), p_hat

def recalc_log_gt_posteriors(log_gt_priors, down, up, p_geom, read_counts_array, nalleles, allele_sizes):
    stutter_dist = geom(p_geom)
    nsamples     = read_counts_array.shape[0]
    LLs          = numpy.zeros((nsamples, nalleles)) + log_gt_priors
    log_down, log_eq, log_up = map(numpy.log, [down, 1-down-up, up])
    for j in xrange(nalleles):
        step_probs = numpy.hstack(([log_down + stutter_dist.logpmf(abs(allele_sizes[x]-allele_sizes[j])) for x in range(0, j)],
                                   [log_eq],
                                   [log_up   + stutter_dist.logpmf(abs(allele_sizes[x]-allele_sizes[j])) for x in range(j+1, nalleles)]))
        LLs [:,j] += numpy.sum(read_counts_array*step_probs, axis=1)
    log_samp_totals = logsumexp(LLs, axis=1)[numpy.newaxis].T
    return LLs - log_samp_totals

def calc_log_likelihood(log_gt_priors, down, up, p_geom, read_counts_array, nalleles, allele_sizes):
    stutter_dist = geom(p_geom)
    nsamples     = read_counts_array.shape[0]
    total_LL     = 0.0
    LLs          = numpy.zeros((nsamples, nalleles)) + log_gt_priors
    log_down, log_eq, log_up = map(numpy.log, [down, 1-down-up, up])
    for j in xrange(nalleles):
         step_probs = numpy.hstack(([log_down + stutter_dist.logpmf(abs(allele_sizes[x]-allele_sizes[j])) for x in range(0, j)],
                                    [log_eq],
                                    [log_up   + stutter_dist.logpmf(abs(allele_sizes[x]-allele_sizes[j])) for x in range(j+1, nalleles)]))
         LLs [:,j] += numpy.sum(read_counts_array*step_probs, axis=1)
    return numpy.sum(logsumexp(LLs, axis=1))
