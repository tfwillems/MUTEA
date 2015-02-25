import collections
import numpy
from scipy.misc import logsumexp
from stutter_model import StutterModel

max_iter        = 100
min_eff_cov     = 100
CONVERGED       = 0
COVERAGE_LIMIT  = 1
ITERATION_LIMIT = 2

def compute_LL(sample_read_counts, motif_len, gt_freqs, stutter_model):
    # Determine the most common frame across all samples
    frame_counts = collections.defaultdict(int)
    tot_read_counts = collections.defaultdict(int)
    for sample,read_counts in sample_read_counts.items():
        for read,count in read_counts.items():
            frame  = read%motif_len
            frame += 0 if frame > 0 else motif_len
            frame_counts[frame] += count
            tot_read_counts[read] += count
    best_frame = sorted(frame_counts.items(), key = lambda x: x[1])[-1][0]

    # Filter any samples with out-of-frame reads
    valid_read_counts = []
    tot_read_counts   = collections.defaultdict(int)
    allele_set        = set([])
    for sample,read_counts in sample_read_counts.items():
        all_in_frame = True
        for read,count in read_counts.items():
            frame  = read%motif_len
            frame += 0 if frame > 0 else motif_len
            if frame != best_frame:
                all_in_frame = False
                break
        if all_in_frame:
            valid_read_counts.append(read_counts)
            for read,count in read_counts.items():
                allele_set.add(read)
                tot_read_counts[read] += count

    allele_sizes   = sorted(list(allele_set)) # Size of allele for each index
    allele_indices = dict(map(reversed, enumerate(allele_sizes)))
    eff_coverage   = 0  # Effective number of reads informative for stutter inference
    read_counts    = [] # Array of dictionaries, where key = allele index and count = # such reads for a sample
    max_stutter    = 0
    for i in xrange(len(valid_read_counts)):
        sorted_sizes  = sorted(valid_read_counts[i].keys())
        max_stutter   = max(max_stutter, sorted_sizes[-1]-sorted_sizes[0])
        count_dict    = dict([(allele_indices[x[0]], x[1]) for x in valid_read_counts[i].items()])
        eff_coverage += sum(valid_read_counts[i].values())-1
        read_counts.append(count_dict)
    stutter_size = min(5, max_stutter)
    num_stutters = 2*stutter_size + 1

    log_stutter_probs = []
    for size in xrange(-stutter_size, stutter_size+1):
        log_stutter_probs.append(stutter_model.get_log_stutter_size_prob(size))
    log_stutter_probs = numpy.array(log_stutter_probs)-logsumexp(log_stutter_probs)

    log_gt_priors = []
    for i in xrange(len(allele_sizes)):
        log_gt_priors.append(numpy.log(gt_freqs[allele_sizes[i]]))
    log_gt_priors = numpy.array(log_gt_priors)-logsumexp(log_gt_priors)

    print(numpy.exp(log_gt_priors))
    print(numpy.exp(log_stutter_probs))

    nalleles  = len(allele_sizes)
    LL        = calc_log_likelihood(log_gt_priors, log_stutter_probs, read_counts, nalleles, num_stutters, allele_sizes)
    return LL




def run_EM(sample_read_counts, motif_len, debug=False):
    # Determine the most common frame across all samples
    frame_counts = collections.defaultdict(int)
    tot_read_counts = collections.defaultdict(int)
    for sample,read_counts in sample_read_counts.items():
        for read,count in read_counts.items():
            frame  = read%motif_len
            frame += 0 if frame > 0 else motif_len
            frame_counts[frame] += count
            tot_read_counts[read] += count
    best_frame = sorted(frame_counts.items(), key = lambda x: x[1])[-1][0]
    if debug:
        print(tot_read_counts)

    # Filter any samples with out-of-frame reads
    valid_read_counts = []
    tot_read_counts   = collections.defaultdict(int)
    allele_set        = set([])
    for sample,read_counts in sample_read_counts.items():
        all_in_frame = True
        for read,count in read_counts.items():
            frame  = read%motif_len
            frame += 0 if frame > 0 else motif_len
            if frame != best_frame:
                all_in_frame = False
                break
        if all_in_frame:
            valid_read_counts.append(read_counts)
            for read,count in read_counts.items():
                allele_set.add(read)
                tot_read_counts[read] += count
    if debug:
        print(tot_read_counts)

    allele_sizes   = sorted(list(allele_set)) # Size of allele for each index
    allele_indices = dict(map(reversed, enumerate(allele_sizes)))
    eff_coverage   = 0  # Effective number of reads informative for stutter inference
    read_counts    = [] # Array of dictionaries, where key = allele index and count = # such reads for a sample
    max_stutter    = 0
    for i in xrange(len(valid_read_counts)):
        sorted_sizes  = sorted(valid_read_counts[i].keys())
        max_stutter   = max(max_stutter, sorted_sizes[-1]-sorted_sizes[0])
        count_dict    = dict([(allele_indices[x[0]], x[1]) for x in valid_read_counts[i].items()])
        eff_coverage += sum(valid_read_counts[i].values())-1
        read_counts.append(count_dict)        
    num_stutters = 1 + 2*min(5, max_stutter) # Number of stutter options considered [-n, -n+1, ..., 0, ..., n-1, n]

    # Check that we have sufficient reads to perform the inference
    if eff_coverage < min_eff_cov:
        return COVERAGE_LIMIT, None, None, eff_coverage, None

    if debug:
        print("READ COUNTS:", read_counts)

    # Initialize parameters
    nalleles          = len(allele_sizes)
    log_gt_priors     = init_log_gt_priors(read_counts, nalleles)
    log_stutter_probs = init_log_stutter_probs(read_counts, num_stutters, allele_sizes)
    if debug:
        print(numpy.exp(log_gt_priors))
        print(numpy.exp(log_stutter_probs))

    # Perform EM iterative procedure until convergence
    converged = False
    prev_LL   = -100000000.0
    niter     = 0
    while not converged and niter < max_iter:
        # Recalculate posteriors
        gt_posteriors = recalc_gt_posteriors(log_gt_priors, log_stutter_probs, read_counts, nalleles, num_stutters, allele_sizes)

        # Reestimate parameters
        log_stutter_probs = recalc_log_stutter_probs(gt_posteriors, read_counts, nalleles, num_stutters, allele_sizes)
        log_gt_priors     = recalc_log_pop_priors(gt_posteriors)

        if debug:
            print(gt_posteriors)
            print("\n")
            print(numpy.exp(log_stutter_probs))
            print("\n")
            print(numpy.exp(log_gt_priors))
            print("\n")

        # Test for convergence
        new_LL    = calc_log_likelihood(log_gt_priors, log_stutter_probs, read_counts, nalleles, num_stutters, allele_sizes)
        print("EM LL = %f"%(new_LL))
        converged = (-(new_LL-prev_LL)/prev_LL < 0.0001) and (new_LL-prev_LL < 0.0001)
        prev_LL   = new_LL
        niter    += 1

    print(numpy.exp(log_stutter_probs))

    # Return optimized values or placeholders if EM failed
    if niter == max_iter:
        return ITERATION_LIMIT, None, None, eff_coverage, None
    else:
        stutter_probs  = numpy.exp(log_stutter_probs)
        stutter_window = (num_stutters-1)/2
        est_stutter    = StutterModel(range(-stutter_window, stutter_window+1), stutter_probs)
        return CONVERGED, log_gt_priors, est_stutter, eff_coverage, new_LL

def get_stutter_index(gt_index, read_index, num_stutters, allele_sizes):
    stutter_window = (num_stutters-1)/2
    stutter_size   = allele_sizes[read_index] - allele_sizes[gt_index]
    stutter_size   = max(min(stutter_size, stutter_window), -stutter_window)
    return stutter_size + stutter_window

def init_log_gt_priors(read_counts, nalleles):
    gt_counts      = numpy.zeros(nalleles) + 1.0 # Pseudocount
    for counts in read_counts:
        num_reads = sum(counts.values())
        for allele_index,count in counts.items():
            gt_counts[allele_index] += 1.0*count/num_reads
    return numpy.log(1.0*gt_counts/gt_counts.sum())

def init_log_stutter_probs(read_counts, num_stutters, allele_sizes):
    stutter_counts = numpy.zeros(num_stutters) + 1.0
    for counts in read_counts:
        num_reads = sum(counts.values())
        for allele_index,count in counts.items():
            posterior = 1.0*count/num_reads
            for read_index,read_count in counts.items():
                stutter_counts[get_stutter_index(allele_index, read_index, num_stutters, allele_sizes)] += read_count*posterior
    return numpy.log(1.0*stutter_counts/stutter_counts.sum())

def recalc_log_pop_priors(gt_posteriors):
    gt_counts = gt_posteriors.sum(axis=0) + 1.0 # Pseudocount
    return numpy.log(1.0*gt_counts/gt_counts.sum())

def recalc_log_stutter_probs(gt_posteriors, read_counts, nalleles, num_stutters, allele_sizes):
    nsamples       = gt_posteriors.shape[0]
    stutter_counts = numpy.zeros(num_stutters) + 1.0 # Pseudocount
    for i in xrange(nsamples):
        for j in xrange(nalleles):
            posterior = gt_posteriors[i][j]
            for read_index,count in read_counts[i].items():
                stutter_counts[get_stutter_index(j, read_index, num_stutters, allele_sizes)] += count*posterior
    return numpy.log(1.0*stutter_counts/stutter_counts.sum())

def recalc_gt_posteriors(log_gt_priors, log_stutter_probs, read_counts, nalleles, num_stutters, allele_sizes):
    nsamples = len(read_counts)
    LLs      = numpy.zeros((nsamples, nalleles))
    LLs     += log_gt_priors
    for i in xrange(nsamples):
        for j in xrange(nalleles):
            for read_index,count in read_counts[i].items():
                LLs[i][j] += count*log_stutter_probs[get_stutter_index(j, read_index, num_stutters, allele_sizes)]
        sample_LL = logsumexp(LLs[i])
        LLs[i]   -= sample_LL
    return numpy.exp(LLs)

def calc_log_likelihood(log_gt_priors, log_stutter_probs, read_counts, nalleles, num_stutters, allele_sizes):
    nsamples   = len(read_counts)
    sample_LLs = []
    for i in xrange(nsamples):
        LLs  = log_gt_priors
        for j in xrange(nalleles):
            for read_index,count in read_counts[i].items():
                LLs[j] += count*log_stutter_probs[get_stutter_index(j, read_index, num_stutters, allele_sizes)]
        sample_LL = logsumexp(LLs)
        sample_LLs.append(sample_LL)
    return numpy.sum(sample_LLs)
