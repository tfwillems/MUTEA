import collections
import numpy
import random
from math import exp,log

import matrix_optimizer

def compute_median(values, counts):
    total_count = sum(counts)
    target_val  = total_count/2.0
    for i in xrange(len(values)):
        if counts[i] > target_val:
            return values[i]
        target_val -= counts[i]
    return values[-1]

class ReadCountDistribution:
    def __init__(self, counts, probs):
        self.counts = counts
        self.probs  = probs

    def random_read_count(self):
        rand_val = random.random()
        for i in xrange(len(self.counts)):
            if rand_val < self.probs[i]:
                return self.counts[i]
            rand_val -= self.probs[i]
        exit("ERROR: Read count probabilites don't sum to 1")

    def __str__(self):
        return "READCOUNTDISTRIBUTION: counts=" + str(self.counts) + ", probs=" + str(self.probs)

def import_read_count_dists(filename):
    dists = []
    data  = open(filename, "r")
    for line in data:
        tokens = line.strip().split()
        nreads = map(int, tokens[0].split(","))
        nsamps = numpy.array(map(int, tokens[1].split(",")))
        probs  = nsamps*1.0/nsamps.sum()
        dists.append(ReadCountDistribution(nreads, probs))
    data.close()
    return dists

def subsample_str_gts(str_gts, nsamples):
    if nsamples > len(str_gts):
        exit("Requested number of samples exceeds the number of unique samples with STR genotypes")
    return dict(random.sample(str_gts.items(), nsamples))

def simulate(tree, mut_model, len_to_tmrca, nsamples, 
             stutter_model, read_count_dist, genotyper, debug=False, valid_samples=None, root_allele=0):
    str_gts, node_gts = {}, {}
    optimizer = matrix_optimizer.MATRIX_OPTIMIZER(mut_model.trans_matrix, mut_model.min_n)
    optimizer.precompute_results()

    # Simulate real STR genotypes on the tree using the mutation model
    for node in tree.preorder_node_iter():
        if debug:
            print("Processing node %s"%(node.oid))
        if node.parent_node is None:
            node_gts[node.oid] = root_allele

        cur_gt = node_gts[node.oid]

        if node.is_leaf():
            sample = node.taxon.label
            str_gts[sample] = node_gts[node.oid]
        else: 
            new_alleles = []
            times = []
            for child in node.child_nodes():
                tmrca       = int(child.edge_length*len_to_tmrca)
                child_probs = optimizer.get_forward_str_probs(cur_gt, tmrca)
                cum_probs   = numpy.cumsum(child_probs)
                index       = 0
                rand_val    = random.random()
                while index < len(cum_probs) and rand_val > cum_probs[index]:
                    index += 1
                if index == len(cum_probs):
                    exit("ERROR: Invalid cumulative probability distribution for transitions")
                child_allele = index + mut_model.min_n
                node_gts[child.oid] = child_allele
                times.append(tmrca)
                new_alleles.append(child_allele)
            if debug:
                msg = ("Parent allele=%d\t"%(cur_gt)) + "\t".join(map(lambda x: "TMRCA=%d,Allele=%d"%(times[x], new_alleles[x]), xrange(len(node.child_nodes()))))
                print(msg)

    # Compute the genotype frequencies
    gt_freqs = collections.defaultdict(int)
    for sample in str_gts:
        gt_freqs[str_gts[sample]] += 1
    num_gts = 1.0*sum(gt_freqs.values())
    for allele in gt_freqs:
        gt_freqs[allele] /= num_gts

    # Subsample to include only valid samples
    if valid_samples is not None:
        str_gts = dict(filter(lambda x: x[0] in valid_samples, str_gts.items()))
    
    # Subsample based on number of observed samples
    if nsamples is not None:
        str_gts = subsample_str_gts(str_gts, nsamples)

    # Utilize stutter model and read count distribution to simulate the reads observed for each sample
    sample_read_counts = {}
    for sample,gt in str_gts.items():
        num_reads   = read_count_dist.random_read_count()
        read_counts = collections.defaultdict(int) 
        for j in xrange(num_reads):
            stutter_size = stutter_model.random_stutter_size()
            read_counts[gt+stutter_size] += 1
        sample_read_counts[sample] = read_counts
        
    # Use minimum and maximum observed allele sizes
    min_allele = min(map(lambda x: min(x.keys()), sample_read_counts.values()))
    max_allele = max(map(lambda x: max(x.keys()), sample_read_counts.values()))

    # Train the genotyper
    trained = genotyper.train(sample_read_counts, min_allele, max_allele)
    if not trained:
        return None, None, genotyper

    # Compute genotype posteriors using genotyper
    gt_posteriors = {}
    for sample,counts in sample_read_counts.items():
        gt_posteriors[sample] = genotyper.get_genotype_posteriors(counts, str_gts[sample], stutter_model)

    # Compute 'central' allele using the allele with the median posterior sum
    gt_counts = collections.defaultdict(int)
    for posteriors in gt_posteriors.values():
        for gt,prob in posteriors.items():
            gt_counts[gt] += prob
    count_items = sorted(gt_counts.items())
    center      = compute_median(map(lambda x: x[0], count_items), map(lambda x: x[1], count_items))
    print("CENTRAL ALLELE = %d"%(center))
    
    # Normalize all genotypes relative to this central allele
    norm_gt_posteriors = {}
    for sample,posteriors in gt_posteriors.items():
        new_posteriors = {}
        for gt,prob in posteriors.items():
            new_posteriors[gt-center] = prob
        norm_gt_posteriors[sample] = new_posteriors

    return norm_gt_posteriors, genotyper




