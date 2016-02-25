import argparse
import collections
import dendropy
import gzip
import math
import numpy
import operator
import os
import os.path
import random
import vcf
from cStringIO import StringIO

from matrix_optimizer import MATRIX_OPTIMIZER
from mutation_model import OUGeomSTRMutationModel
from main import optimize_loglikelihood, read_tree, compute_node_order, determine_allele_range, tree_depths
import read_powerplex

import tree_posterior_inference
import merge_1kg_capillary_calls

from scipy.misc import logsumexp

def check_args(args, arg_lst, option):
    var_dict = vars(args)
    for arg in arg_lst:
        if var_dict[arg] is None:
            exit("Argument --%s is required for --%s option. Exiting..."%(arg, option))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pplex",     required=True,  dest="powerplex", type=str,   default=None,    help="File containg powerplex STR genotypes")
    parser.add_argument("--tree",      required=True,  dest="tree",      type=str,                    help="File containing phylogeny for samples in newick format")
    parser.add_argument("--min_mu",    required=False, dest="min_mu",    type=float, default=0.00001, help="Lower optimization boundary for mu")
    parser.add_argument("--max_mu",    required=False, dest="max_mu",    type=float, default=0.05,    help="Upper optimization boundary for mu")
    parser.add_argument("--min_pgeom", required=False, dest="min_pgeom", type=float, default=0.5,     help="Lower optimization boundary for pgeom")
    parser.add_argument("--max_pgeom", required=False, dest="max_pgeom", type=float, default=1.0,     help="Upper optimization boundary for pgeom")
    parser.add_argument("--min_beta",  required=False, dest="min_beta",  type=float, default=0.0,     help="Lower optimization boundary for beta")
    parser.add_argument("--max_beta",  required=False, dest="max_beta",  type=float, default=0.75,    help="Upper optimization boundary for beta")
    parser.add_argument("--locus",     required=True,  dest="locus",     type=str,                    help="Name of locus to impute")
    parser.add_argument("--out",       required=True,  dest="out",       type=str,                    help="Output file to which results are written")
    parser.add_argument("--sample",    required=False, dest="sample",    type=str,                    help="Sample whose profile is to be imputed")
    
    args = parser.parse_args()

    output            = open(args.out, "w")
    max_tmrca         = 2800   # TMRCA of all men in phylogeny (in generations)
    min_node_conf     = 0      # Remove nodes with conf < threshold
    tree,haplogroups  = read_tree(args.tree, min_node_conf)
    node_lst          = compute_node_order(tree)
    _,_,median_depth  = tree_depths(tree)
    len_to_tmrca      = max_tmrca/median_depth
    powerplex_gt_dict = read_powerplex.read_genotypes(args.powerplex, conv_to_likelihood=True)
    impute_locus(args, tree, node_lst, max_tmrca, len_to_tmrca, powerplex_gt_dict, output)
    output.close()


def imputed_profile_match(args, tree, node_lst, max_tmrca, len_to_tmrca, powerplex_gt_dict, output):
    sample_set   = set(reduce(lambda x,y: x+y, map(lambda x: x.keys(), powerplex_gt_dict.values())))
    bad_samples  = set()
    for sample in sample_set:
        for locus in powerplex_gt_dict:
            if sample not in powerplex_gt_dict[locus]:
                bad_samples.add(sample)
                break
    for sample in bad_samples:
        sample_set.remove(sample)
        for locus in powerplex_gt_dict:
            if sample in powerplex_gt_dict[locus]:
                del powerplex_gt_dict[locus][sample]

    # Avoid processing sample if it was missing a genotype for 1 or more loci
    if args.sample not in sample_set:
        return

    # Initialize match probabilities
    log_match_probs = {}
    for sample in sample_set:
        log_match_probs[sample] = 0

    # Compute match probabilities for each locus
    for locus in powerplex_gt_dict:
        imp_data = {args.sample : powerplex_gt_dict[locus][args.sample]}
        ref_data = powerplex_gt_dict[locus]
        del ref_data[args.sample]

        # Recompute median using only reference samples
        if len(ref_data) %2 == 0:
            center = int(numpy.median(map(lambda x: x.keys()[0], ref_data.values()[1:])))
        else:
            center = int(numpy.median(map(lambda x: x.keys()[0], ref_data.values())))

        # Normalize both data dictionaries
        new_ref_data = {}
        for sample in ref_data:
            new_ref_data[sample] = {}
            for gt in ref_data[sample]:
                new_ref_data[sample][gt-center] = ref_data[sample][gt]
        new_imp_data = {}
        for sample in imp_data:
            new_imp_data[sample] = {}
            for gt in imp_data[sample]:
                new_imp_data[sample][gt-center] = imp_data[sample][gt]
        ref_data = new_ref_data
        imp_data = new_imp_data

        print("Estimating mutation parameters for %s"%(locus))
        str_gts = ref_data
        min_str = min(reduce(lambda x,y: x+y, map(lambda x: x.keys(), str_gts.values())))
        max_str = max(reduce(lambda x,y: x+y, map(lambda x: x.keys(), str_gts.values())))
        opt_res = optimize_loglikelihood(tree, len_to_tmrca, max_tmrca,
                                         str_gts, min_str, max_str, node_lst,
                                         min_mu=args.min_mu,       max_mu=args.max_mu,
                                         min_beta=args.min_beta,   max_beta=args.max_beta,
                                         min_pgeom=args.min_pgeom, max_pgeom=args.max_pgeom, num_iter=3)
        mu    = numpy.power(10.0, opt_res.x[0])
        beta  = opt_res.x[1]
        pgeom = opt_res.x[2]
             
        print("Determining allele range")
        allele_range, max_step = determine_allele_range(max_tmrca, mu, beta, pgeom, min_str, max_str)

        print("Constructing mutation model")
        mut_model = OUGeomSTRMutationModel(pgeom, mu, beta, allele_range)

        print("Inferring posteriors")
        node_log_posteriors, estimated_gt_probs = tree_posterior_inference.compute_node_posteriors(tree, mut_model, ref_data, allele_range, len_to_tmrca) 

        # Update the match probabilities for each sample
        sample_gt_probs = estimated_gt_probs[args.sample]
        for sample in sample_set:
            if sample == args.sample:
                match_prob = sample_gt_probs[imp_data[sample].keys()[0]]
            else:
                match_prob = sample_gt_probs[ref_data[sample].keys()[0]]
            log_match_probs[sample] += numpy.log10(match_prob)

    # Print out match probabilities of each sample with the target sample
    for sample in sorted(list(sample_set)):
        output.write("%s\t%s\t%f\n"%(args.sample, sample, log_match_probs[sample]))
    output.flush()
    os.fsync(output.fileno())


def impute_locus(args, tree, node_lst, max_tmrca, len_to_tmrca, powerplex_gt_dict, output):
    num_iter          = 100
    num_imp_samples   = 70
    for locus in powerplex_gt_dict:
        if locus != args.locus:
            continue

        invalid_imp_samples = set()
        all_samples         = set(powerplex_gt_dict[locus].keys())
        for i in xrange(num_iter):
            imp_samples = set(random.sample([samp for samp in all_samples if samp not in invalid_imp_samples], num_imp_samples))
            ref_samples = set([samp for samp in all_samples if samp not in imp_samples])
            all_data    = powerplex_gt_dict[locus]
            ref_data    = {}
            imp_data    = {}
            for sample in all_data:
                if sample in ref_samples:
                    ref_data[sample] = all_data[sample]
                else:
                    imp_data[sample] = all_data[sample]

            # Recompute median using only reference samples
            if len(ref_data) %2 == 0:
                center = int(numpy.median(map(lambda x: x.keys()[0], ref_data.values()[1:])))
            else:
                center = int(numpy.median(map(lambda x: x.keys()[0], ref_data.values())))

            # Normalize both data dictionaries
            new_ref_data = {}
            for sample in ref_data:
                new_ref_data[sample] = {}
                for gt in ref_data[sample]:
                    new_ref_data[sample][gt-center] = ref_data[sample][gt]
            new_imp_data = {}
            for sample in imp_data:
                new_imp_data[sample] = {}
                for gt in imp_data[sample]:
                    new_imp_data[sample][gt-center] = imp_data[sample][gt]
            ref_data = new_ref_data
            imp_data = new_imp_data


            print("Estimating mutation parameters for %s"%(locus))
            str_gts = ref_data
            min_str = min(reduce(lambda x,y: x+y, map(lambda x: x.keys(), str_gts.values())))
            max_str = max(reduce(lambda x,y: x+y, map(lambda x: x.keys(), str_gts.values())))
            opt_res = optimize_loglikelihood(tree, len_to_tmrca, max_tmrca,
                                             str_gts, min_str, max_str, node_lst,
                                             min_mu=args.min_mu,       max_mu=args.max_mu,
                                             min_beta=args.min_beta,   max_beta=args.max_beta,
                                             min_pgeom=args.min_pgeom, max_pgeom=args.max_pgeom, num_iter=3)
            mu    = numpy.power(10.0, opt_res.x[0])
            beta  = opt_res.x[1]
            pgeom = opt_res.x[2]
             
            print("Determining allele range")
            allele_range, max_step = determine_allele_range(max_tmrca, mu, beta, pgeom, min_str, max_str)

            print("Constructing mutation model")
            mut_model = OUGeomSTRMutationModel(pgeom, mu, beta, allele_range)

            print("Inferring posteriors")
            node_log_posteriors, estimated_gt_probs = tree_posterior_inference.compute_node_posteriors(tree, mut_model, ref_data, allele_range, len_to_tmrca) 

            for sample in estimated_gt_probs:
                if sample not in imp_data:
                    continue
                true_gt          = imp_data[sample].keys()[0]
                est_gt, est_prob = sorted(estimated_gt_probs[sample].items(), key = lambda x: x[1])[-1]
                est_dosage       = sum(map (lambda x: x[0]*x[1], estimated_gt_probs[sample].items()))
                output.write("%s\t%s\t%d\t%d\t%f\t%f\n"%(locus, sample, true_gt, est_gt, est_prob, est_dosage))
            output.flush()
            os.fsync(output.fileno())

if __name__ == "__main__":
    main()
