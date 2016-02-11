import dendropy
from dendropy import treecalc
import collections
import math
import numpy
import operator
import random
import scipy.stats
import sys
import vcf

'''
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm
'''

import scipy.optimize
import argparse
import os

from   mutation_model import OUGeomSTRMutationModel
import read_powerplex
import read_str_vcf
import stutter_em
import geom_stutter_em
import stutter_model
import genotypers
import simulate_strs
import mutation_model
import matrix_optimizer
import stutter_info

# Useful for studying time consumption
#from pycallgraph import PyCallGraph
#from pycallgraph.output import GraphvizOutput

def read_tree(input_file, min_conf):
    tree = dendropy.Tree(stream=open(input_file), schema="newick")
    # Count initial number of leaves
    leaf_count = 0
    for node in tree.postorder_node_iter():
        if node.is_leaf():
            leaf_count += 1
    print("# leaves before pruning = %d"%leaf_count)
    
    # Prune low confidence nodes and their subtrees
    for node in tree.postorder_node_iter():
        if not node.is_leaf() and node.label is not None and node.parent_node is not None:
            if int(node.label) < min_conf:
                tree.prune_subtree(node)
        continue

    haplogroups       = {}
    #'''
    haplogroup_counts = {}
    # Remove portion of label describing sample's haplogroup
    for leaf in tree.leaf_nodes():
        hapgroup = leaf.taxon.label.split(" ")[1]
        leaf.taxon.label = leaf.taxon.label.split(" ")[0]
        haplogroups[leaf.taxon.label] = hapgroup
        if hapgroup not in haplogroup_counts:
            haplogroup_counts[hapgroup] = 1
        else:
            haplogroup_counts[hapgroup] += 1
    #'''

    # Remove samples with unusual distance from root
    for label in ["HG02982", "HG01890", "HG02040", "HG00628", "NA19384"]:
        tree.prune_subtree(tree.find_node_with_taxon_label(label))

    # Count number of leaves post pruning
    leaf_count = 0
    for node in tree.postorder_node_iter():
        if node.is_leaf():
            leaf_count += 1
    print("# leaves after pruning  = %d"%leaf_count)
    return tree, haplogroups

def num_meioses(tree, len_to_tmrca):
    sum_len = 0
    for edge in tree.postorder_edge_iter():
        if edge.length is not None:
            sum_len += edge.length
    return sum_len*len_to_tmrca


def max_edge_length(tree):
    max_len = 0
    for edge in tree.postorder_edge_iter():
        max_len = max(max_len, edge.length)
    return max_len

def tree_depths(tree):
    min_depth = 10000000000
    max_depth = 0
    depths    = []
    for node in tree.leaf_nodes():
        dist = node.distance_from_root()
        min_depth = min(min_depth, dist)
        max_depth = max(max_depth, dist)
        depths.append(dist)
    return min_depth,max_depth,numpy.median(depths)

def determine_allele_range(max_tmrca, mu, beta, p_geom,
                           min_obs_allele, max_obs_allele,
                           min_possible_range=1, max_possible_range=200,
                           max_leakage=1e-5, debug=False):
    if debug:
        print("Determining allele range for MAX_TMRCA=%d, mu=%f, beta=%f, p=%f"%(max_tmrca, mu, beta, p_geom))
    min_possible_range = max(min_possible_range, max(-min_obs_allele, max_obs_allele))

    for allele_range in xrange(min_possible_range, max_possible_range):
        mut_model = OUGeomSTRMutationModel(p_geom, mu, beta, allele_range)
        trans_mat = mut_model.trans_matrix**max_tmrca
        leakages  = []
        for allele in min_obs_allele, max_obs_allele:
            vec       = numpy.zeros((mut_model.N,1))
            vec[allele-mut_model.min_n] = 1
            prob = numpy.array(trans_mat.dot(vec).transpose())[0]
            leakages.append(prob[0]+prob[-1])
        if leakages[0] < max_leakage and leakages[1] < max_leakage:
            return allele_range, mut_model.max_step
        if debug:
            print(allele_range, prob_a[0], prob_b[0])
    exit("Unable to find an allele range with leakage < the provided bounds and < the specified maximum")

def determine_allele_range_from_seed(max_tmrca, mu, beta, p_geom,
                                     min_obs_allele, max_obs_allele,
                                     seed, max_possible_range=200,
                                     max_leakage=1e-5, debug=False):
    if debug:
        print("Determining allele range for MAX_TMRCA=%d, mu=%f, beta=%f, p=%f"%(max_tmrca, mu, beta, p_geom))
    min_possible_range = max(-min_obs_allele, max_obs_allele)

    # Make sure allele range is large enough by checking for leakage, starting from seed
    allele_range = max(seed, min_possible_range)
    while allele_range < max_possible_range:
        mut_model = OUGeomSTRMutationModel(p_geom, mu, beta, allele_range)
        leakages  = []
        trans_mat = mut_model.trans_matrix**max_tmrca
        for allele in min_obs_allele, max_obs_allele:
            vec       = numpy.zeros((mut_model.N,1))
            vec[allele-mut_model.min_n] = 1
            prob = numpy.array(trans_mat.dot(vec).transpose())[0]
            leakages.append(prob[0]+prob[-1])
        if leakages[0] < max_leakage and leakages[1] < max_leakage:
            break
        allele_range += 1
    if allele_range == max_possible_range:
        exit("Unable to find an allele range with leakage < the provided bounds and < the specified maximum")

    # Attempt to reduce allele range, in case seed was larger than needed
    while allele_range >= min_possible_range:
        mut_model = OUGeomSTRMutationModel(p_geom, mu, beta, allele_range)
        leakages  = []
        trans_mat = mut_model.trans_matrix**max_tmrca
        for allele in min_obs_allele, max_obs_allele:
            vec       = numpy.zeros((mut_model.N,1))
            vec[allele-mut_model.min_n] = 1
            prob = numpy.array(trans_mat.dot(vec).transpose())[0]
            leakages.append(prob[0]+prob[-1])
        if leakages[0] > max_leakage or leakages[1] > max_leakage:
            break
        allele_range -= 1
    allele_range += 1

    return allele_range, mut_model.max_step


def determine_total_loglikelihood(tree, node_lst, allele_range, mut_model, gt_probs, len_to_tmrca, debug=False):
    #debug = True
    optimizer = matrix_optimizer.MATRIX_OPTIMIZER(mut_model.trans_matrix, mut_model.min_n)
    optimizer.precompute_results()
        
    node_likelihoods = {}
    root_id          = None
    for node in node_lst:
        if debug:
            print("Processing node %s with distance from root=%f"%(str(node.oid), node.distance_from_root()*len_to_tmrca))
        likelihoods = {}
        if node.parent_node is None:
            root_id = node.oid

        if node.is_leaf():
            likelihoods = []
            sample      = node.taxon.label
            if sample not in gt_probs:
                if debug:
                    print("\t Node is leaf w/o genotype data (sample %s)"%(sample))
                continue            
            sample_probs = gt_probs[sample]
            for val in range(-allele_range, allele_range+1):
                if val in sample_probs:
                    likelihoods.append(numpy.log(sample_probs[val]))
                else:
                    likelihoods.append(-numpy.inf) # TO DO: Add pseudocount or non-zero likelihood?
            node_likelihoods[node.oid] = numpy.array(likelihoods)
            if debug:
                print("\tNode likelihoods = %s"%(str(likelihoods)))
        else:
            children         = node.child_nodes()
            missing_statuses = map(lambda x: x.oid not in node_likelihoods, children)
            if all(missing_statuses):
                if debug:
                    print("\tInternal node w/o any procesed children")
                continue
            else:
                comb_probs = numpy.zeros((mut_model.max_n-mut_model.min_n+1))
                for i,child in enumerate(children):
                    if missing_statuses[i]:
                        continue

                    trans_matrix  = numpy.log(optimizer.get_transition_matrix(int(child.edge_length*len_to_tmrca))).transpose()
                    trans_matrix += node_likelihoods[child.oid]
                    max_vec       = numpy.max(trans_matrix, axis=1)
                    new_probs     = numpy.log(numpy.sum(numpy.exp(trans_matrix-max_vec), axis=1)) + max_vec
                    comb_probs   += numpy.array(new_probs.transpose())[0]

                node_likelihoods[node.oid] = comb_probs
                if debug:
                    print("\tNode likelihoods = %s"%(str(comb_probs)))

    # Overall likelihood corresponds to average of root probabilities (uniform prior)
    root_likelihoods  = node_likelihoods[root_id]
    if debug:
        print("\tRoot node likelihoods = %s"%(str(root_likelihoods)))
    root_likelihoods += numpy.log(1.0/len(root_likelihoods))
    max_prob = numpy.max(root_likelihoods)
    tot_prob = max_prob + numpy.log(numpy.sum(numpy.exp(root_likelihoods-max_prob)))
    if debug:
        print("\t Total likelihood = %f"%(tot_prob))
    return tot_prob

'''
def determine_maximum_loglikelihood(tree, node_lst, allele_range, mut_model, gt_probs, len_to_tmrca, debug=False):
    node_likelihoods = {}
    root_id          = None
    for node in node_lst:
        if debug:
            print("Processing node %s with distance from root =%f"%(str(node.oid), node.distance_from_root()*len_to_tmrca))
        likelihoods = {}
        if node.parent_node is None:
            root_id = node.oid

        if node.is_leaf():
            likelihoods = []
            sample      = node.taxon.label
            if sample not in gt_probs:
                if debug:
                    print("\t Node is leaf w/o genotype data (sample %s)"%(sample))
                continue            
            sample_probs = gt_probs[sample]
            for val in range(-allele_range, allele_range+1):
                if val in sample_probs:
                    likelihoods.append(numpy.log(sample_probs[val]))
                else:
                    likelihoods.append(-numpy.inf) # TO DO: Add non-zero likelihood?
            node_likelihoods[node.oid] = numpy.array(likelihoods)
            if debug:
                print("\tNode likelihoods = %s"%(str(likelihoods)))
        else:
            children         = node.child_nodes()
            missing_statuses = map(lambda x: x.oid not in node_likelihoods, children)
            if all(missing_statuses):
                if debug:
                    print("\tInternal node w/o any procesed children")
                continue
            else:
                comb_probs = numpy.zeros((mut_model.max_n-mut_model.min_n+1))
                for i,child in enumerate(children):
                    if missing_statuses[i]:
                        continue
                    mut_model.compute_forward_matrix(int(child.edge_length*len_to_tmrca))
                    trans_matrix = numpy.log(mut_model.forward_matrix).transpose()
                    child_probs  = node_likelihoods[child.oid]
                    new_probs    = numpy.array(numpy.max((child_probs + trans_matrix), axis=1).transpose())[0]
                    comb_probs  += new_probs
                node_likelihoods[node.oid] = comb_probs
                if debug:
                    print("\tNode likelihoods = %s"%(str(comb_probs)))

    # Overall maximum likelihood corresponds to likelihood for most probable root node
    tot_prob = numpy.max(node_likelihoods[root_id])
    return tot_prob
'''

def plot_str_probs(mut_model, start_allele, tmrcas, output_pdf, min_x=-50, max_x=50, min_y=1e-10):
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    x = numpy.arange(mut_model.min_n, mut_model.max_n+1, 1)
    for tmrca in tmrcas:
        str_probs = mut_model.get_forward_str_probs(start_allele, tmrca)
        filt_x, filt_y = map(list, zip(*filter(lambda x: x[1] > min_y, zip(x, str_probs))))
        ax.plot(filt_x, filt_y, label=str(tmrca))
    ax.set_xlabel("STR Unit Size")
    ax.set_ylabel("P(Size)")
    ax.set_xlim((min_x,max_x))
    ax.set_yscale('log')
    ax.set_title(r"$\mu=$%1.1e, $\beta=$%1.1e, $p_{geom}=$%1.1e"%(mut_model.mu, mut_model.beta, mut_model.p_geom))
    ax.legend(title=r"$N_{generations}$")
    output_pdf.savefig(fig)
    plt.close(fig)

prev_allele_range = 1

def neg_LL_function(tree, len_to_tmrca, max_tmrca,
                    str_gts, min_str, max_str, node_lst,
                    mu, beta, p_geom,
                    min_mu, max_mu, min_beta, max_beta, min_pgeom, max_pgeom):
    if mu < min_mu or mu > max_mu or beta < min_beta or beta > max_beta or p_geom < min_pgeom or p_geom > max_pgeom:
        return numpy.inf

    global prev_allele_range
    allele_range, _   = determine_allele_range_from_seed(max_tmrca, mu, beta, p_geom, min_str, max_str, prev_allele_range)
    prev_allele_range = allele_range
    mut_model = OUGeomSTRMutationModel(p_geom, mu, beta, allele_range)
    tot_ll = determine_total_loglikelihood(tree, node_lst, allele_range+mut_model.max_step, mut_model, 
                                           str_gts, len_to_tmrca, debug=False)
    if tot_ll > 0:
        determine_total_loglikelihood(tree, allele_range+mut_model.max_step, mut_model, str_gts, len_to_tmrca, debug=True)
        print("mu=%.9f, beta=%.9f, p=%.9f, LL=%f"%(mu, beta, p_geom, tot_ll))
        exit("ERROR: Invalid total LL > 0 for parameters above")

    return -tot_ll

def callback_function(val):
    print("Current parameters: mu=%f\tbeta=%f\tp=%f"%(val[0], val[1], val[2]))

def optimize_loglikelihood(tree, len_to_tmrca, max_tmrca,
                           str_gts, min_str, max_str, node_lst,
                           min_mu=0.00001, max_mu=0.05, 
                           min_beta=0.0,   max_beta=0.25, 
                           min_pgeom=0.5,  max_pgeom=1.0,
                           num_iter=2, max_cycle_per_iter=250):
    # X = (log(mu)/log(10), beta, p_geom)
    method      = 'Nelder-Mead'
    fn          = (lambda x: neg_LL_function(tree, len_to_tmrca, max_tmrca, str_gts, min_str, max_str, node_lst, 10**x[0], x[1], x[2], 
                                             min_mu, max_mu, min_beta, max_beta, min_pgeom, max_pgeom))
    best_res = None
    for i in xrange(num_iter):
        # Choose seed where the likelihood is non-zero for seed
        print("Selecting seed")
        while True:
            x0 = [random.uniform(math.log(min_mu)/math.log(10), math.log(max_mu)/math.log(10)),
                  random.uniform(min_beta, max_beta),
                  random.uniform(min_pgeom, max_pgeom)]
            print(x0)
            if not numpy.isnan(fn(x0)):
                break
        print("Finished selecting seed")

        callback_function(x0)
        res = scipy.optimize.minimize(fn, x0, callback=callback_function, method=method, options={'maxiter':max_cycle_per_iter, 'xtol':0.001, 'ftol':0.001})
        if best_res is None or (res.fun < best_res.fun and res.message.replace(" ", "_") == "Optimization_terminated_successfully."):
            best_res = res
    return best_res


def compute_node_order(tree):
    node_lst = []
    for node in tree.postorder_node_iter():
        node_lst.append(node)
    return node_lst

def run_jackknife_procedure(tree, len_to_tmrca, max_tmrca,
                            str_gts, min_str, max_str, node_lst,
                            output_file, sample_size, niter, locus,
                            min_mu=0.00001, max_mu=0.05, 
                            min_beta=0.0,   max_beta=0.25, 
                            min_pgeom=0.5,  max_pgeom=1.0,
                            max_cycle_per_iter=500):
    output  = open(output_file, "w")
    for i in xrange(niter):
        sub_gts = dict(random.sample(str_gts.items(), sample_size))
        success = False
        while not success:
            # Use the best of 3 iterations to reduce the likelihood of poor convergence
            opt_res = optimize_loglikelihood(tree, len_to_tmrca, max_tmrca,
                                             sub_gts, min_str, max_str, node_lst, 
                                             min_mu=min_mu,        max_mu=max_mu, 
                                             min_beta=min_beta,    max_beta=max_beta, 
                                             min_pgeom=min_pgeom,  max_pgeom=max_pgeom,
                                             num_iter=3,           max_cycle_per_iter=max_cycle_per_iter)
            success = opt_res.success

            # Write out optimization results
            output.write("%s\t%f\t%f\t%f\t%f\t%d\t%s\t%d\n"%
                         (locus, opt_res.x[0], opt_res.x[1], opt_res.x[2], 
                          opt_res.fun, opt_res.nit, opt_res.message.replace(" ", "_"), len(sub_gts)))
            output.flush()
            os.fsync(output.fileno())
    output.close()

def check_args(args, arg_lst, option):
    var_dict = vars(args)
    for arg in arg_lst:
        if var_dict[arg] is None:
            exit("Argument --%s is required for --%s option. Exiting..."%(arg, option))

def RepresentsInt(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False

# Check validity of arguments controlling observed reads per sample
# and build the correspoding distribution
def create_read_count_distribution(args):
    read_counts   = args.read_counts.split(",")
    read_percents = args.read_percents.split(",")
    if len(read_counts) != len(read_percents):
        exit("ERROR: Number of arguments to --obs_read_counts and --obs_read_percents must match")
    for i in xrange(len(read_counts)):
        if not RepresentsInt(read_counts[i]):
            exit("ERROR: --obs_read_counts arguments must be a comma-separated list of integers")
        if not RepresentsInt(read_percents[i]):
            exit("ERROR: --obs_read_percents argument must be a comma-separated list of integers")
        read_counts[i]   = int(read_counts[i])
        read_percents[i] = int(read_percents[i])
    if sum(read_percents) != 100:
        exit("ERROR: --obs_read_percents must sum to 100")
    dist = simulate_strs.ReadCountDistribution(read_counts, numpy.array(read_percents)/100.0)
    print(dist)
    return dist

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--tree",  required=True,  dest="tree",      type=str,                 help="File containing phylogeny for samples in newick format")
    parser.add_argument("--out",   required=True,  dest="out",       type=str,                 help="Prefix for all output files")
    parser.add_argument("--pplex", required=False, dest="powerplex", type=str,   default=None, help="File containg powerplex STR genotypes")   
    parser.add_argument("--vcf",   required=False, dest="vcf",       type=str,   default=None,
                        help="VCF file containing STR genotypes (and potentially read counts in ALLREADS FORMAT field)")

    # Options to control the edge length -> # generation scaling factor
    parser.add_argument("--max_tmrca", required=False, type=int, dest="max_tmrca", default=175000, help="Maximum TMRCA of tree (years)")
    parser.add_argument("--gen_time",  required=False, type=int, dest="gen_time",  default=30,     help="Generation time (years)")

    # Options used for both the jackknife and simulations
    parser.add_argument("--nsamp", required=False, dest="nsamp", type=int, default=None, help="")
    parser.add_argument("--niter", required=False, dest="niter", type=int, default=None, help="")

    # Option to use the read counts in the VCF instead of the genotype posteriors
    # Computes the genotype posteriors using data from the specified stutter model file and the read counts
    parser.add_argument("--use_read_counts",   required=False, dest="use_read_counts", type=str, default=None, help="")

    # Various analysis options to use REAL STR genotypes to
    # 1) Estimate stutter models
    # 2) Estimate mutation rates using all sample
    # 3) Estimate jackknifed mutation rates
    calc_stutter_probs_desc = """ Estimate the stutter model for each locus in the VCF using the EM method.
    Requires a VCF that contains the ALLREADS FORMAT field, which is used to obtain the STR sizes
    observed in each read"""
    parser.add_argument("--calc_stutter_probs", required=False, action="store_true",              default=False, help=calc_stutter_probs_desc)
    parser.add_argument("--calc_mut_rates",     required=False, action="store_true",              default=False, help="")
    parser.add_argument("--loc_range",          required=False, dest="loc_range",       type=str, default=None,  help="")

    parser.add_argument("--jackknife",          required=False, action="store_true",              default=False, help="")
    parser.add_argument("--locus",              required=False, dest="locus",           type=str, default=None,  help="")
    parser.add_argument("--plot_ss",            required=False, action="store_true",              default=False, help="")
    
    # Options for
    # 1) Simulating STR genotypes
    # 2) Simulating observed read counts after stutter distortion
    # 3) EM stutter model estimation procedure using simulated reads
    # 4) Mutation rate estimation using simulated reads and different genotypers
    sim_stutter_em_desc = """Simulates STRs down the phylogeny using the provided mutation model
    Generates observed reads for each sample using the read count distributions and the stutter model parameters.
    Estimates the stutter model parameters using the phylogeny and observed reads"""
    sim_and_est_desc    = """Performs the same initial procedure as --sim_stutter_em.
    However, it proceeds by using the specified genotyper, phylogeny and
    observed read counts to estimate the underlying mutation model"""
    genotyper_desc      = """Genotyper to use to compute STR posteriors from simulated read counts.
    Options = (EXACT, EXACT_STUTTER, EST_STUTTER, FRACTION). Default = EXACT"""
    stutter_dec_desc    = "Probability that stutter decreases the number of observed repeats in a read. Default=0.0"
    stutter_inc_desc    = "Probability that stutter increases the number of observed repeats in a read. Default=0.0"
    parser.add_argument("--sim_stutter_em",    required=False, action="store_true",              default=False,      help=sim_stutter_em_desc)
    parser.add_argument("--sim_and_est",       required=False, action="store_true",              default=False,      help=sim_and_est_desc)

    parser.add_argument("--sim_replication",   required=False, action="store_true",              default=False,      help="")
    parser.add_argument("--subset_size",       required=False, dest="subset_size",  type=int,    default=None,       help="")

    parser.add_argument("--sim_str_reads",     required=False, action="store_true",              default=False,      help="")
    parser.add_argument("--mu",                required=False, dest="mu",            type=float, default=None,       help="Mutation rate for simulating STRs (mut/gen)")
    parser.add_argument("--beta",              required=False, dest="beta",          type=float, default=None,       help="Length constraint for simulating STRs")
    parser.add_argument("--pgeom",             required=False, dest="p_geom",        type=float, default=None,       help="Geometric parameter for simulating STRs")
    parser.add_argument("--stutter_geom",      required=False, dest="stutter_geom",  type=float, default=1.0,        help="Geometric parameter for stutter model. Default=1.0")
    parser.add_argument("--stutter_dec",       required=False, dest="stutter_dec",   type=float, default=0.0,        help=stutter_dec_desc)
    parser.add_argument("--stutter_inc",       required=False, dest="stutter_inc",   type=float, default=0.0,        help=stutter_inc_desc)
    parser.add_argument("--genotyper",         required=False, dest="genotyper",     type=str,   default="EXACT",    help=genotyper_desc)
    parser.add_argument("--obs_read_counts",   required=False, dest="read_counts",   type=str,   default="1,2,3",    help="Categories of obs. reads per sample. Default=1,2,3")
    parser.add_argument("--obs_read_percents", required=False, dest="read_percents", type=str,   default="65,25,10", help="Frequency of each obs. read count. Default=65,25,10")

    # Options to specify the optimization boundaries for each parameter
    parser.add_argument("--min_mu",    required=False, dest="min_mu",    type=float, default=0.00001, help="Lower optimization boundary for mu")
    parser.add_argument("--max_mu",    required=False, dest="max_mu",    type=float, default=0.05,    help="Upper optimization boundary for mu")
    parser.add_argument("--min_pgeom", required=False, dest="min_pgeom", type=float, default=0.5,     help="Lower optimization boundary for pgeom")
    parser.add_argument("--max_pgeom", required=False, dest="max_pgeom", type=float, default=1.0,     help="Upper optimization boundary for pgeom")
    parser.add_argument("--min_beta",  required=False, dest="min_beta",  type=float, default=0.0,     help="Lower optimization boundary for beta")
    parser.add_argument("--max_beta",  required=False, dest="max_beta",  type=float, default=0.75,    help="Upper optimization boundary for beta")

    args = parser.parse_args()
    max_tmrca     = args.max_tmrca/args.gen_time # TMRCA of all men in phylogeny (in generations)
    min_node_conf = 0         # Remove nodes with conf < threshold

    # Create the read count distribution
    read_count_dist = create_read_count_distribution(args)

    # Read and prune tree based on node confidences
    tree,haplogroups = read_tree(args.tree, min_node_conf)
    node_lst         = compute_node_order(tree)

    # Determinine tree depth range
    min_depth,max_depth,median_depth = tree_depths(tree)
    print("Minimum tree depth = %f, Maximum tree depth = %f, Median tree depth = %f"%(min_depth, max_depth, median_depth))

    # Calculate conversion from edge length to tmrca
    len_to_tmrca = max_tmrca/median_depth
    #len_to_tmrca = max_tmrca/max_depth

    # Determine maximum edge length
    max_edge_len = max_edge_length(tree)
    print("Maximum edge length = %f"%(max_edge_len))

    # Determine number of meioses in phylogeny
    print("# Meioses = %f"%(num_meioses(tree, len_to_tmrca)))

    valid_samples = None
    
    # If using read counts, read information about the stutter model
    if args.use_read_counts is not None:
        stutter_params = stutter_info.read_stutter_info(args.use_read_counts, min_pgeom=0.0)

    if args.calc_mut_rates:
        if args.powerplex is not None and args.vcf is not None:
            exit("ERROR: Options --powerplex and --vcf can't both be supplied with the --calc_mut_rates option")
        elif args.powerplex is not None:
            powerplex_gt_dict = read_powerplex.read_genotypes(args.powerplex, conv_to_likelihood=True)
            output = open(args.out+".powerplex.mut_rates.txt", "w")
            for locus in powerplex_gt_dict:
                print("Estimating mutation parameters for %s"%(locus))
                str_gts = powerplex_gt_dict[locus]
                if valid_samples is not None:
                    str_gts = dict(filter(lambda x: x[0] in valid_samples, str_gts.items()))
                min_str = min(reduce(lambda x,y: x+y, map(lambda x: x.keys(), str_gts.values())))
                max_str = max(reduce(lambda x,y: x+y, map(lambda x: x.keys(), str_gts.values())))
                opt_res = optimize_loglikelihood(tree, len_to_tmrca, max_tmrca,
                                                 str_gts, min_str, max_str, node_lst, 
                                                 min_mu=args.min_mu,       max_mu=args.max_mu,
                                                 min_beta=args.min_beta,   max_beta=args.max_beta,
                                                 min_pgeom=args.min_pgeom, max_pgeom=args.max_pgeom, num_iter=3)
                output.write("%s\t%f\t%f\t%f\t%f\t%d\t%s\t%d\n"%
                             (locus, opt_res.x[0], opt_res.x[1], opt_res.x[2], 
                              opt_res.fun, opt_res.nit, opt_res.message.replace(" ", "_"), len(str_gts)))
                output.flush()
                os.fsync(output.fileno())
            output.close()
        elif args.vcf is not None:
            check_args(args, ["loc_range"], "calc_mut_rates")
            chrom      = args.loc_range.split(":")[0]
            start,stop = map(int, args.loc_range.split(":")[1].split("-"))
            vcf_reader = vcf.Reader(filename=args.vcf)
            vcf_reader.fetch(chrom, start=start, end=stop+100)
            output     = open(args.out+".mut_rates.txt", "w")
            while True:
                if args.use_read_counts is None:
                    success,str_gts,min_str,max_str,locus = read_str_vcf.get_str_gts(vcf_reader)
                else:
                    success,chrom,start,end,motif_len,read_count_dict,in_frame_count,out_frame_count,locus = read_str_vcf.get_str_read_counts(vcf_reader)
                    if not success:
                        break
                    key = chrom + ":" + str(start) + "-" + str(end)
                    if key not in stutter_params:
                        continue
                    p_geom, down, up = stutter_params[key]["P_GEOM"], stutter_params[key]["DOWN"], stutter_params[key]["UP"]
                    print("Using stutter parameters %f %f %f"%(p_geom, down, up))                    
                    str_gts,min_str,max_str = read_str_vcf.counts_to_centalized_posteriors(read_count_dict, p_geom, down, up)

                if valid_samples is not None:
                    str_gts = dict(filter(lambda x: x[0] in valid_samples, str_gts.items()))
                if not success:
                    break
                print("Estimating mutation parameters for %s using STR gentoypes for %d samples"%(locus, len(str_gts)))
                opt_res = optimize_loglikelihood(tree, len_to_tmrca, max_tmrca,
                                                 str_gts, min_str, max_str, node_lst, 
                                                 min_mu=args.min_mu,       max_mu=args.max_mu,
                                                 min_beta=args.min_beta,   max_beta=args.max_beta,
                                                 min_pgeom=args.min_pgeom, max_pgeom=args.max_pgeom, num_iter=3)
                output.write("%s\t%f\t%f\t%f\t%f\t%d\t%s\t%d\n"%
                             (locus, opt_res.x[0], opt_res.x[1], opt_res.x[2], 
                              opt_res.fun, opt_res.nit, opt_res.message.replace(" ", "_"), len(str_gts)))
                output.flush()
                os.fsync(output.fileno())
            output.close()
        else:
            exit("ERROR: Either --powerplex or --vcf must be supplied to run the --calc_mut_rates option")

    if args.plot_ss:
        print("Generating step size distribution plots")
        pp   = PdfPages(args.out + ".pdf")
        min_str, max_str = -5, 5
        mu     = args.mu
        beta   = args.beta
        p_geom = args.p_geom                                                                                                                                     
        for mu in [0.01, 0.001, 0.0001]:
            for beta in [0, 0.1, 0.25]:
                for p_geom in [0.5, 0.75, 0.9, 0.95, 0.99, 1.0]:
                    print("mu=%f, beta=%f, p=%f"%(mu, beta, p_geom))
                    allele_range, max_step = determine_allele_range(max_tmrca, mu, beta, p_geom, min_str, max_str) 
                    mut_model = OUGeomSTRMutationModel(p_geom, mu, beta, allele_range)
                    plot_str_probs(mut_model, 0, [1, 10, 25, 100, 1000, 10000], pp)
        pp.close()

    if args.jackknife:
        check_args(args, ["niter", "locus", "nsamp"], "jackknife")
        if args.powerplex is not None and args.vcf is not None:
            exit("ERROR: Options --powerplex and --vcf can't both be supplied with the --jackknife option")
        elif args.powerplex is not None:
            powerplex_gt_dict = read_powerplex.read_genotypes(args.powerplex, conv_to_likelihood=True)
            if args.locus not in powerplex_gt_dict:
                exit("ERROR: Requested locus not in PowerPlex data file. Exiting...")
            str_gts = powerplex_gt_dict[args.locus]
            if valid_samples is not None:
                str_gts = dict(filter(lambda x: x[0] in valid_samples, str_gts.items()))
            min_str = min(reduce(lambda x,y: x+y, map(lambda x: x.keys(), str_gts.values())))
            max_str = max(reduce(lambda x,y: x+y, map(lambda x: x.keys(), str_gts.values())))
        elif args.vcf is not None:
            vcf_reader = vcf.Reader(filename=args.vcf)
            str_chrom,str_start = args.locus.split(":")
            vcf_reader.fetch(str_chrom, start=int(str_start), end=int(str_start)+10000)
            if args.use_read_counts is None:
                success,str_gts,min_str,max_str,locus = read_str_vcf.get_str_gts(vcf_reader)
            else:
                success,chrom,start,end,motif_len,read_count_dict,in_frame_count,out_frame_count,locus = read_str_vcf.get_str_read_counts(vcf_reader)
                key = chrom + ":" + str(start) + "-" + str(end)
                if not success:
                    exit("Locus %s not found in provided VCF file. Exiting..."%(args.locus))
                if key not in stutter_params:
                    exit("ERROR: No stutter information available for locus %s"%(args.locus))
                p_geom, down, up = stutter_params[key]["P_GEOM"], stutter_params[key]["DOWN"], stutter_params[key]["UP"]
                print("Using stutter parameters %f %f %f"%(p_geom, down, up))
                str_gts,min_str,max_str = read_str_vcf.counts_to_centalized_posteriors(read_count_dict, p_geom, down, up)

            if valid_samples is not None:
                str_gts = dict(filter(lambda x: x[0] in valid_samples, str_gts.items()))
        else:
            exit("ERROR: Either --powerplex or --vcf must be supplied to run the --jackknife option")
                
        output_file = args.out + (".powerplex" if args.powerplex is not None else ".vcf") + ".jackknife.txt"
        sample_size = args.nsamp

        print("Performing jackknife anaylsis")
        #with PyCallGraph(output=GraphvizOutput()):
        run_jackknife_procedure(tree, len_to_tmrca, max_tmrca,
                                str_gts, min_str, max_str, node_lst,
                                output_file, sample_size, int(args.niter), locus,
                                min_mu=args.min_mu,       max_mu=args.max_mu,
                                min_beta=args.min_beta,   max_beta=args.max_beta,
                                min_pgeom=args.min_pgeom, max_pgeom=args.max_pgeom,
                                max_cycle_per_iter=500)

    if args.calc_stutter_probs:
        check_args(args, ["vcf"], "calc_stutter_probs")
        vcf_reader = vcf.Reader(filename=args.vcf)
        res_out    = open(args.out+".stutter_probs.txt", "w")
        while True:
            success,chrom,start,end,motif_len,read_count_dict,in_frame_count,out_frame_count,locus = read_str_vcf.get_str_read_counts(vcf_reader)
            if not success:
                break

            print("%s\t%d\t%d\t%d\t%f"%(chrom, start, in_frame_count, out_frame_count, 100.0*out_frame_count/(in_frame_count+out_frame_count)))
            # Skip loci with large fractions of samples with out-of-frame reads
            if 100.0*out_frame_count/(in_frame_count+out_frame_count) > 5:
                continue

            genotyper  = genotypers.EstStutterGenotyper()
            obs_sizes  = sorted(list(set(reduce(lambda x,y: x+y, map(lambda x: x.keys(), read_count_dict.values())))))
            min_allele = obs_sizes[0]
            max_allele = obs_sizes[-1]
            genotyper.train(read_count_dict, min_allele, max_allele)
            res_out.write("%s\t%d\t%d\t%d\t%s\t%s\t%s\t%s\t%d\t%d\t%d\n"%(chrom, start, end, motif_len, genotyper.est_pgeom, genotyper.est_down, genotyper.est_up,
                                                                          genotyper.status, genotyper.eff_coverage, in_frame_count, out_frame_count))
            res_out.flush()
            os.fsync(res_out.fileno())
        res_out.close()

    if args.sim_and_est:
        check_args(args, ["mu", "beta", "p_geom", "niter", "nsamp"], "sim_and_est")

        # Construct the PCR stutter model, genotyper and read count distribution
        pcr_stutter_model = stutter_model.GeomStutterModel(args.stutter_geom, args.stutter_dec, args.stutter_inc, tolerance=10**-6)
        genotyper         = genotypers.get_genotyper(args.genotyper)
        
        res_out = open(args.out+".sim_and_est.txt", "w")
        for i in xrange(args.niter):
            print("Simulating STR genotypes with mu=%f, beta=%f and p=%f, genotyper=%s and stutter model=%s"
                  %(args.mu, args.beta, args.p_geom, genotyper.__str__(), pcr_stutter_model.__str__()))
            root_allele = random.randint(-4, 4)
            allele_range, max_step = determine_allele_range(max_tmrca, args.mu, args.beta, args.p_geom, min(0, root_allele), max(0, root_allele))
            mut_model = OUGeomSTRMutationModel(args.p_geom, args.mu, args.beta, allele_range)
            simulated_gts, est_stutter = simulate_strs.simulate(tree, mut_model, len_to_tmrca, args.nsamp, pcr_stutter_model,
                                                                read_count_dist, genotyper, debug=False,
                                                                valid_samples=valid_samples, root_allele=root_allele)
            print("Done simulating genotypes")

            # Output comparison
            if simulated_gts is None:
                res_out.write("%s\n"%(est_stutter.status)) # Stutter estimation failed, so output the reason  
            else:
                # Determine the range of STR genotypes observed
                obs_strs = sorted(list(set(reduce(lambda x,y: x+y, map(lambda x: x.keys(), simulated_gts.values())))))
                min_str  = obs_strs[0]
                max_str  = obs_strs[-1]
                print(min_str, max_str)
                
                opt_res = optimize_loglikelihood(tree, len_to_tmrca, max_tmrca,
                                                 simulated_gts,  min_str, max_str, node_lst, 
                                                 min_mu=args.min_mu,       max_mu=args.max_mu,
                                                 min_beta=args.min_beta,   max_beta=args.max_beta,
                                                 min_pgeom=args.min_pgeom, max_pgeom=args.max_pgeom, num_iter=3)

                true_up, true_down, true_pgeom = pcr_stutter_model.get_prob_up(), pcr_stutter_model.get_prob_down(), pcr_stutter_model.get_pgeom()

                if args.genotyper == "EST_STUTTER":
                    est_down,est_up,est_pgeom  = est_stutter.est_down, est_stutter.est_up, est_stutter.est_pgeom
                else:
                    est_up,est_down,est_pgeom = "N/A", "N/A", "N/A"
                res_out.write("%f\t%f\t%f\t%f\t%f\t%f\t%f\t%d\t%s\t%d\t%s\t%f\t%f\t%f\t%s\t%s\t%s\t%s\t%s\n"%
                              (math.log(args.mu)/math.log(10), args.beta, args.p_geom, opt_res.x[0], opt_res.x[1], opt_res.x[2], 
                               opt_res.fun, opt_res.nit, opt_res.message.replace(" ", "_"), len(simulated_gts), 
                               genotyper.__str__(), true_pgeom, true_down, true_up, est_pgeom, est_down, est_up, 
                               args.read_counts, args.read_percents))
                res_out.flush()
                os.fsync(res_out.fileno())
        res_out.close()


    if args.sim_replication:
        check_args(args, ["mu", "beta", "p_geom", "niter", "nsamp", "subset_size"], "sim_replication")
        if args.subset_size > args.nsamp:
            exit("ERROR: Subset size cannot exceed the total number of samples")

        # Construct the PCR stutter model, genotyper and read count distribution
        subset_sizes      = [args.nsamp-args.subset_size, args.subset_size]
        pcr_stutter_model = stutter_model.GeomStutterModel(args.stutter_geom, args.stutter_dec, args.stutter_inc, tolerance=10**-6)
        genotyper         = genotypers.get_genotyper(args.genotyper)

        res_out = open(args.out+".sim_replication.txt", "w")
        for i in xrange(args.niter):
            print("Simulating STR genotypes with mu=%f, beta=%f and p=%f, genotyper=%s and stutter model=%s"
                  %(args.mu, args.beta, args.p_geom, genotyper.__str__(), pcr_stutter_model.__str__()))
            root_allele = random.randint(-4, 4)
            allele_range, max_step = determine_allele_range(max_tmrca, args.mu, args.beta, args.p_geom, min(0, root_allele), max(0, root_allele))
            mut_model = OUGeomSTRMutationModel(args.p_geom, args.mu, args.beta, allele_range)
            simulated_gt_sets, est_stutter = simulate_strs.simulate(tree, mut_model, len_to_tmrca, args.nsamp, pcr_stutter_model,
                                                                    read_count_dist, genotyper, debug=False,
                                                                    valid_samples=valid_samples, root_allele=root_allele, subset_sizes=subset_sizes)
            print("Done simulating genotypes")

            # Output comparison
            if simulated_gt_sets is None:
                res_out.write("%s\n"%(est_stutter.status)) # Stutter estimation failed, so output the reason
            else:
                true_up, true_down, true_pgeom = pcr_stutter_model.get_prob_up(), pcr_stutter_model.get_prob_down(), pcr_stutter_model.get_pgeom()
                res_out.write("%f\t%f\t%f\t%s\t%f\t%f\t%f\t%s\t%s"%
                              (math.log(args.mu)/math.log(10), args.beta, args.p_geom, genotyper.__str__(), true_pgeom, true_down, true_up,
                               args.read_counts, args.read_percents))

                for j in xrange(len(subset_sizes)):
                    # Determine the range of STR genotypes observed
                    obs_strs = sorted(list(set(reduce(lambda x,y: x+y, map(lambda x: x.keys(), simulated_gt_sets[j].values())))))
                    min_str  = obs_strs[0]
                    max_str  = obs_strs[-1]
                    print(min_str, max_str)

                    opt_res = optimize_loglikelihood(tree, len_to_tmrca, max_tmrca,
                                                     simulated_gt_sets[j],  min_str, max_str, node_lst,
                                                     min_mu=args.min_mu,       max_mu=args.max_mu,
                                                     min_beta=args.min_beta,   max_beta=args.max_beta,
                                                     min_pgeom=args.min_pgeom, max_pgeom=args.max_pgeom, num_iter=3)
                    res_out.write("\t%f\t%f\t%f\t%f\t%d\t%s\t%d"%
                                  (opt_res.x[0], opt_res.x[1], opt_res.x[2], opt_res.fun, opt_res.nit, opt_res.message.replace(" ", "_"), len(simulated_gt_sets[j])))
                res_out.write("\n")
                res_out.flush()
                os.fsync(res_out.fileno())
        res_out.close()

    if args.sim_stutter_em:
        check_args(args, ["mu", "beta", "p_geom", "niter", "nsamp"], "sim_stutter_em")

        # Construct the PCR stutter model
        pcr_stutter_model = stutter_model.GeomStutterModel(args.stutter_geom, args.stutter_dec, args.stutter_inc, tolerance=10**-6)

        res_out = open(args.out+".stutter_em.txt", "w")
        for i in xrange(args.niter):
            min_str, max_str = -8, -8
            genotyper        = genotypers.EstStutterGenotyper() # Since we're interested in stutter model estimation, use this genotyper type      
            print("Simulating STR genotypes with mu=%f, beta=%f and p=%f, genotyper=%s and stutter model=%s"
                  %(args.mu, args.beta, args.p_geom, genotyper.__str__(), pcr_stutter_model.__str__()))
            root_allele = random.randint(-4, 4)
            allele_range, max_step = determine_allele_range(max_tmrca, args.mu, args.beta, args.p_geom, min(0, root_allele), max(0, root_allele))
            mut_model = OUGeomSTRMutationModel(args.p_geom, args.mu, args.beta, allele_range)
            
            # Simulate STRs and their observed reads and estimate the underlying stutter model
            simulated_read_counts, est_stutter = simulate_strs.simulate(tree, mut_model, len_to_tmrca, args.nsamp,
                                                                        pcr_stutter_model, read_count_dist, genotyper,
                                                                        debug=False, root_allele=root_allele)

            # Determine the range of observed STR genotypes
            obs_strs = sorted(list(set(reduce(lambda x,y: x+y, map(lambda x: x.keys(), simulated_read_counts.values())))))
            min_str  = obs_strs[0]
            max_str  = obs_strs[-1]

            # Output comparison
            if simulated_read_counts is None:
                res_out.write("%s\n"%(est_stutter.status)) # Stutter estimation failed, so output the reason
            else:
                true_up, true_down, true_pgeom = pcr_stutter_model.get_prob_up(), pcr_stutter_model.get_prob_down(), pcr_stutter_model.get_pgeom()
                est_up,  est_down,  est_pgeom  = est_stutter.est_up, est_stutter.est_down, est_stutter.est_pgeom
                res_out.write("%f\t%f\t%f\t%f\t%f\t%f\t%s\t%s\t%s\t%d\t%d\t%s\t%s\t%s\t%s\n"%
                              (math.log(args.mu)/math.log(10), args.beta, args.p_geom, 
                               true_pgeom, true_down, true_up, est_pgeom, est_down, est_up, len(simulated_read_counts), est_stutter.eff_coverage, 
                               args.read_counts, args.read_percents, est_stutter.LL, est_stutter.status)) 
            res_out.flush()
            os.fsync(res_out.fileno())
        res_out.close()

    if args.sim_str_reads:
        check_args(args, ["mu", "beta", "p_geom", "nsamp"], "sim_str_reads")

        # Construct the PCR stutter model
        pcr_stutter_model = stutter_model.GeomStutterModel(args.stutter_geom, args.stutter_dec, args.stutter_inc, tolerance=10**-6)

        print("Simulating STR genotypes with mu=%f, beta=%f and p=%f and stutter model=%s"
              %(args.mu, args.beta, args.p_geom, pcr_stutter_model.__str__()))
        root_allele = random.randint(-4, 4)
        allele_range, max_step = determine_allele_range(max_tmrca, args.mu, args.beta, args.p_geom, min(0, root_allele), max(0, root_allele))
        mut_model = OUGeomSTRMutationModel(args.p_geom, args.mu, args.beta, allele_range)

        genotyper = genotypers.ReadCounter()

        # Simulate STRs and their observed reads and estimate the underlying stutter model
        simulated_read_counts, est_stutter = simulate_strs.simulate(tree, mut_model, len_to_tmrca, args.nsamp,
                                                                    pcr_stutter_model, read_count_dist, genotyper,
                                                                    debug=False, root_allele=root_allele)
        output = open(args.out + ".str_reads.txt", "w")
        for sample in simulated_read_counts:
            items = sorted(simulated_read_counts[sample].items())
            output.write("%s\t%s\t%s\n"%(sample, ",".join(map(str, map(lambda x: x[0], items))), ",".join(map(str, map(lambda x:x[1], items)))))
        output.close()

if __name__ == "__main__":
    main()
