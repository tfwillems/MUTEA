# Functions to compute the genotype posteriors for every node in a tree given a set of observed genotypes
# Utilizes a simple and exact message passing scheme, whereby messages are 
# i)   Passed from the leaves to the root
# ii)  Passed form the root to the leaves
# iii) Combined into posteriors

import collections
import ete2
import numpy
import random
from scipy.misc import logsumexp
from ete2 import Tree, TreeStyle, TextFace, PieChartFace, faces

from matrix_optimizer import MATRIX_OPTIMIZER
from mutation_model import OUGeomSTRMutationModel

# Return true iff the tree is binary
def is_binary(tree):
    for node in tree.preorder_node_iter():
        if not node.is_leaf() and len(node.child_nodes()) != 2:
            return False
    return True

# Compute P(D_child | parent = z), the probability of all the data in the subtree of a node's child given the node's genotype
# Nodes with no data in their subtree are absent from the resulting dictionary
# Returns dict{ Key = parent,  Value =  dict{ Key = child, Value = [prob(D_child | parent=min_allele), ..., prob(D_child | parent=max_allele)] } }
def compute_backward_messages(optimizer, postorder_node_lst, allele_range, gt_probs, gen_per_len):
    data_likelihoods = {}
    for node in postorder_node_lst:
        if node.is_leaf():
            continue
        else:
            parent_dict = {} 
            for child in node.child_nodes():
                if child.is_leaf():
                    if child.taxon.label in gt_probs:
                        sample_probs = gt_probs[child.taxon.label]
                        comb_probs   = numpy.array([numpy.log(sample_probs[val]) if val in sample_probs else -numpy.inf for val in xrange(-allele_range, allele_range+1)])
                    else:
                        continue
                elif child.oid in data_likelihoods:
                    comb_probs = numpy.sum(data_likelihoods[child.oid].values(), axis=0)
                else:
                    continue
            
                trans_matrix  = comb_probs + numpy.log(optimizer.get_transition_matrix(int(child.edge_length*gen_per_len))).transpose()
                tot_probs     = logsumexp(trans_matrix, axis=1)
                parent_dict[child.oid] = tot_probs

            if len(parent_dict) != 0:
                data_likelihoods[node.oid] = parent_dict
    return data_likelihoods

# Compute P(N | D_above), the probability of a node given all the data not in its childrens subtrees
def compute_forward_messages(optimizer, preorder_node_lst, gen_per_len, subtree_data_likelihoods):
    node_likelihoods = {}
    for node in preorder_node_lst:
        # Skip root node
        if node.parent_node is None:
            root_id = node.oid
            continue

        have_data    = False
        trans_matrix = numpy.log(optimizer.get_transition_matrix(int(node.edge_length*gen_per_len))).transpose()
        if node.parent_node.oid in node_likelihoods:
            trans_matrix += node_likelihoods[node.parent_node.oid]
            have_data     = True

        if node.parent_node.oid in subtree_data_likelihoods:
            for sibling in node.parent_node.child_nodes():
                if sibling.oid == node.oid:
                    continue
                if sibling.oid in subtree_data_likelihoods[node.parent_node.oid]:
                    have_data     = True  
                    trans_matrix += subtree_data_likelihoods[node.parent_node.oid][sibling.oid]

        if have_data:
            tot_probs      = logsumexp(trans_matrix, axis=1)
            norm_factor    = logsumexp(tot_probs)
            log_posteriors = tot_probs - norm_factor
            node_likelihoods[node.oid] = log_posteriors

    return node_likelihoods


# Compute P(N | D), the posterior probability of a node given all of the observed data
def compute_node_posteriors(tree, mut_model, gt_probs, allele_range, gen_per_len, tree_image=None):
    print("Precomputing transition probabilities")
    optimizer = MATRIX_OPTIMIZER(mut_model.trans_matrix, mut_model.min_n)
    optimizer.precompute_results()

    if not is_binary(tree):
        exit("ERROR: Tree must be binary for posterior inference. Exiting...")

    print("Passing messages from leaves -> root")
    subtree_data_likelihoods = compute_backward_messages(optimizer, tree.postorder_node_iter(), allele_range+mut_model.max_step, gt_probs, gen_per_len)

    print("Passing messages from root -> leaves")
    node_likelihoods = compute_forward_messages(optimizer, tree.preorder_node_iter(), gen_per_len, subtree_data_likelihoods)

    print("Merging messages into posteriors")
    node_log_posteriors = {}
    estimated_gt_probs  = {}
    for node in tree.nodes():
        probs     = numpy.array([0 for val in xrange(-allele_range-mut_model.max_step, allele_range+mut_model.max_step+1)])
        have_data = False

        if node.oid in node_likelihoods:
            have_data = True
            probs    += node_likelihoods[node.oid]
        
        if node.oid in subtree_data_likelihoods:
            probs    += numpy.sum(subtree_data_likelihoods[node.oid].values(), axis=0)
            have_data = True

        if node.is_leaf() and node.taxon.label in gt_probs:
            sample_probs = gt_probs[node.taxon.label]
            probs += numpy.array([numpy.log(sample_probs[val]) if val in sample_probs else -numpy.inf 
                                  for val in xrange(-allele_range-mut_model.max_step, allele_range+mut_model.max_step+1)])

        if not have_data:
            exit("ERROR: No forward or backward messages available for node %s during posterior inference. Exiting..."%(node.label))
        
        norm_factor    = logsumexp(probs)
        log_posteriors = probs - norm_factor
        node_log_posteriors[node.get_node_str()] = log_posteriors

        if node.is_leaf() and node.taxon.label not in gt_probs:
            estimated_gt_probs[node.taxon.label] = dict(zip(range(-allele_range-mut_model.max_step, allele_range+mut_model.max_step+1), numpy.exp(log_posteriors)))

    return node_log_posteriors, estimated_gt_probs

