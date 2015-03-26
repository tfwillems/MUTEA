import math
import numpy
import random
from scipy.misc import logsumexp
from scipy.stats import norm, geom
    
class StutterModel:
    def __init__(self, stutter_sizes, stutter_probs):
        self.stutter_sizes   = stutter_sizes
        self.stutter_probs   = stutter_probs
        self.LL_stutter_dict = dict(zip(stutter_sizes, map(lambda x: math.log(x) if x > 0 else -100, stutter_probs)))

    def get_prob_up(self):
        total = 0.0
        for i in xrange(len(self.stutter_sizes)):
            if self.stutter_sizes[i] > 0:
                total += self.stutter_probs[i]
        return total

    def get_prob_down(self):
        total = 0.0
        for i in xrange(len(self.stutter_sizes)):
            if self.stutter_sizes[i] < 0:
                total += self.stutter_probs[i]
        return total

    def get_log_stutter_size_prob(self, size):
        if size not in self.LL_stutter_dict:
            exit("ERROR: No LL for stutter size %d"%(size))
        return self.LL_stutter_dict[size]

    def get_genotype_posteriors(self, log_gt_priors, read_counts):
        # Compute log-likelihood for each allele
        gts,LLs = [],[]
        for gt in log_gt_priors.keys():
            #LL = log_gt_priors[gt]
            LL  = 0 # Ignore priors when computing posteriors
            for size,count in read_counts.items():
                stutter_size = size - gt
                if stutter_size not in self.LL_stutter_dict:
                    LL = -numpy.inf
                else:
                    LL += count*self.LL_stutter_dict[stutter_size]
            LLs.append(LL)
            gts.append(gt)

        # Compute total log-likelihood
        total_LL = logsumexp(LLs)

        # Normalize each allele's LL by the total LL to obtain posteriors
        for i in xrange(len(gts)):
            LLs[i] = numpy.exp(LLs[i]-total_LL)

        return dict(zip(gts, LLs))
            
    # Randomly select a stutter size based on the stutter probabilities
    def random_stutter_size(self):
        rand_val = random.random()
        for i in xrange(len(self.stutter_probs)):
            if rand_val < self.stutter_probs[i]:
                return self.stutter_sizes[i]
            rand_val -= self.stutter_probs[i]
        exit("ERROR: Stutter probabilities don't add up to 1")

    def __str__(self):
        return "Step Sizes = " + str(self.stutter_sizes) + ", Step Probs = " + str(self.stutter_probs)


class GeomStutterModel(StutterModel):
    def __init__(self, p_geom, p_down, p_up, tolerance=10**-6):
        self.p_geom = p_geom
        self.p_up   = p_up
        self.p_down = p_down

        # Check validity of specified probabilities
        if p_geom <= 0 or p_geom > 1:
            exit("p_geom must be in (0, 1]")
        if p_up < 0 or p_up > 1:
            exit("p_up must be in [0, 1]")
        if p_down < 0:
            exit("p_down must be in [0, 1]")
        if p_up + p_down > 1:
            exit("p_up + p_down must be <= 1")

        if p_geom == 1.0:
            StutterModel.__init__(self, [-1, 0, 1], [p_down, 1.0-p_down-p_up, p_up])
        else:
            # Determine number of steps required to achieve desired tolerance
            prob       = p_geom
            tot_rem    = 1.0
            max_step   = 0
            while tot_rem > tolerance:
                tot_rem  -= prob
                prob     *= (1-p_geom)
                max_step += 1

            # Compute step probabilities
            steps           = numpy.arange(1, max_step+1, 1)        
            step_probs      = geom.pmf(steps, self.p_geom)
            step_probs[-1] += geom.sf(steps[-1], self.p_geom)

            # Construct the model
            stutter_sizes = list(map(lambda x:-x, steps[::-1])) + [0] + list(steps)
            stutter_probs = list(p_down*step_probs[::-1]) + [1.0-p_down-p_up] + list(p_up*step_probs)
            StutterModel.__init__(self, stutter_sizes, stutter_probs)

    def get_pgeom(self):
        return self.p_geom

    def __str__(self):
        return "GeomStutterModel (p_geom=%f, p_down=%f, p_up=%f)"%(self.p_geom, self.p_down, self.p_up)

    

if __name__ == "__main__":
    sizes      = [-3, -2, -1, 0, 1, 2, 3]
    probs      = [0.01, 0.05, 0.1, 0.68, 0.1, 0.05, 0.01]
    model      = StutterModel(sizes, probs)
    geom_model = GeomStutterModel(0.8, 0.01, 0.1)
    print(geom_model.stutter_sizes)
    print(geom_model.stutter_probs)
    print(sum(geom_model.stutter_probs))
    print(geom_model.get_prob_up(), geom_model.get_prob_down())
