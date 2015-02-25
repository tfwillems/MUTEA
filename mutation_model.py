import numpy
from scipy.stats import norm, geom
from numpy.linalg import matrix_power
from numpy import linalg as LA

'''
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
'''

class OUGeomSTRMutationModel:
    def __init__(self, p_geom, mu, beta, allele_range, max_step=None):
        if p_geom <= 0.0 or p_geom > 1:
            exit("Invalid geometric distribution probability = %f"%p_geom)

        if mu > 0.1 or mu <= 0:
            exit("Invalid mutation rate mu = %f"%mu)

        if beta < 0 or beta > 1.0:
            exit("Invalid length constraint beta = %f"%beta)

        if max_step is None:
            prob       = p_geom
            tot_rem    = 1.0
            max_step   = 0
            while tot_rem > 1e-6:
                tot_rem  -= prob
                prob     *= (1-p_geom)
                max_step += 1
            self.max_step = max_step
        else:
            self.max_step = max_step
        
        self.allele_range   = allele_range
        self.mu             = mu
        self.beta           = beta
        self.p_geom         = p_geom
        #self.max_step       = max_step
        self.init_transition_matrix()

        # Memoized matrix info
        self.prev_forward_tmrca  = None
        self.forward_matrix      = None
        
        # Based on properties of exact geometric distribution
        self.step_size_variance = (2-p_geom)/(p_geom**2)


    def init_transition_matrix(self):
        max_step     = self.max_step
        max_n        = self.allele_range + max_step
        min_n        = -self.allele_range - max_step
        N            = 2*self.allele_range + 2*max_step + 1
        trans_matrix = numpy.zeros((N, N))
        steps        = numpy.arange(1, 2*max_n+1, 1)
        step_probs   = geom.pmf(steps, self.p_geom)

        # Fill in transition matrix                                                                                                                                             
        for i in xrange(min_n+1, max_n):
            up_prob      = min(1, max(0, 0.5*(1-self.beta*self.p_geom*i)))
            down_prob    = 1.0-up_prob
            lrem         = sum(step_probs[i-min_n-1:])
            rrem         = sum(step_probs[max_n-i-1:])
            trans_matrix[:,i-min_n] = numpy.hstack((numpy.array([self.mu*down_prob*lrem]),
                                                    self.mu*down_prob*numpy.array(step_probs[:i-min_n-1][::-1]),
                                                    numpy.array([1-self.mu]),
                                                    self.mu*up_prob*numpy.array(step_probs[:max_n-i-1]),
                                                    numpy.array([self.mu*up_prob*rrem])))

        # Add boundaries to prevent probability leakage
        trans_matrix[:,0]     = 0
        trans_matrix[0,0]     = 1
        trans_matrix[:,N-1]   = 0
        trans_matrix[N-1,N-1] = 1
                        
        # Convert to matrix, or numpy functions won't do appropriate thing, and save for later use
        self.trans_matrix = numpy.matrix(trans_matrix)

        # Save various matrix-related variables
        self.min_n = min_n
        self.max_n = max_n
        self.N     = N

    def compute_forward_matrix(self, tmrca, allow_bigger_err=True):
        if tmrca != self.prev_forward_tmrca:
            self.prev_forward_tmrca = tmrca
            self.forward_matrix     = self.trans_matrix**tmrca
            numpy.clip(self.forward_matrix, 1e-10, 1.0, out=self.forward_matrix)

    def get_forward_matrix(self):
        return self.forward_matrix

    def get_forward_str_probs(self, start_allele, tmrca):
        self.compute_forward_matrix(tmrca)
        vec = numpy.zeros((self.N,1))                                   
        vec[start_allele-self.min_n] = 1           
        return numpy.array(self.forward_matrix.dot(vec).transpose())[0]

    def plot_transition_probabilities(self, x_vals, output_pdf, window_width=5):
        fig, axes = plt.subplots(1, len(x_vals), sharex=True, sharey=True)
        if len(x_vals) == 1:
            axes = [axes]
        for i in xrange(len(x_vals)):
            trans_probs = numpy.array(self.trans_matrix[:,x_vals[i]-self.min_n].transpose())[0]
            trans_probs = trans_probs[max(0, x_vals[i]-window_width-self.min_n): min(len(trans_probs), x_vals[i]+window_width+1-self.min_n)]
            x = range(max(0, x_vals[i]-window_width-self.min_n), min(self.max_n-self.min_n+2, x_vals[i]+window_width+1-self.min_n))
            x = numpy.array(x) + self.min_n - x_vals[i]
            axes[i].bar(x, trans_probs, align="center")
            axes[i].set_title("Allele=%d"%(x_vals[i]))
        map(lambda x: x.xaxis.set_ticks_position('bottom'), axes)
        map(lambda x: x.yaxis.set_ticks_position('left'), axes)
        map(lambda x: x.set_ylim((0.0, 0.8)), axes)
        axes[0].set_ylabel("Transition probabilities")
        axes[len(x_vals)/2].set_xlabel("STR Step Size")
        output_pdf.savefig(fig)
        plt.close(fig)

def main():
    numpy.set_printoptions(formatter={'float': '{: 0.3f}'.format}, threshold=numpy.nan, linewidth=200)
    allele_range = 5
    start_allele = 0
    mu           = 0.001
    beta         = 0.4
    p_geom       = 1.0
    mut_model = OUGeomSTRMutationModel(p_geom, mu, beta, allele_range)
    print(mut_model.trans_matrix**10000)
    print(mut_model.get_forward_str_probs(0, 1), 0, 1.0)
    print(mut_model.P)
    return
    

    
    allele_range = 12
    mu           = 0.0007909675
    beta         = 0.2466938013
    p_geom       = 0.9820623995
    
   
    #beta   = 0.15
    #p_geom = 0.5
    
    mu        = 1.0
    p_geom    = 0.9
    pp = PdfPages("step_profiles.pdf")
    for beta in [0.0, 0.1, 0.25]: 
        mut_model = OUGeomSTRMutationModel(p_geom, mu, beta, allele_range)
        x_vals = [-3, 0, 3]
        mut_model.plot_transition_probabilities(x_vals, pp, window_width=5)
    pp.close()
    return


    for tmrca in [1000]: #[1, 10, 100, 1000, 10000]:
        

        mut_model.compute_forward_matrix(tmrca)    
        f1_probs = numpy.clip(numpy.real(mut_model.get_forward_str_probs(start_allele, tmrca)), 0, 1.0)
        print(mut_model.forward_matrix)
        mut_model.forward_matrix = mut_model.trans_matrix**tmrca
        print(mut_model.forward_matrix)
        f2_probs = numpy.clip(numpy.real(mut_model.get_forward_str_probs(start_allele, tmrca)), 0, 1.0)
        #print(f1_probs)
        #print(f2_probs)
    return
    

    mus    = [0.01]
    betas  = numpy.linspace(0,   1.0, 100)
    pgeoms = numpy.linspace(0.5, 1.0, 80)
    X,Y = numpy.meshgrid(betas, pgeoms)

    log_10 = lambda x: numpy.log(x)/numpy.log(10)

    fig = plt.figure()
    for mu in mus:
        max_z, min_z = [], []
        for beta in betas:
            for p_geom in pgeoms:
                mut_model = OUGeomSTRMutationModel(p_geom, mu, beta, allele_range)
                mut_model.compute_forward_matrix(1000)
                min_val = numpy.min(mut_model.forward_matrix)
                max_val = numpy.max(mut_model.forward_matrix)
                #z.append(log_10(log_10(max_val)))
                max_z.append(max_val > 1)
                min_z.append(min_val < -1e-5)
        max_Z = numpy.array(max_z).reshape((len(betas), len(pgeoms)))
        min_Z = numpy.array(min_z).reshape((len(betas), len(pgeoms)))
        fig = plt.figure()
        ax1  = fig.add_subplot(121)
        ax1.imshow(max_Z)
        ax2 = fig.add_subplot(122)
        ax2.imshow(min_Z)
        #plt.contourf(X, Y, Z)
        #plt.colorbar()
        plt.show()



if __name__ == "__main__":
    main()

