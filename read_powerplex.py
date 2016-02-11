import collections
import numpy
import sys

def read_genotypes(input_file, conv_to_likelihood=False):
    data = open(input_file, "r")
    line = data.readline()
    while line[0] == "#":
        line = data.readline()

    # Read and store STR genotypes
    marker_names  = line.strip().split()[3:]
    str_genotypes = dict((marker,{}) for marker in marker_names)
    for line in data:
        tokens = line.strip().split()
        if len(tokens) != len(marker_names)+3:
            if len(tokens) != 0:
                print("Ignoring calls for %s as it has an unusual number of fields"%tokens[1])
        else:
            sample = tokens[1]
            for j in xrange(len(marker_names)):
                try:
                    gt = int(tokens[3+j])                    
                    str_genotypes[marker_names[j]][sample] = gt
                except ValueError:
                    print("Using missing genotype for %s"%tokens[3+j])
    data.close()

    for marker in str_genotypes:
        # Determine 'center' of distribution, which by default is the median allele
        if len(str_genotypes[marker].keys()) %2 == 0:
            center = int(numpy.median(str_genotypes[marker].values()[1:]))
        else:
            center = int(numpy.median(str_genotypes[marker].values()))
        
        # Convert lengths relative to center
        for sample in str_genotypes[marker]:
            str_genotypes[marker][sample] -= center

    # Make each (marker, sample) key map to a dictionary containing the STR genotype
    # and an associated probability of 1.0
    if conv_to_likelihood:
        for marker in str_genotypes:
            for sample in str_genotypes[marker]:
                str_genotypes[marker][sample] = {str_genotypes[marker][sample]:1.0}


    return str_genotypes

def plot_str_distribution(str_gts, locus_name, output_pdf):
    fig    = plt.figure()
    ax     = fig.add_subplot(111)
    counts = collections.defaultdict(int)
    for gt in str_gts.values():
        counts[gt] += 1
    x, y   = map(list, zip(*sorted(counts.items(), key = lambda x: x[0])))
    ncalls = sum(y)
    y      = numpy.array(y)*1.0/ncalls
    ax.bar(x, y, align='center')
    ax.set_xlabel("# STR Repeats")
    ax.set_ylabel("Frequency")
    ax.set_title(locus_name + r" ($n_{calls}=%d$)"%(ncalls))
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    output_pdf.savefig(fig)
    plt.close(fig)
