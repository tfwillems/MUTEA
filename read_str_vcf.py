import collections
import genotypers
import operator
import vcf


# Returns a dictionary mapping each sample name to a dictionary 
# containing the observed repeat lengths and counts for that sample's reads
def get_str_read_counts(vcf_reader):
    # Attempt to read record from VCF
    try:
        record = vcf_reader.next()
    except StopIteration:
        return False, "N/A", -1, -1, -1, {}, -1, -1

    # Determine most common frame
    motif_len    = len(record.INFO['MOTIF'])
    frame_counts = collections.defaultdict(int)
    for sample in record:
        if sample['GT'] is not None:
            read_count_data  = map(lambda x: (int(x[0]), int(x[1])), map(lambda x: x.split("|"), sample['ALLREADS'].split(";")))
            tot_sample_reads = sum(map(lambda x: x[1], read_count_data))
            for str_length,read_count in read_count_data:
                frame_counts[str_length%motif_len] += read_count*1.0/tot_sample_reads 
    if len(frame_counts) == 0:
        return get_str_read_counts(vcf_reader)
    best_frame,value = max(frame_counts.items(), key=operator.itemgetter(1))

    # Construct a dictionary mapping sample names -> read count dict for each sample without out-of-frame reads
    sample_read_count_dict = {}
    in_frame_count,out_frame_count = 0,0
    for sample in record:
        if sample['GT'] is not None:
            read_count_data = map(lambda x: (int(x[0]), int(x[1])), map(lambda x: x.split("|"), sample['ALLREADS'].split(";")))
            read_sizes      = map(lambda x: x[0], read_count_data)
            all_in_frame    = all(map(lambda x: x%motif_len == best_frame, read_sizes))
            if all_in_frame:
                # Convert bp diffs to repeat diffs
                read_count_data = map(lambda x: ((x[0]-best_frame)/motif_len, x[1]), read_count_data)

                sample_read_count_dict[sample.sample] = dict(read_count_data)
                in_frame_count += 1
            else:
                out_frame_count += 1
    return True,record.CHROM,record.POS,record.INFO['END'],len(record.INFO['MOTIF']),sample_read_count_dict, in_frame_count, out_frame_count

def get_str_gts(vcf_reader):
    # Attempt to read record from VCF
    try:
        record = vcf_reader.next()
    except StopIteration:
        return False, {}, 0, 0, ""

    # Ignore missing and heterozygous genotypes
    all_lens  = []
    for sample in record:
        gts = sample['GT']
        if gts is not None:
            gt_a, gt_b = map(int, gts.split("/"))
            if gt_a == gt_b:
                length = len(record.alleles[gt_a])
                all_lens.append(length)

    # Determine most common frame 
    motif_len    = len(record.INFO['MOTIF'])
    all_lens     = sorted(all_lens)
    frame_counts = motif_len*[0]

    if len(all_lens) == 0:
        exit("No samples have a valid genotype for the STR of interest")

    for length in all_lens:
        frame_counts[length%motif_len] += 1
    best_frame, value = max(enumerate(frame_counts), key=operator.itemgetter(1))
    
    # Utilize median observed allele as 'center'
    all_lens = filter(lambda x: x%motif_len == best_frame, all_lens)
    all_lens = sorted(all_lens)
    center   = all_lens[len(all_lens)/2]
    min_str  = (all_lens[0]  - center)/motif_len
    max_str  = (all_lens[-1] - center)/motif_len

    # Determine a list of alleles that are in-frame with the most common frame
    in_frame_indices = []
    for i in xrange(len(record.alleles)):
        if record.alleles[i] is None:
            continue
        if len(record.alleles[i])%motif_len == best_frame:
            in_frame_indices.append(i)

    # Determine posterior genotype likelihoods for samples
    # Samples whose most probable genotype is missing, heterozygous or out-of-frame are not assigned any likelihoods
    # Lengths of genotypes are stored relative to the 'center' and in terms of the number of repeat differences
    gt_probs  = {}
    for sample in record:
        gts = sample['GT']
        if gts is not None:
            gt_a, gt_b = map(int, gts.split("/"))
            if gt_a == gt_b and len(record.alleles[gt_a])%motif_len == best_frame:
                # Use genotype likelihoods to assign posterior likelihoods to every homozygous in-frame STR genotype assuming a uniform genotype prior
                PLs = sample['PL']
                if not isinstance(PLs, list):
                    PLs = [PLs]
                Ls           = map(lambda x: 10**(-x/10.0), PLs)
                homoz_Ls     = map(lambda x: Ls[(x+1)*(x+2)/2 - 1], in_frame_indices) # Annoying VCF PL indexing [(0,0), (0,1), (1,1), (0,2), (1,2), (2,2) ...]
                tot_homoz_L  = sum(homoz_Ls)
                sample_probs = {}
                for index,i in enumerate(in_frame_indices): 
                    sample_probs[(len(record.alleles[i])-center)/motif_len] = homoz_Ls[index]*1.0/tot_homoz_L
                gt_probs[sample.sample] = sample_probs
    return True,gt_probs,min_str,max_str,(record.CHROM+":"+str(record.POS))



def compute_median(values, counts):
    total_count = sum(counts)
    target_val  = total_count/2.0
    for i in xrange(len(values)):
        if counts[i] > target_val:
            return values[i]
        target_val -= counts[i]
    return values[-1]

def counts_to_centalized_posteriors(sample_read_counts, p_geom, down, up):
    # Compute the minimum and maximum observed allele
    min_allele = min(map(lambda x: min(x.keys()), sample_read_counts.values()))
    max_allele = max(map(lambda x: max(x.keys()), sample_read_counts.values()))

    # Create the stutter genotyper
    genotyper = genotypers.EstStutterGenotyper()
    genotyper.create(down, up, p_geom, min_allele, max_allele)

    # Compute genotype posteriors using genotyper
    gt_posteriors = {}
    for sample,counts in sample_read_counts.items():
        gt_posteriors[sample] = genotyper.get_genotype_posteriors(counts)

    # Compute 'central' allele using the allele with the median posterior sum
    gt_counts = collections.defaultdict(int)
    for posteriors in gt_posteriors.values():
        for gt,prob in  posteriors.items():
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
    return gt_posteriors,min_allele-center,max_allele-center
