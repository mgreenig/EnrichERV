import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom
from matplotlib.patches import Rectangle
from numba import jit
from utils.general import GenomeLookup, Annotation, Decorators
import warnings

class HERVDensity(Annotation):
    
    @staticmethod
    def addCumulativeIndex(regions, chr_start_site_dict):
        
        regions[['CumStart', 'CumEnd']] = 0
        for i in regions.index:
            chrom = regions.loc[i, 'Chr'].split('_')[0].strip('chr')
            regions.loc[i, 'CumStart'] = regions.loc[i, 'Start'] + chr_start_site_dict[chrom]
            regions.loc[i, 'CumEnd'] = regions.loc[i, 'End'] + chr_start_site_dict[chrom]
        
        return regions
     
    def __init__(self, 
                 human_ann: str = None,
                 HERV_ann: str = None,
                 LTR_ann: str = None,
                 excl_regions: str = '../data/hg38_centromeres.txt',
                 drop_LINEs: bool = True):
        
        # load annotations using the parent class
        super().__init__(human_ann = human_ann, HERV_ann = HERV_ann, LTR_ann = LTR_ann, drop_LINEs = drop_LINEs)
        
        # read in excluded regons, filter for autosomes, add cumulative start/end sites
        excl_regions = pd.read_csv(excl_regions, sep = '\t', names = ['Bin', 'Chr', 'Start', 'End', 'ID'])
        excl_regions = excl_regions[excl_regions['Chr'].isin(self.chr_start_site_dict.keys())]
        excl_regions_chr_start_sites = np.array([self.chr_start_site_dict[chrom] for chrom in excl_regions['Chr'].values])
        excl_regions['CumStart'] = excl_regions['Start'] + excl_regions_chr_start_sites
        excl_regions['CumEnd'] = excl_regions['End'] + excl_regions_chr_start_sites
    
        self.excl_regions = excl_regions
        excl_size = (self.excl_regions['End'] - self.excl_regions['Start']).abs().sum()
        
        self.human_autosome_size =  sum(self.chr_lengths) - excl_size
        
    @Decorators.check_element_type
    def binomial(self, chromosome: int = None, element_type: str = 'HERV') -> dict:
        
        HERVs_in_region = self.HERVs_in_region if element_type == 'HERV' else self.LTRs_in_region
        all_HERVs = self.HERV_annotations if element_type == 'HERV' else self.LTR_annotations
        
        # calculate proportion of the genome/chromosome occupied by the region
        if chromosome is None:
            bg_HERVs = all_HERVs[(all_HERVs['Chr'] != 'chrX') & (all_HERVs['Chr'] != 'chrY')]
            p_null = self.region_size / self.human_autosome_size
        else:
            bg_HERVs = all_HERVs[all_HERVs['Chr'].str.contains('chr{}'.format(chromosome))]
            p_null = self.region_size / self.chr_length_dict[chromosome]
            if len(HERVs) == 0:
                raise ValueError('No HERVs found on specified chromosome')
                
        # calculate p-value from binomial
        p_value = 1 - binom.cdf(len(HERVs_in_region) - 1, len(bg_HERVs), p_null)
        results = {'num_{}s'.format(element_type): len(HERVs_in_region), 'p': p_value, 
                   'region_proportion': p_null, 'n': len(bg_HERVs), 'k': len(HERVs_in_region)}
        
        return results
    
    # sample a contiguous region of a given size from the human genome
    def _sampleRegions(self, 
                       region_size: int, 
                       n: int, 
                       chromosomes: list[int],
                       rng: np.random.Generator,
                       filter_sites: bool = True) -> tuple[list, list, list]:
        
        # convert chromosome list to array
        chromosomes = np.array(chromosomes)
        # adjust chromosome lengths to get the number of start sites that can be sampled
        adj_chr_lengths = np.array([size - region_size + 1 for i, size in enumerate(self.chr_lengths) if i+1 in chromosomes],
                                   dtype = 'int64')
        
        # probability of choosing a region on a given chromosome is proportional to its length
        sample_probs = adj_chr_lengths / np.sum(adj_chr_lengths)
        sampled_chr = rng.choice(chromosomes - 1, size = n, replace = True, p = sample_probs)
        
        # get the adjusted lengths of all sampled chromosomes
        sampled_chr_lengths = adj_chr_lengths[sampled_chr]
        # get a uniformly-distributed number in (0, 1] for each sampled chromosome
        sampled_positions = rng.uniform(size = len(sampled_chr_lengths))
        
        # convert uniform variable to position on the chromosome
        start_sites = np.floor(sampled_positions * sampled_chr_lengths)
        end_sites = start_sites + region_size
        
        # get cumulative start and end sites
        cumulative_start_sites = self.chr_start_sites[sampled_chr] + start_sites
        cumulative_end_sites = cumulative_start_sites + region_size
        
        # only keep sites that do not overlap with excluded regions
        if filter_sites == True:
            sites_to_keep = np.array([i for i, (s, e) in enumerate(zip(cumulative_start_sites, cumulative_end_sites))
                                      if not GenomeLookup.in_region(self.excl_regions['CumStart'].to_numpy(), 
                                                                    self.excl_regions['CumEnd'].to_numpy(), s, e).any()])
            start = start_sites[sites_to_keep]
            end = start + region_size
            sampled_chr = sampled_chr[sites_to_keep] + 1
        else:
            start = start_sites
            end = start + region_size
            sampled_chr = sampled_chr + 1

        return start, end, sampled_chr

    # count number of HERVs/genes/LTRs in each of n sampled regions
    def monteCarlo(self, 
                   n: int, 
                   state: int = None,
                   chromosomes: list[int] = None,
                   filter_sites: bool = True,
                   track_families: bool = False) -> dict:
        
        if chromosomes is None:
            chromosomes = [i for i in range(1, 23)]
        else:
            assert all([isinstance(chrom, int) and chrom in range(1, 23) for chrom in chromosomes]), 'Chromosomes should be a list of integers taking values in the range 1-22'
        
        if not hasattr(self, 'region_size'):
            raise AttributeError('Class instance does not have region-related attributes: call loadRegion() first.')
        
        # hide pandas userwarnings
        warnings.simplefilter(action = 'ignore', category = UserWarning)
        
        # sample n regions
        rng = np.random.default_rng(seed = state)
        starts, ends, chroms = self._sampleRegions(self.region_size, n, chromosomes, 
                                                   rng = rng, filter_sites = filter_sites)
        sampled_regions = np.column_stack([starts, ends, chroms])
                
        # split sampled regions and annotations by chromosome, only keep start/end sites and convert to np.ndarray
        sampled_chromosomes = np.unique(sampled_regions[:,2],).astype('int32')
        sampled_sites_per_chrom = {i: sampled_regions[sampled_regions[:,2] == i] for i in sampled_chromosomes}
        HERV_locs_per_chrom = {i: self.HERV_annotations[self.HERV_annotations['Chr'].str.contains('chr{}($|_)'.format(i), regex = True)]
                               for i in sampled_chromosomes}
        gene_locs_per_chrom = {i: self.human_annotations[self.human_annotations['Chr'].str.contains('chr{}($|_)'.format(i), regex = True)]
                               for i in sampled_chromosomes}
        LTR_locs_per_chrom = {i: self.LTR_annotations[self.LTR_annotations['Chr'].str.contains('chr{}($|_)'.format(i), regex = True)]
                              for i in sampled_chromosomes}
        
        # initialise arrays for HERV counts
        HERV_counts = np.empty(shape = len(starts), dtype = 'int32')
        gene_counts = np.empty(shape = len(starts), dtype = 'int32')
        LTR_counts = np.empty(shape = len(starts), dtype = 'int32')
        
        if track_families == True:
            
             # get the HERV family profile of the sampled region
            if 'family' in self.LTRs_in_region.columns:
                LTR_family_profile = self.LTRs_in_region['family'].value_counts()
                LTR_family_counts = np.empty(shape = (len(starts), len(LTR_family_profile)), dtype = 'int32')
            
            if 'family' in self.HERVs_in_region.columns:
                HERV_family_profile = self.HERVs_in_region['family'].value_counts()
                HERV_family_counts = np.empty(shape = (len(starts), len(HERV_family_profile)), dtype = 'int32')
        
        # loop through sites, get the number of HERVs in each site
        idx = 0
        for chrom in sampled_sites_per_chrom:
            
            sites = sampled_sites_per_chrom[chrom]
            start_sites, end_sites = sites[:,0], sites[:,1]
            
            # count number of HERVs in region
            HERV_locs_in_chrom = HERV_locs_per_chrom[chrom]
            HERVs_in_region_mask = GenomeLookup.in_region_vct(start_sites, end_sites,
                                                              HERV_locs_in_chrom['Start'].to_numpy(),
                                                              HERV_locs_in_chrom['End'].to_numpy())
            HERV_region_counts = HERVs_in_region_mask.sum(axis = 1)
            HERV_counts[idx:(idx+len(start_sites))] = HERV_region_counts
                
            # count number of genes in region
            gene_locs_in_chrom = gene_locs_per_chrom[chrom]
            genes_in_region_mask = GenomeLookup.in_region_vct(start_sites, end_sites,
                                                              gene_locs_in_chrom['Start'].to_numpy(),
                                                              gene_locs_in_chrom['End'].to_numpy())
            gene_counts[idx:(idx+len(start_sites))] = genes_in_region_mask.sum(axis = 1)
            
            # count number of LTRs in region
            LTR_locs_in_chrom = LTR_locs_per_chrom[chrom]
            LTRs_in_region_mask = GenomeLookup.in_region_vct(start_sites, end_sites,
                                                             LTR_locs_in_chrom['Start'].to_numpy(),
                                                             LTR_locs_in_chrom['End'].to_numpy())
            LTR_region_counts = LTRs_in_region_mask.sum(axis = 1)
            LTR_counts[idx:(idx+len(start_sites))] = LTR_region_counts
            
            if track_families == True:
                
                # get HERV family profile in each region
                if 'family' in HERV_locs_in_chrom.columns:
                    HERV_families_in_chrom = HERV_locs_in_chrom['family'].to_numpy()
                    HERV_families_expanded = np.repeat([HERV_families_in_chrom], len(start_sites), axis = 0)
                    HERV_families_in_regions = HERV_families_expanded[HERVs_in_region_mask]
                    region_HERV_indices = np.cumsum(HERV_region_counts)
                    HERV_families_per_region = np.array([pd.Series(fams).value_counts().reindex(HERV_family_profile.index, fill_value = 0).to_numpy()
                                                         for fams in np.split(HERV_families_in_regions, region_HERV_indices)[:-1]])
                    HERV_family_counts[idx:(idx+len(start_sites))] = HERV_families_per_region
                    
                # get the LTR family profile of each region
                if 'family' in LTR_locs_in_chrom.columns:
                    LTR_families_in_chrom = LTR_locs_in_chrom['family'].to_numpy()
                    LTR_families_expanded = np.repeat([LTR_families_in_chrom], len(start_sites), axis = 0)
                    LTR_families_in_regions = LTR_families_expanded[LTRs_in_region_mask]
                    region_LTR_indices = np.cumsum(LTR_region_counts)
                    LTR_families_per_region = np.array([pd.Series(fams).value_counts().reindex(LTR_family_profile.index, fill_value = 0).to_numpy()
                                                        for fams in np.split(LTR_families_in_regions, region_LTR_indices)[:-1]])
                
                    LTR_family_counts[idx:(idx+len(start_sites))] = LTR_families_per_region
            
            idx = idx + len(start_sites)
            
        HERV_p_value = sum(HERV_counts >= len(self.HERVs_in_region)) / len(HERV_counts)
        gene_p_value = sum(gene_counts >= len(self.genes_in_region)) / len(gene_counts)
        LTR_p_value = sum(LTR_counts >= len(self.LTRs_in_region)) / len(LTR_counts)
        
        self.MC_results = {'HERV_counts': HERV_counts, 
                           'gene_counts': gene_counts, 
                           'LTR_counts': LTR_counts, 
                           'HERV_p_value': HERV_p_value, 
                           'gene_p_value': gene_p_value, 
                           'LTR_p_value': LTR_p_value, 
                           'regions': sampled_regions}
        
        if track_families == True:
            if 'family' in self.LTRs_in_region.columns:
                self.MC_results['LTR_family_counts'] = LTR_family_counts
            if 'family' in self.HERVs_in_region.columns:
                self.MC_results['HERV_family_counts'] = HERV_family_counts
        
        return self.MC_results
    
    
    # make a plot to compare sampled HERV counts with 
    @staticmethod
    @Decorators.check_element_type
    def plotDensity(MC_results: dict, 
                    binomial_results: dict, 
                    n_bins: int = 50,
                    max_counts: int = 100, 
                    savefig: bool = False,
                    element_type: str = 'HERV'):
        
        fig, axs = plt.subplots(1, 2, figsize = (14, 6), sharey = True)
        fig.add_subplot(111, frameon=False)
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        counts = [i for i in range(max_counts)]
        mhc_bar_pos = binomial_results['k'] // (max_counts / n_bins)
        binomial_probs = [binom.pmf(k, binomial_results['n'], binomial_results['p_null']) for k in counts]
        axs[0].bar(counts, binomial_probs, edgecolor = 'black', linewidth = 0.6)
        axs[0].set_title('Binomial model', fontsize = 18)
        mhc_bar = [elem for elem in axs[0].get_children() if isinstance(elem, Rectangle) and 
                   elem.get_height() == binom.pmf(binomial_results['k'], binomial_results['n'], binomial_results['p_null'])]
        mhc_bar[0].set_color('r')
        mhc_bar[0].set_edgecolor('black')
        axs[0].tick_params(axis = 'both', labelsize = 14)
        axs[1].hist(MC_results['{}_counts'.format(element_type)], bins = n_bins, density = True, edgecolor = 'black', range = (0, max_counts))
        histogram_bars = [elem for elem in axs[1].get_children() if isinstance(elem, Rectangle)]
        histogram_bars[int(mhc_bar_pos)].set_color('r')
        histogram_bars[int(mhc_bar_pos)].set_edgecolor('black')
        axs[1].set_title('Sampled regions', fontsize = 18)
        axs[1].tick_params(axis = 'x', labelsize = 14)
        plt.setp(axs, xlim=(0, max_counts))
        plt.xlabel('\nNumber of full-length HERV elements in region-sized windows', fontsize = 18)
        plt.ylabel('Frequency\n\n', fontsize = 18)
        plt.legend([histogram_bars[int(mhc_bar_pos)]], ['Region'], fontsize = 14)
        
        if savefig == True:
            plt.savefig('count_distribution.png', dpi = 300)
            
        plt.show()