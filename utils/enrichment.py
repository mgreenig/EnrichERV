import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import matplotlib as mpl
from utils.general import Annotation, GenomeLookup, Decorators
from scipy.stats import hypergeom

class HERVEnrichment(Annotation):
    
    # TFBS_ann should be a bed file with four columns
    def __init__(self, 
                 TFBS_ann: str = None,
                 human_ann: str = None,
                 HERV_ann: str = None,
                 LTR_ann: str = None,
                 TFBS_TF_col: str = 'TF',
                 TFBS_chr_col: str = 'Chr',
                 drop_LINEs: bool = True):
        
        super().__init__(human_ann = human_ann, HERV_ann = HERV_ann, LTR_ann = LTR_ann, drop_LINEs = drop_LINEs)
        
        # import remap data set
        if TFBS_ann is None:
            self.TFBS = pd.read_parquet('data/remap2020_hg38.pqt')
        else:
            self.TFBS = pd.read_csv(TFBS_ann, sep = '\t')
            self.TFBS.columns[:4] = [TFBS_chr_col, 'Start', 'End', TFBS_TF_col]
            
        self.TFBS_start_sites = self.TFBS['Start'].to_numpy()
        self.TFBS_end_sites = self.TFBS['End'].to_numpy()

        self.TFBS_TF_col = TFBS_TF_col
        self.TFBS_chr_col = TFBS_chr_col
    
    # calculate enrichment of a TF's binding site in HERVs (compared to all locations)
    @staticmethod
    def _TFHERVEnrichment(TF: str, 
                          TFBS_subset: np.ndarray, 
                          TFBS_bg: np.ndarray) -> float:
        M = len(TFBS_bg)
        N = len(TFBS_subset)
        l = int((TFBS_bg == TF).sum())
        n = int((TFBS_subset == TF).sum())
        
        if n > 0:
            p_value = hypergeom.sf(n - 1, M, l, N)    
        else:
            p_value = 1
            
        results = pd.Series([l, n, p_value], 
                            index = ['num_TF_peaks', 
                                     'num_TF_peaks_in_HERVs', 
                                     'p'])
        return results
    
    # calculate enrichment of a HERV feature amongst a subset of HERVs
    @staticmethod
    def _HERVFeatureEnrichment(feature: str, 
                               HERV_subset: pd.DataFrame, 
                               HERV_bg: pd.DataFrame, 
                               colname: str) -> float:
        
        M = len(HERV_bg)
        N = len(HERV_subset)
        l = int(HERV_bg[colname].str.contains(feature).sum())
        n = int(HERV_subset[colname].str.contains(feature).sum())
        
        if n > 0:
            p_value = hypergeom.sf(n - 1, M, l, N)    
        else:
            p_value = 1
            
        results = pd.Series([l, n, p_value], 
                            index = ['num_background_HERVs_with_feature', 
                                     'num_subset_HERVs_with_feature',
                                     'p'])
        return results
    
    # calculate enrichment of TFBS in specific HERV family
    @staticmethod
    def _TFenrichmentHERVfamily(family: str, 
                                TFBS_family_subset: np.ndarray, 
                                TFBS_family_bg: np.ndarray) -> float:
        
        M = len(TFBS_family_bg)
        N = len(TFBS_family_subset)
        l = int((TFBS_family_bg == family).sum())
        n = int((TFBS_family_subset == family).sum())
        if n > 0:
            p_value = hypergeom.sf(n - 1, M, l, N)    
        else:
            p_value = 1
        results = pd.Series([l, n, p_value], 
                            index = ['num_all_peaks_in_family', 
                                     'num_TF_peaks_in_family', 
                                     'p'])
        return results
    
    # correct p-values with Benjamini-Hochberg procedure
    @staticmethod
    def _BHCorrection(results: pd.DataFrame) -> pd.Series:
        results = results.sort_values('p')
        p_values = results['p']
        # adjust p values
        adj_p_values = p_values * len(p_values) / np.arange(1, len(p_values) + 1)
        for i, p in enumerate(adj_p_values):
            adj_p_values[i] = min(adj_p_values[i:])
            if p > 1:
                adj_p_values[i] = 1
        results['adj_p'] = adj_p_values
        return results
    
    # get the index of an element from a set with maximum overlap with a given sequence start, end
    @staticmethod
    def _getMaxOverlap(starts, ends, HERV_starts, HERV_ends):
        max_starts = np.maximum(starts[:,None], HERV_starts[None, :])
        min_ends = np.minimum(ends[:,None], HERV_ends[None, :])
        overlaps = min_ends - max_starts
        max_overlap_idx = np.argmax(overlaps, axis = 1)
        return max_overlap_idx
    
    # function for finding TFs with at least one overlapping binding site with a given HERV
    @Decorators.check_element_type
    def findOverlappingTFBS(self, element_type = 'LTR') -> list[str]:
        
        if element_type == 'HERV':
            overlapping_mask = GenomeLookup.in_region_vct(self.HERVs_in_region['Start'].to_numpy(), 
                                                          self.HERVs_in_region['End'].to_numpy(),
                                                          self.TFBS_in_region['Start'].to_numpy(), 
                                                          self.TFBS_in_region['End'].to_numpy())
        else:
            overlapping_mask = GenomeLookup.in_region_vct(self.LTRs_in_region['Start'].to_numpy(), 
                                                          self.LTRs_in_region['End'].to_numpy(),
                                                          self.TFBS_in_region['Start'].to_numpy(), 
                                                          self.TFBS_in_region['End'].to_numpy())
            
        # loop through HERVs in the mask, save overlapping transcription factors
        all_overlapping_TFs = []
        for mask in overlapping_mask:
            if mask.any():
                overlapping_TFs = self.HERVs_in_region.loc[mask, self.TFBS_TF_col].str.cat(sep = '; ')
                overlapping_TFs = ' ' + overlapping_TFs
            else:
                overlapping_TFs = None
            all_overlapping_TFs.append(overlapping_TFs)
        return all_overlapping_TFs
    
    def loadRegion(self,
                   start: int,
                   end: int, 
                   chrom: int,
                   annotation: pd.DataFrame = None):
        
        super().loadRegion(start, end, chrom, annotation)
        warnings.filterwarnings('ignore', 'This pattern has match groups')
        self.TFBS_in_region = self.TFBS[GenomeLookup.in_region(self.TFBS_start_sites, 
                                                               self.TFBS_end_sites, 
                                                               self.start, self.end) &
                                        self.TFBS[self.TFBS_chr_col].str.contains('chr{}($|_)'.format(self.chrom))].copy()
        
        self.region_TFBS_start_sites = self.TFBS_in_region['Start'].to_numpy()
        self.region_TFBS_end_sites = self.TFBS_in_region['End'].to_numpy()
        self.region_HERV_start_sites = self.HERVs_in_region['Start'].to_numpy()
        self.region_HERV_end_sites = self.HERVs_in_region['End'].to_numpy()
        self.region_LTR_start_sites = self.LTRs_in_region['Start'].to_numpy()
        self.region_LTR_end_sites = self.LTRs_in_region['End'].to_numpy()
        
        # find if each TFBS overlaps with region HERVs and LTRs
        HERV_overlapping_TFBS_mask = GenomeLookup.in_region_vct(self.region_TFBS_start_sites, 
                                                                self.region_TFBS_end_sites,
                                                                self.region_HERV_start_sites, 
                                                                self.region_HERV_end_sites)
        
        self.TFBS_in_region['in_HERV'] = HERV_overlapping_TFBS_mask.any(axis = 1)
        
        LTR_overlapping_TFBS_mask = GenomeLookup.in_region_vct(self.region_TFBS_start_sites, 
                                                               self.region_TFBS_end_sites,
                                                               self.region_LTR_start_sites, 
                                                               self.region_LTR_end_sites)
        
        self.TFBS_in_region['in_LTR'] = LTR_overlapping_TFBS_mask.any(axis = 1)
    
    @Decorators.check_element_type
    def TFEnrichmentAnalysis(self, 
                             TFs_of_interest: list = None, 
                             element_type: str = 'LTR') -> pd.DataFrame:
        
        overlapping_column = 'in_HERV' if element_type == 'HERV' else 'in_LTR'
    
        if TFs_of_interest is None:
            TFBS = self.TFBS_in_region
        else:
            TFBS = self.TFBS_in_region[self.TFBS_in_region[self.TFBS_TF_col].isin(TFs_of_interest)]
            if len(TFBS) == 0:
                raise ValueError('None of the specified transcription factors were found in the annotation data')
            
        # get all TF binding sides in MHC LTRs
        TFs_in_region = list(set([tf for tf in TFBS[self.TFBS_TF_col]]))
        TFs_in_region = pd.Series(TFs_in_region)
        
        # calculate enrichment p-value for each TF in HERV TFBS vs all TFBS
        TFBS_in_HERVs = TFBS[TFBS[overlapping_column]].copy()
        
        if len(TFs_in_region) > 0:
            
            TFBS_HERV_enrichment_results = TFs_in_region.apply(
                lambda TF: self._TFHERVEnrichment(TF, TFBS_in_HERVs[self.TFBS_TF_col], TFBS[self.TFBS_TF_col])
            )
            TFBS_HERV_enrichment_results = pd.DataFrame(TFBS_HERV_enrichment_results)
            
            TFBS_HERV_enrichment_results.index = TFs_in_region
            
            # correct p-values with benjamini-hochberg
            TFBS_HERV_enrichment_results = self._BHCorrection(TFBS_HERV_enrichment_results)
            
        else:
            
            TFBS_HERV_enrichment_results = pd.DataFrame(columns = ['num_binding_sites', 'num_binding_sites_in_HERVs', 'p', 'adj_p'])
            print('No TFs have binding sites in this region.')
            
        return TFBS_HERV_enrichment_results
    
    
    @Decorators.check_element_type
    def TFHERVFamilyEnrichmentAnalysis(self, 
                                       TF: str,
                                       background_TFs: list = None,
                                       element_type: str = 'LTR') -> pd.DataFrame:
        
        overlapping_column = 'in_HERV' if element_type == 'HERV' else 'in_LTR'
        HERVs = self.HERVs_in_region if element_type == 'HERV' else self.LTRs_in_region
        
        if 'family' not in HERVs.columns:
            raise ValueError(f'{element_type} annotation does not contain a "family" column. Either add the relevant column or use the default annotation.')
        
        if background_TFs is None:
            TFBS = self.TFBS_in_region
            TFs_of_interest = TFBS[self.TFBS_TF_col]
        else:
            TFBS = self.TFBS_in_region[self.TFBS_in_region[self.TFBS_TF_col].isin(background_TFs)]
            if len(TFBS) == 0:
                raise ValueError('None of the specified transcription factors were found in the annotation data')
            
        overlapping_TFBS = TFBS[TFBS[overlapping_column]].copy()
        overlapping_TFBS = self.getClosestHERVs(overlapping_TFBS, HERVs)
        
        # get HERV families for all transcription factor binding sites of interest in the region
        TFBS_HERV_families = pd.Series(HERVs['family'].unique())

        # subset TFBS that overlap with HERVs
        TF_HERV_TFBS = overlapping_TFBS[overlapping_TFBS[self.TFBS_TF_col] == TF].copy()
        TF_HERV_TFBS = self.getClosestHERVs(TF_HERV_TFBS, HERVs)
        
        if len(TF_HERV_TFBS) == 0:
            raise ValueError('No HERV-overlapping binding sites for the specified transcription factor were found in the region of interest')
        
        family_enrichment_pvalues = TFBS_HERV_families.apply(
            lambda fam: self._TFenrichmentHERVfamily(fam, TF_HERV_TFBS['family'].to_numpy(), overlapping_TFBS['family'].to_numpy())
        )
        family_enrichment_pvalues.index = TFBS_HERV_families
        family_enrichment_results = self._BHCorrection(family_enrichment_pvalues)
        
        return family_enrichment_results
    
    # exports sequences of a TF's binding sites that overlap with HERVs to a fasta file
    @Decorators.check_element_type
    def exportOverlappingTFBS(self, TF: str, 
                              chromosome_sequence: str,
                              element_type: str = 'LTR'):
        
        # get sequences for DUX4 and EZH2 TF binding sites
        with open(chromosome_sequence, 'r') as file:
            lines = [l.strip() for l in file.readlines()]
            seq = ''.join(lines[1:])
            
        HERVs = self.HERVs_in_region if element_type == 'HERV' else self.LTRs_in_region
        
        # subset TFBS that overlap with HERVs, get closest HERV
        TF_TFBS = self.TFBS_in_region[self.TFBS_in_region[self.TFBS_TF_col] == TF].copy()
        
        assert len(TF_TFBS) > 0, 'No binding sites for the specified transcription factor were found in the region of interest'
        
        # get start/end for closest LTR, calculate start/end for TFBS/LTR overlap
        TF_TFBS = self.getClosestHERVs(TF_TFBS, HERVs)
        TF_TFBS[['closest_LTR_start', 'closest_LTR_end']] = HERVs.iloc[TF_TFBS['closest_LTR_idx']][['Start', 'End']].values
        TF_TFBS['LTR_motif_start'] = TF_TFBS.apply(lambda TFBS: max(TFBS['Start'], TFBS['closest_LTR_start']), axis = 1)
        TF_TFBS['LTR_motif_end'] = TF_TFBS.apply(lambda TFBS: max(TFBS['End'], TFBS['closest_LTR_end']), axis = 1)
        TF_TFBS_seqs = TF_TFBS.apply(lambda BS: seq[BS['LTR_motif_start']:BS['LTR_motif_end']], axis = 1)
        
        with open('data/{}_HERV_TFBS.fasta'.format(TF), 'w') as file:
            for i in TF_TFBS_seqs.index:
                start, end = self.TFBS_in_region.loc[i, ['Start', 'End']]
                header = '\n>{}_chr{}_{}-{}\n'.format(TF, self.chrom, start, end)
                file.write(header)
                seq = TF_TFBS_seqs[i]
                chunked_seq = [seq[i:i+80] for i in range(0, len(seq), 80)]
                joined_seq = '\n'.join(chunked_seq)
                file.write(joined_seq)
             
    def getClosestHERVs(self, TFBS_ann: pd.DataFrame, HERVs: pd.DataFrame):
        
        # get closest LTR to all LTR-overlapping binding sites in MHC + corresponding LTR family
        closest_HERV_idx = self._getMaxOverlap(starts = TFBS_ann['Start'].to_numpy(),
                                               ends = TFBS_ann['End'].to_numpy(),
                                               HERV_starts = HERVs['Start'].to_numpy(),
                                               HERV_ends = HERVs['End'].to_numpy())
                                            
        TFBS_ann['closest_HERV_id'] = HERVs.iloc[closest_HERV_idx]['id'].values
        
        TFBS_ann['family'] = HERVs.iloc[closest_HERV_idx]['family'].values
        
        return TFBS_ann
    
    @staticmethod
    def TFEnrichmentPlot(enrichment_results: pd.DataFrame, 
                         top_n: int = 20, 
                         save: bool = True, 
                         path: str = None,
                         xlab: str = '\nProportion of TF binding sites overlapping with LTRs', 
                         title: str = 'Transcription factor enrichment in LTR elements'):
        
        # get top n transcription factors ranked by pvalue
        top_n_TFs_sorted = enrichment_results.sort_values('adj_p').iloc[0:min(top_n, len(enrichment_results))]
        top_n_TF_df = pd.DataFrame({'adj_p': top_n_TFs_sorted['adj_p'], 
                                    'proportion': top_n_TFs_sorted['num_TF_peaks_in_HERVs'] / top_n_TFs_sorted['num_TF_peaks'],
                                    'count': top_n_TFs_sorted['num_TF_peaks_in_HERVs']}, 
                                   index = top_n_TFs_sorted.index)
        top_n_TF_df['log_adj_p'] = np.log10(top_n_TF_df['adj_p'])
        
        top_n_TF_df.sort_values(by = 'log_adj_p', ascending = False, inplace = True)
        
        fig = plt.figure(figsize = (10, 8))
        plt.rc('grid', linestyle='dashed')
        ax = fig.add_subplot(1,1,1)
        ax.grid(True)
        ax.set_axisbelow(True)
        plot = plt.scatter(top_n_TF_df['proportion'], top_n_TF_df.index, 
                           c = -top_n_TF_df['log_adj_p'], cmap = 'cool', 
                           s = 0.50*(top_n_TF_df['count'])**(1.3), edgecolor = 'black', 
                           linewidth = 0.6)
        plt.tick_params(axis='both', which='both', bottom=False, top=False, left = False, labelsize = 13)
        plt.ylabel('Transcription factor\n', fontsize = 16)
        plt.xlabel(xlab, fontsize = 16)
        plt.title(title, fontsize = 16)
        kw = dict(prop="sizes", num=5, func = lambda s: (s/0.50)**(1/1.3))
        plt.legend(*plot.legend_elements(**kw), title = 'Number of\nLTR-overlapping\nbinding sites', 
                   bbox_to_anchor = [1.22, 1], fontsize = 10, title_fontsize = 12, labelspacing = 1.2, frameon = False)
        axs = fig.add_axes([0.94, 0.15, 0.02, 0.34])
        cbar = plt.colorbar(shrink = 0.5, cax = axs)
        cbar.set_label('-log10(adj. p-value)', y = 1.07, rotation=0, fontsize = 12)
        if save == True:
            if path is not None:
                plt.savefig(path, dpi = 300, facecolor = 'white', bbox_inches = 'tight')
            else:
                plt.savefig('enrichment_plot.png', dpi = 300, facecolor = 'white', bbox_inches = 'tight')
                
        plt.show()