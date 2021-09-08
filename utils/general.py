import pandas as pd
import numpy as np
import warnings
from numba import jit

# decorator class
class Decorators:
    
    # decorator for checking element type
    @staticmethod
    def check_element_type(func):
        def wrapper(*args, **kwargs):
            if 'element_type' in kwargs:
                element_type = kwargs['element_type'].upper()
                if element_type not in ['HERV', 'LTR']:
                    raise ValueError('Argument element_type should be one of (HERV, LTR)')
                kwargs['element_type'] = element_type.upper()
            elif any([arg.upper() == 'HERV' or arg.upper() == 'LTR' for arg in args if type(arg) == str]):
                args = [arg.upper() if type(arg) == str and (arg.upper() == 'HERV' or arg.upper() == 'LTR') else arg for arg in args]
            return func(*args, **kwargs)
        return wrapper
            

# functions for indexing genomic locations
class GenomeLookup:

    # function for masking elements in a data frame that are within a specified, start, end range
    @staticmethod
    @jit(nopython = True)
    def in_region(start_sites: np.ndarray, end_sites: np.ndarray, 
                  start: int, end: int) -> np.ndarray:
        return (start_sites <= end) & (end_sites >= start)
    
    # vectorised version of in_region, where start/end can be np arrays (returns 2d array)
    @staticmethod
    def in_region_vct(start_sites: np.ndarray, end_sites: np.ndarray, 
                      start: np.ndarray, end: np.ndarray) -> np.ndarray:
        return (start_sites[:,None] <= end[None,:]) & (end_sites[:,None] >= start[None,:])
    
    # find the closest gene to a given HERV element (row in mhc_HERVs)
    @staticmethod
    def find_closest_gene(HERV: pd.Series, ann: pd.DataFrame) -> tuple[str, str, str, str]:
        start, end = HERV[['Start', 'End']]
        overlapping_mask = GenomeLookup.in_region(ann['Start'].to_numpy(), ann['End'].to_numpy(), start, end)
        # if any overlapping gene is found, find gene with the max overlap
        if overlapping_mask.any():
            overlapping = ann[overlapping_mask]
            overlap_lengths = overlapping.apply(lambda gene: min(gene['End'], end) - max(gene['Start'], start), axis = 1)
            max_overlap_mask = overlap_lengths == overlap_lengths.max()
            closest_gene = overlapping.loc[max_overlap_mask, :]
            results = (closest_gene['name'].values[0], 0, 'Overlapping', closest_gene['Strand'].values[0])
        # otherwise find the closest gene
        else:
            # get distances between all possible combinations of HERV/human start/end
            start_to_start = (start - ann['Start']).abs().to_numpy()
            start_to_end = (start - ann['End']).abs().to_numpy()
            end_to_start = (end - ann['Start']).abs().to_numpy()
            end_to_end = (end - ann['End']).abs().to_numpy()
            # combine all distance arrays into a single 2D array, get minimum distance to each human gene
            all_distances = np.array([start_to_start, start_to_end, end_to_start, end_to_end]).transpose()
            min_distances = np.amin(all_distances, axis = 1)
            # find the gene with the minimum distance
            min_distance_mask = min_distances == min_distances.min()
            closest_gene = ann.loc[min_distance_mask, :].iloc[0, :]
            if closest_gene['Start'] < start:
                location = 'Downstream' if closest_gene['Strand'] == '+' else 'Upstream'
            else:
                location = 'Upstream' if closest_gene['Strand'] == '+' else 'Downstream'
            distance = min_distances.min()
            results = (closest_gene['name'], distance, location, closest_gene['Strand'])
        return results
    
class Annotation:
    
    def __init__(self,
                 human_ann: str = None,
                 HERV_ann: str = None,
                 LTR_ann: str = None,
                 drop_LINEs: bool = True, 
                 chr_lengths: list[int] = [248956422, 242193529, 198295559, 190214555, 181538259, 
                                           170805979, 159345973, 145138636, 138394717, 133797422, 
                                           135086622, 133275309, 114364328, 107043718, 101991189, 
                                           90338345, 83257441, 80373285, 58617616, 64444167, 
                                           46709983, 50818468]):
        
        # get chromosome lengths and start sites
        self.chr_lengths = chr_lengths
        self.chr_length_dict = {i+1: chr_lengths[i] for i in range(len(chr_lengths))}
        
        self.chr_end_sites = np.cumsum(self.chr_lengths, dtype = 'int64')
        self.chr_start_sites = np.roll(self.chr_end_sites, 1) + 1
        self.chr_start_sites[0] = 0
        
        self.chr_start_site_dict = {f'chr{i+1}': start_site for i, start_site in enumerate(self.chr_start_sites)}
        
        # import human annotations
        if human_ann is None:
            self.human_annotations = pd.read_pickle('data/gencode_v33.pkl')
        else:
            self.human_annotations = pd.read_csv(human_ann, sep = '\t')
            self.human_annotations.columns[:4] = ['Chr', 'Start', 'End', 'Name']
               
        # import HERV annotations
        if HERV_ann is None:
            self.HERV_annotations = pd.read_pickle('data/HERVs.pkl')
            
            # drop L1 elements if specified
            if drop_LINEs == True:
                self.HERV_annotations = self.HERV_annotations[self.HERV_annotations['Source'] != 'l1base']
                
        else:
            self.HERV_annotations = pd.read_csv(HERV_ann, sep = '\t')
            self.HERV_annotations.columns[:4] = ['Chr', 'Start', 'End', 'Name']
            
        # import LTR annotations    
        if LTR_ann is None:
            self.LTR_annotations = pd.read_pickle('data/LTRs.pkl')
        else:
            self.LTR_annotations = pd.read_csv(LTR_ann, sep = '\t')
            
        # isolate autosomal HERVs and LTRs
        self.autosomal_HERVs = self.HERV_annotations[(self.HERV_annotations['Chr'] != 'chrX') & (self.HERV_annotations['Chr'] != 'chrY')]
        self.autosomal_LTRs = self.LTR_annotations[(self.LTR_annotations['Chr'] != 'chrX') & (self.LTR_annotations['Chr'] != 'chrY')]
        
    def loadRegion(self,
                   start: int,
                   end: int, 
                   chrom: int,
                   gene_annotation: pd.DataFrame = None,
                   HERV_annotation: pd.DataFrame = None,
                   LTR_annotation: pd.DataFrame = None):
        
        warnings.filterwarnings('ignore', 'This pattern has match groups')
        
        if chrom not in self.chr_length_dict:
            raise ValueError('Chromosome {} not found (remember only autosomes are accepted).'.format(chrom))
        elif start > self.chr_length_dict[chrom]:
            raise ValueError('Start site is greater than the length of chromosome {}.'.format(chrom))
        elif end > self.chr_length_dict[chrom]:
            raise ValueError('End site is greater than the length of chromosome {}.'.format(chrom))
        elif start < 0:
            raise ValueError('Start site should be a positive integer.')
        elif end < 0:
            raise ValueError('End site should be a positive integer.')
        if start > end:
            raise ValueError('Start site should be less than end site.')
        
        self.start = int(start)
        self.end = int(end)
        self.region_size = abs(self.end - self.start)
        self.chrom = int(chrom)
        
        # isolate LTR elements in the region
        if gene_annotation is None:
            genes_in_region_mask = ((GenomeLookup.in_region(self.human_annotations['Start'].to_numpy(), 
                                                            self.human_annotations['End'].to_numpy(),
                                                            self.start, self.end)) &
                                    (self.human_annotations['Chr'].str.contains('chr{}($|_)'.format(self.chrom))))
            self.genes_in_region = self.human_annotations[genes_in_region_mask].copy()
            
        else:
            genes_in_region_mask = ((GenomeLookup.in_region(gene_annotation['Start'].to_numpy(), 
                                                            gene_annotation['End'].to_numpy(),
                                                            self.start, self.end)) &
                                    (gene_annotation['Chr'].str.contains('chr{}($|_)'.format(self.chrom))))
            self.genes_in_region = gene_annotation[genes_in_region_mask].copy()
            
        
        # isolate HERV elements in the region
        if HERV_annotation is None:
            HERVs_in_region_mask = ((GenomeLookup.in_region(self.HERV_annotations['Start'].to_numpy(), 
                                                            self.HERV_annotations['End'].to_numpy(),
                                                            self.start, self.end)) &
                                    (self.HERV_annotations['Chr'].str.contains('chr{}($|_)'.format(self.chrom))))
            self.HERVs_in_region = self.HERV_annotations[HERVs_in_region_mask].copy()
            
        else:
            HERVs_in_region_mask = ((GenomeLookup.in_region(HERV_annotation['Start'].to_numpy(), 
                                                            HERV_annotation['End'].to_numpy(),
                                                            self.start, self.end)) &
                                    (HERV_annotation['Chr'].str.contains('chr{}($|_)'.format(self.chrom))))
            self.HERVs_in_region = HERV_annotation[HERVs_in_region_mask].copy()
            
        # isolate LTR elements in the region
        if LTR_annotation is None:
            LTRs_in_region_mask = ((GenomeLookup.in_region(self.LTR_annotations['Start'].to_numpy(), 
                                                           self.LTR_annotations['End'].to_numpy(),
                                                           self.start, self.end)) &
                                   (self.LTR_annotations['Chr'].str.contains('chr{}($|_)'.format(self.chrom))))
            self.LTRs_in_region = self.LTR_annotations[LTRs_in_region_mask].copy()
        else:
            LTRs_in_region_mask = ((GenomeLookup.in_region(LTR_annotation['Start'].to_numpy(), 
                                                            LTR_annotation['End'].to_numpy(),
                                                            self.start, self.end)) &
                                    (LTR_annotation['Chr'].str.contains('chr{}($|_)'.format(self.chrom))))
            self.LTRs_in_region = LTR_annotation[LTRs_in_region_mask].copy()