import gseapy as gp
import pandas as pd

from preprocess import read_gene_signature


# dataset_list = ["GSE2820", "GSE3307", "GSE9397", "GSE10760", "GSE15090", "GSE26852", "GSE36398", "GSE140261",
#                 "GSE115650", "GSE56787"]
# cDNA study was removed because it's not particularly useful for characterizing gene signatures
# (control and FSHD in the same sample)


def check_gene_signature(ssdf, signatures_dict):
    """
    Check for genes in signature that are not in the dataset
    :return: valid_signatures, a dictionary containing valid genes (gene also in expression file) from the gene
    signature groups
    """
    genes = ssdf['Gene_Symbol'].to_list()
    valid_signatures = {}

    for key in signatures_dict:
        signature_set = signatures_dict[key]
        check = all(item in genes for item in signature_set)
        print('\tAll %s in genes? ' % key, check)
        # If signature set not contained in gene, find the missing genes (and the number)
        missing_genes = [item for item in signature_set if item not in genes]
        print('\t', len(missing_genes), 'genes in %s but not in expression file' % key)
        print('\tMissing %s gene signature in expression file' % key, missing_genes)
        # Find the valid genes
        valid_genes = [item for item in signature_set if item not in missing_genes]
        num_valid_genes = len(valid_genes)
        valid_signatures[key] = valid_genes
        valid_signatures['Number of valid %s' % key] = num_valid_genes
    return valid_signatures


def get_ssGSEA(signatures_dict: dict, GSE_list:list) -> None:
    """
    Calculate the enrichment score for each sample in each dataset, and write the valid gene signature (gene signature used to calculate ssGSEA) into a txt file.
    """
    for GSE in GSE_list:
        # Read the expression file into dataframe, and discard the Probe_ID and Ensembl_Gene_ID column (if exist)
        ssdf = pd.read_csv('data/%s.csv' % GSE)
        ssdf = ssdf.drop(["Unnamed: 0", "Probe_ID", "Ensembl_Gene_ID"], axis=1,
                         errors='ignore')  # drop these columns only if exists (suppress error)
        # Check for genes in signature that are not in the dataset
        print('\nFor dataset %s:' % GSE)
        valid_signatures = check_gene_signature(ssdf, signatures_dict)
        # Get ssGSEA
        print('\nProcessing Dataset %s ------------------------------------------------------------' % GSE)
        ss = gp.ssgsea(data=ssdf,
                       gene_sets=signatures_dict,
                       outdir='scores/%s_ssgsea_report' % GSE,
                       scale=False,  # set scale to False to get real original ES
                       sample_norm_method='rank',  # choose 'custom' for your own rank list
                       permutation_num=0,  # skip permutation procedure, because you don't need it
                       min_size=1,  # Minimum allowed number of genes from gene set also the data set. Default: 15.
                       max_size=5000,  # Maximum allowed number of genes from gene set also the data set. Default: 2000.
                       no_plot=True,  # skip plotting, because you don't need these figures
                       processes=4, format='png', seed=9)
        # Write the gene signature used to calculate ssGSEA(some gene could be missing in the expression file)
        with open('scores/%s_ssgsea_report/%s_valid_gene_signature_list.txt' % (GSE, GSE), 'w') as f:
            for (key, value) in valid_signatures.items():
                f.write('%s:%s\n' % (key, value))


if __name__ == '__main__':
    GSE_list = ["GSE140261", "GSE115650", "GSE56787"]
    signatures_dict = read_gene_signature()
    get_ssGSEA(signatures_dict, GSE_list)
