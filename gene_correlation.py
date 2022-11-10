import pandas as pd

from signature_correlation import Correlation
from preprocess import read_gene_signature, read_expression, find_common_genes


class GeneCorrelation(Correlation):
    """Calculate the Pearson's signature_correlation between patient characteristics and gene expression level"""

    def __init__(self, GSE_list, genes):
        self.GSE_list = GSE_list
        self.genes = genes
        self.expression_df = read_expression(GSE_list, genes)
        self.combined_df = super().match_patients(self.expression_df)
        self.corr_df = super().get_correlation(self.combined_df, method='spearman')
        self.p_val_df = super().get_p_value(self.combined_df, method='spearman')
        # super().plot_correlation(self.expression_df, self.corr_df)


def get_p_signature(dataset_list):
    """
    Select a subset of gene signatures based p_value cutoffs from all signatures (gene_signatures_v0.csv)
    :param dataset_list: a list of datasets
    :return:
    """
    sig_dict = read_gene_signature()
    keys = list(sig_dict.keys())  # ['DUX4_target', 'D4Z4_interactome', 'PAX7_target']
    corr_object_list = []
    p_val_list = []
    for key in keys:
        corr = GeneCorrelation(dataset_list, sig_dict[key])
        corr_object_list.append(corr)
        p_val_list.append(corr.p_val_df.sort_values(by='p-val'))

    """ Select a subset of gene signatures based p_value cutoffs"""
    sig_versions = ['_v10', '_v11', '_v12']
    cutoff_list = [0.2, 0.1, 0.05]  # Define cutoff points
    for p_cutoff, version in zip(cutoff_list, sig_versions):
        CSS_selected_sig = {}
        num_genes = 0
        for key, p_df in zip(keys, p_val_list):
            genes = p_df[p_df['p-val'] < p_cutoff].index.to_list()
            CSS_selected_sig[key] = genes
            num_genes += len(genes)
        print('Number of genes selected with p_cutoff %.1f: %d' % (p_cutoff, num_genes))
        df = pd.DataFrame.from_dict(CSS_selected_sig, orient='index')
        df.to_csv('gene_signatures/gene_signatures%s.csv' % version, header=False)


def get_DGE_signature(dataset_list):
    """
    Use 500 most DE genes as the pool for selecting the gene signature
    :param dataset_list:
    :return:
    """
    gene_list = []
    for GSE in dataset_list:
        df = pd.read_csv('%s_500DEG.csv' % GSE, header=None)
        result = filter(lambda val: 'ENSG' not in val, df[0].to_list())  # filter out genes begin with ENSG
        gene_list.append(list(result))
    common_valid_genes = list(set.intersection(*map(set, gene_list)))
    corr = GeneCorrelation(dataset_list, common_valid_genes)
    p_val = corr.p_val_df.sort_values(by='p-val')

    # Write new signatures
    sig_versions = ['_v13', '_v14']
    cutoff_list = [0.2, 0.5]
    for p_cutoff, version in zip(cutoff_list, sig_versions):
        genes = p_val[p_val['p-val'] < p_cutoff].index.to_list()
        df = pd.DataFrame.from_dict(genes)
        df.to_csv('gene_signatures/gene_signatures%s.csv' % version, header=['Genes'], index=False)


def get_correlated_signatures(dataset_list):
    common_genes = find_common_genes(dataset_list)
    corr = GeneCorrelation(dataset_list, common_genes)
    p_val = corr.p_val_df.sort_values(by='p-val')

    # Write new signatures
    sig_versions = ['_v15']
    cutoff_list = [0.01]
    for p_cutoff, version in zip(cutoff_list, sig_versions):
        genes = p_val[p_val['p-val'] < p_cutoff].index.to_list()
        df = pd.DataFrame.from_dict(genes)
        df.to_csv('gene_signatures/gene_signatures%s.csv' % version, header=['Genes'], index=False)
        print('Number of genes selected with p_cutoff %.3f: %d' % (p_cutoff, len(genes)))


if __name__ == '__main__':
    dataset_list = ["GSE115650", "GSE140261"]

    """ Select a subset of gene signatures based p_value cutoffs"""
    get_p_signature(dataset_list)

    """ Use most differentially expressed genes instead of proposed gene signatures """
    get_DGE_signature(dataset_list)

    """ Get Correlation based on all genes """
