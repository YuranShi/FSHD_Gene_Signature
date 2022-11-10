import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from preprocess import read_gene_signature, read_expression, read_meta


def get_data(GSE_list: list, sig_list: list) -> (pd.DataFrame, pd.DataFrame):
    """
    Get the expression data and metadata based on input gene signatures for given datasets.
    :param GSE_list: a list of studies
    :param sig_list: a list of gene signatures
    :return: expression and metadata as dataframes
    """
    sig_dict = read_gene_signature()
    gene_list = []
    for group in sig_list:
        gene_list.extend(sig_dict.get(group))
    expression_df = read_expression(GSE_list, gene_list)
    expression_df = expression_df.T.astype('float')
    meta_df = read_meta(GSE_list)
    return expression_df, meta_df


def get_distribution_df(GSE_list: list, sig_list: list) -> pd.DataFrame:
    """
    Get the histogram distribution of the expression data.
    :return: a histogram dataframe contains the distribution
    """
    expression_df, meta_df = get_data(GSE_list, sig_list)
    # binning
    # max_expression = expression_df.max().max()  # get the max value in all expressions
    # log_bins = np.logspace(0, np.log(max_expression), 30)  # get a logarithmic binning
    log_bins = np.logspace(0, 6, 30)
    log_bins[0] = 0  # make the first bin from 0
    hist_dic = {}
    for sample in expression_df.columns:
        expression_list = expression_df.loc[:, sample].to_list()
        hist, _ = np.histogram(expression_list, bins=log_bins)
        hist_dic[sample] = hist
    hist_df = pd.DataFrame.from_dict(hist_dic)
    hist_df = hist_df.set_index(log_bins[1:])
    # reorder hist_df by putting all control samples before FSHD samples
    control = meta_df[meta_df['sampletype'] == 'Control'].index.to_list()
    fshd = meta_df[meta_df['sampletype'] == 'FSHD'].index.to_list()
    new_order = control + fshd
    meta_df = meta_df.reindex(new_order)
    hist_df = hist_df[new_order]
    hist_df.index = np.round(hist_df.index, 1)
    hist_df.columns = [hist_df.columns.to_list(), meta_df.values.flatten().tolist()]
    return hist_df


def plot_heatmap(df: pd.DataFrame, sig_list: list) -> pd.DataFrame:
    """
    Plot a heatmap for the given dataframe
    """
    plt.subplots(figsize=(12, 9))
    hm = sns.heatmap(df, cmap='coolwarm', cbar_kws={'label': 'Number of Genes'})
    hm.set(title='Expression Distribution Heatmap for %s' % sig_list)
    hm.set(xlabel='Samples', ylabel='Normalized Expression Value')
    plt.show()


def write_distribution(hist_df_list: list, runs):
    """
    Write distribution dataframe to an Excel file
    """
    pass


if __name__ == '__main__':
    GSE_list = ["GSE140261", "GSE115650"]
    # runs = [['DUX4_target', 'D4Z4_interactome', 'PAX7_target'], ['DUX4_target'], ['D4Z4_interactome'],
    #         ['PAX7_target']]  # 'DUX4_target', 'D4Z4_interactome', 'PAX7_target'
    runs = [['DUX4_target']]
    hist_df_list = []
    for sig_list in runs:
        hist_df = get_distribution_df(GSE_list, sig_list)
        plot_heatmap(hist_df, sig_list)
        hist_df_list.append(hist_df)
