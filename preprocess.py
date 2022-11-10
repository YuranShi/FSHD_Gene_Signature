import numpy as np
import pandas as pd
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, PowerTransformer, QuantileTransformer, RobustScaler


# ------------------------------ Feature Scaling and Selection ------------------------------ #

def feature_scaling(feature_df, scaler='PowerTransformer'):
    """
    Scale the gene expression in range (0, 1)
    :return: scaled feature
    """
    if scaler == 'MinMaxScaler':
        scaler = MinMaxScaler(feature_range=(0, 1))
    elif scaler == 'PowerTransformer':
        scaler = PowerTransformer()
    elif scaler == 'QuantileTransformer':
        scaler = QuantileTransformer()
    elif scaler == 'RobustScaler':
        scaler = RobustScaler()
    scaled_feature = scaler.fit_transform(feature_df)
    scaled_feature = pd.DataFrame(scaled_feature, columns=feature_df.columns, index=feature_df.index)
    return scaled_feature


def variance_feature_selection(feature, label, k):
    """
    Feature selection is performed using ANOVA F measure via the f_classif() function. Select K best features.
    :param feature: original feature
    :param label: original label
    :param k: number of features to select
    :return:
    """
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(feature, label)
    cols = selector.get_support(indices=True)
    features_df_new = feature.iloc[:, cols]
    return features_df_new, label


def boruta_feature_selection(feature, label, perc):
    """
    Use random forest based boruta feature selection to select important features
    :param feature: original feature
    :param label: original label
    :param perc: percentile to pick threshold for comparison between shadow and real features (0-100, 100 most stringent)
    :return:
    """
    # split data
    X_train, X_test, y_train, y_test = train_test_split(feature, label, test_size=.2, random_state=1)
    rfc = RandomForestClassifier()
    boruta_selector = BorutaPy(rfc, n_estimators='auto', alpha=0.10, perc=perc, random_state=42)
    boruta_selector.fit(np.array(X_train), np.array(y_train))
    selected_rf_features = pd.DataFrame({'Feature': list(X_train.columns), 'Ranking': boruta_selector.ranking_})
    important_features_list = selected_rf_features[selected_rf_features['Ranking'] == 1]['Feature'].to_list()
    important_features = feature[[gene for gene in feature.columns if gene in important_features_list]]
    return important_features


# ------------------------------ Read and Load Data ------------------------------ #

def load_data(GSE_list: list, signatures: list, signature_version='v0') -> (pd.DataFrame, pd.DataFrame):
    """
    Load the features and labels as dataframes for classifiers
    :param signature_version: version for the gene signature
    :param GSE_list: a list of datasets for the classification tasks
    :param signatures: a list of gene signatures to serve as features. Options: 'DUX4_target', 'D4Z4_interactome', 'PAX7_target'
    :return: features and label dataframes
    """
    sig_dict = read_gene_signature(signature_version)
    gene_list = []
    for group in signatures:
        gene_list.extend(sig_dict.get(group))
    # search common genes (intersection) in the expression files as features
    valid_gene_list = []
    for GSE in GSE_list:
        valid_genes = search_signature(GSE=GSE, signature_set=gene_list)
        valid_gene_list.append(valid_genes)
    common_valid_genes = list(set.intersection(*map(set, valid_gene_list)))
    feature = read_expression(GSE_list, common_valid_genes)
    label = read_meta(GSE_list)
    feature = feature_scaling(feature, scaler='PowerTransformer')
    return feature, label


def load_data_from_list(GSE_list: list, gene_list) -> (pd.DataFrame, pd.DataFrame):
    """
    Load the features and labels as dataframes for classifiers based on input gene list
    :param GSE_list: a list of datasets for the classification tasks
    :param gene_list: a list of genes to read as features
    :return: features and label dataframes
    """
    feature = read_expression(GSE_list, gene_list)
    feature = feature.dropna(axis=1)
    label = read_meta(GSE_list)
    return feature, label


def read_expression(GSE_list: list, gene_list: list) -> pd.DataFrame:
    """
    Read normalized expression profiles based on gene list input
    :return: concatenated expression dataframe
    """
    df_list = []
    for GSE in GSE_list:
        expression_df = pd.read_csv('data/%s.csv' % GSE, usecols=lambda x: x not in ["Probe_ID", "Ensembl_Gene_ID"])
        expression_df = expression_df.T  # Transpose dataframe to make sample as rows
        expression_df = expression_df.rename(columns=expression_df.iloc[0]).drop(
            expression_df.index[0])  # remove header
        expression_df = expression_df.loc[:, ~expression_df.columns.duplicated()]  # remove duplicated columns
        expression_df = expression_df.filter(items=gene_list)
        df_list.append(expression_df)
    return pd.concat(df_list)


def read_meta(GSE_list: str) -> pd.DataFrame:
    """
    Read metadata for list of datasets
    :return: concatenated dataframe for metadata
    """
    df_list = []
    for GSE in GSE_list:
        meta_df = pd.read_csv('meta/%s_meta.csv' % GSE, index_col=0, usecols=[0, 1])
        meta_df.sampletype = ['FSHD' if ('FSHD' in description) else 'Control' for description in meta_df.sampletype]
        df_list.append(meta_df)
    return pd.concat(df_list)


def read_es(GSE_list: str) -> pd.DataFrame:
    """
    Read normalized enrichment score into dataframe
    :return: dataframe that contain the normalized enrichment score
    """
    df_list = []
    for GSE in GSE_list:
        normalized_es_path = 'scores/%s_ssgsea_report/gseapy.samples.normalized.es.txt' % GSE
        es_df = pd.read_csv(normalized_es_path, sep="\t", skiprows=2, index_col=0)
        es_df = es_df.T  # transpose the dataframe to make samples as rows
        df_list.append(es_df)
    return pd.concat(df_list)


def read_characteristic(GSE_list) -> pd.DataFrame:
    """
    Read patient characteristics files into dataframe
    :param GSE_list: a list of datasets to read
    :return: a dataframe that contains the characteristics of input datasets
    """
    df_list = []
    for GSE in GSE_list:
        patient_df = pd.read_csv('patient_characteristics/%s_patient.csv' % GSE, index_col=0)
        df_list.append(patient_df)
    return pd.concat(df_list)


def read_gene_signature(version='v0') -> dict:
    """
    Read the gene sets as dictionary
    :param version: gene signature version
    :return: signatures_dict containing keys as the name of a gene set, values as list of genes
    """
    signatures_pd = pd.read_csv('gene_signatures/gene_signatures_%s.csv' % version, index_col=0, header=None)
    keys = signatures_pd.index.tolist()
    values = signatures_pd.values.tolist()
    values = [list(filter(lambda x: x == x, inner_list)) for inner_list in values]  # filter all nan value
    signatures_dict = dict(zip(keys, values))
    return signatures_dict


def read_signature_as_list(version='v0') -> list:
    """
    Read the gene sets as a list
    :param version:
    :return:
    """
    sig_dict = read_gene_signature(version)
    sig_list = []
    for key in sig_dict:
        value = sig_dict[key]
        sig_list = sig_list + value
    return sig_list


def search_signature(GSE: str, signature_set: list) -> list:
    """
    Search for genes in an expression file given a set of gene signatures
    :param GSE: GSE accession of the dataset
    :param signature_set:a list of genes
    :return: a list of genes in the expression file that is also in the signature set
    """
    ssdf = pd.read_csv('data/%s.csv' % GSE)
    ssdf = ssdf.drop(["Unnamed: 0", "Probe_ID", "Ensembl_Gene_ID"], axis=1,
                     errors='ignore')  # drop these columns if exists
    genes = ssdf['Gene_Symbol'].to_list()
    valid_genes = [item for item in signature_set if item in genes]
    return valid_genes


def find_common_genes(datasets) -> list:
    """
    Find the common genes between datasets, also remove genes that don't have an official symbol
    :param datasets:
    :return:
    """
    gene_list = []
    for GSE in datasets:
        expression_df = pd.read_csv('data/%s.csv' % GSE)
        genes = filter(lambda val: 'ENSG' not in val,
                       expression_df['Gene_Symbol'].to_list())  # filter out genes begin with ENSG
        gene_list.append(genes)
    common_valid_genes = list(set.intersection(*map(set, gene_list)))
    return common_valid_genes


def show_signature(gene_list):
    """
    Given a list of genes, return a dataframe that show which signature group the gene is from
    :param gene_list:
    :return:
    """
    sig = read_gene_signature()
    sig_list = []
    for gene in gene_list:
        for key in sig:
            if gene in sig.get(key):
                sig_list.append(key)
                break
    df = pd.DataFrame(list(zip(gene_list, sig_list)),
                      columns=['Genes', 'Signatures'])
    df.set_index('Genes')
    return df


# Testing
if __name__ == '__main__':
    GSE_list = ["GSE140261", "GSE115650", "GSE56787"]
    feature, label = load_data(GSE_list, ['DUX4_target', 'D4Z4_interactome'])
    print(feature.head())
    print(label.head())
