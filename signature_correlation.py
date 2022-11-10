import matplotlib.pyplot as plt
import pandas as pd
import pingouin as pg

from preprocess import read_es, read_characteristic


class Correlation:
    """Calculate the Pearson's signature_correlation between patient characteristics and gene signature enrichment scores"""

    def __init__(self, GSE_list):
        self.GSE_list = GSE_list
        self.es_df = read_es(self.GSE_list)
        self.combined_df = self.match_patients(self.es_df)
        self.corr_df = self.get_correlation(self.combined_df)

    def match_patients(self, df):
        """
        Add CSS and repeat length (if exist) to the enrichment score dataframe by matching samples to patients.
        :param df: dataframe to match the patient data
        :return: enrichment score dataframe with CSS and/or repeat length
        """
        patient_df = read_characteristic(self.GSE_list)
        combined_df = pd.merge(df, patient_df, left_index=True, right_index=True)

        return combined_df.astype(float)

    def get_correlation(self, combined_df, method='spearman'):
        """
        Calculate Pearson's signature_correlation between patient characteristics and gene signatures enrichment scores
        :param combined_df: a dataframe to calculate the signature_correlation
        :return: corr_df dataframe containing the signature_correlation coefficients
        """
        tmp_df = combined_df.drop(columns='Age', errors='ignore')  # Exclude the age column
        corr_df = tmp_df.corr(method=method)  # signature_correlation between columns
        corr_df = corr_df.drop(index=['CSS', 'Repeat_length'], errors='ignore')  # remove non-useful rows
        columns_to_keep = [item for item in corr_df.columns.to_list() if item in ['CSS', 'Repeat_length']]
        corr_df = corr_df[columns_to_keep]
        print("\n%s correlation for %s:\n" % (method, self.GSE_list), corr_df)
        return corr_df

    @staticmethod
    def get_p_value(combined_df, characteristic='CSS', method='pearson'):
        cols_to_remove = ['Age', 'CSS', 'Repeat_length']
        x = combined_df.drop(columns=cols_to_remove, errors='ignore')
        y = combined_df[characteristic]
        corr_p_value_df = pd.DataFrame(columns=x.columns, index=['r', 'p-val'])
        for item in x:
            tmp = pg.corr(x[item], y, method=method)
            corr_p_value_df.loc['r', item] = tmp.loc[method, 'r']
            corr_p_value_df.loc['p-val', item] = tmp.loc[method, 'p-val']
        print("\n%s r and p-value for %s:\n" % (method, characteristic), corr_p_value_df)
        return corr_p_value_df.T

    def partial_correlation(self, df):
        """
        Calculate the partial signature_correlation with age as a covariant
        """
        signatures = ['D4Z4_interactome', 'DUX4_target', 'PAX7_target']
        for characteristic in self.corr_df.columns:
            list = []
            for sig in signatures:
                pc_df = pg.partial_corr(data=self.es_df, x=characteristic, y=sig, covar='Age')
                list.append(pc_df)
            pcorr_df = pd.concat(list)
            pcorr_df = pcorr_df.set_index(pd.Index(signatures))
            print("Partial %s signature_correlation for %s:\n" % (characteristic, self.GSE), pcorr_df)

    def plot_correlation(self, df, corr_df):
        """
        Plot the signature_correlation between patient characteristics and gene signatures enrichment scores
        :param df: dataframe contain enrichment score or gene expression
        :param corr_df: signature_correlation dataframe
        """
        genes = [item for item in df.columns if item not in ['CSS', 'Repeat_length', 'Age']]
        for characteristic in corr_df.columns:
            if len(genes) == 3:
                fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            elif len(genes) == 4:
                fig, ax = plt.subplots(2, 2, figsize=(10, 10))
            else:
                fig, ax = plt.subplots(4, 3, figsize=(15, 20))
            # fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(13, 4))
            ax = ax.flatten()  # flatten the axis to iterate through 2 rows
            for j, signature in enumerate(genes):
                ax[j].scatter(df[signature], df[characteristic])
                ax[j].title.set_text('Correlation = ' + "{:.2f}".format(corr_df[characteristic][signature]))
                ax[j].set(xlabel=signature, ylabel=characteristic)
            fig.subplots_adjust(wspace=.3)
            plt.show()

    def write_correlation(self):
        """
        Wirte the corr_df to csv for record keeping
        :param corr_df: signature_correlation dataframe containing pearson's signature_correlation between patient characteristics and gene
                        signatures enrichment scores
        """
        self.corr_df.to_csv('signature_correlation/%s_correlation.csv' % self.GSE, encoding='utf-8')


if __name__ == '__main__':
    dataset_list = ["GSE15090", "GSE36398", "GSE56787", "GSE115650", "GSE140261"]
    gse0 = Correlation(dataset_list)

