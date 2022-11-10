import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import mean, std
from pandas import DataFrame
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import cross_validate, RepeatedStratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import label_binarize
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from preprocess import feature_scaling, load_data, variance_feature_selection, find_common_genes, load_data_from_list, read_signature_as_list


def plot_experiments(run_list, legend):
    """
    Plot a bar graph to compare the performance on the classification task
    :param run_list: A list of Classification experiments.
    :param legend: A list of strings to serve as the legend.
    """
    fig, ax = plt.subplots()
    x_label = ['LR', 'SVM', 'NB', 'LDA', 'DT', 'RF', 'MLP', 'KNN']
    x_axis = np.arange(len(x_label))
    plt.ylabel('ROC_AUC')
    plt.ylim(0.2, 1)
    num_bars = len(run_list)
    bar_width = 0.3 - (num_bars - 3) * 0.1  # set bar width (2 bars=0.4, 3 bars=0.3, 4 bars=0.2)
    # colors = [plt.get_cmap('Set2')(1. * i / len(run_list)) for i in range(len(run_list))]
    colors = [plt.get_cmap('Set2')(0.1 + 0.1 * i) for i in range(num_bars)]
    for i, run in enumerate(run_list):
        ax.bar(x=x_axis + bar_width * i,  # bar location (change with bar width)
               width=bar_width,
               height=[run.test_scores[model]['average'] for model in run.test_scores],
               yerr=[run.test_scores[model]['std'] for model in run.test_scores],
               color=colors[i])
    plt.xticks(x_axis, x_label)
    plt.legend(legend)
    plt.title('ROC-AUC on Binary Classification')
    plt.show()


class Classification:
    """Classification experiments based various classification models"""

    def __init__(self, score_func, features: DataFrame, target_labels: DataFrame):
        """
        :param score_func: function for scoring and calculate error
        :param features: raw features
        :param target_labels: raw target
        """
        self.score = make_scorer(score_func, greater_is_better=True, needs_threshold=True)
        self.models = self.get_models()
        self.features = feature_scaling(features)
        self.target_labels = label_binarize(target_labels, classes=['FSHD', 'Control'])
        self.test_scores = {}
        self.roc_param = {}

    @staticmethod
    def get_models():
        """
        Initialize linear and non-linear models for classification
        :return: list of models
        """
        # linear models
        LR = LogisticRegression(max_iter=1000)
        SVM = SVC(max_iter=20000)
        NB = GaussianNB()
        LDA = LinearDiscriminantAnalysis()
        # non-linear models
        DT = DecisionTreeClassifier()
        RF = RandomForestClassifier()
        MLP = MLPClassifier()
        KNN = KNeighborsClassifier()
        return [LR, SVM, NB, LDA, DT, RF, MLP, KNN]

    def train_and_eval(self):
        """
        Train and evaluate each model
        Average and standard deviation are recorded
        """
        cv_splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=42)
        for model in self.models:
            temp = []
            cv_result = cross_validate(model, self.features, self.target_labels.flatten(),
                                       cv=cv_splitter,
                                       scoring=self.score,
                                       n_jobs=-1)
            temp.extend(cv_result['test_score'])
            self.test_scores[model] = {'average': mean(temp), 'std': std(temp)}

    def plot(self):
        """
         Plot the overall results
        """
        fig, ax = plt.subplots()
        plt.ylabel('ROC_AUC')
        plt.ylim(0.2, 1)
        x_label = ['LR', 'SVM', 'NB', 'LDA', 'DT', 'RF', 'MLP', 'KNN']
        ax.bar(x=x_label,  # x=[str(model) for model in self.models],
               height=[self.test_scores[model]['average'] for model in self.test_scores],
               yerr=[self.test_scores[model]['std'] for model in self.test_scores])
        plt.show()

    def print_best_model(self):
        """
        Print The model with max average
        """
        best = max(self.test_scores, key=lambda model: self.test_scores[model]['average'])
        print('The best performing model is %s' % best)


if __name__ == '__main__':
    GSE_list = ["GSE140261", "GSE115650"]  # "GSE140261", "GSE115650", 'Heuvel_expression'
    # sig_list = ['DUX4_target', 'D4Z4_interactome', 'PAX7_target']  # 'DUX4_target', 'D4Z4_interactome', 'PAX7_target'

    common_gene_list = find_common_genes(GSE_list)

    # Setting element of comparison between runs
    legend = ['No feature selection', 'SelectKBest (k=30)']
    num_runs = len(legend)
    run_list = [None] * num_runs

    # ------------------------ Load Data & Create Classification Object ------------------------ #
    for i, item in enumerate(legend):
        feature, label = load_data_from_list(GSE_list, common_gene_list)
        if i == 1:
            feature, label = variance_feature_selection(feature, label, 30)
        run_list[i] = Classification(score_func=roc_auc_score,
                                     features=feature,
                                     target_labels=label)
    print('Finished Loading... \nStarting to Train and Evaluate...')

    # ---------------------------------- Train and Evaluate ----------------------------------- #
    scores = []
    for run in run_list:
        run.train_and_eval()
        run.print_best_model()
        scores.append(pd.DataFrame.from_dict(run.test_scores).T)
        print(scores[-1])
    plot_experiments(run_list, legend)
