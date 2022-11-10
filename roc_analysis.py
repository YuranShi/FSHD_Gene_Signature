import numpy as np
from matplotlib import pyplot as plt
from numpy import mean
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc, roc_curve, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import label_binarize

from preprocess import load_data, boruta_feature_selection, show_signature, find_common_genes, load_data_from_list, \
    variance_feature_selection, feature_scaling


def roc_analysis(feature, label, model, title):
    """
    Plot a ROC curve for binary classification using 20 times 5-fold cross validation
    :param feature: features
    :param label: labels
    :param model: a classification model
    :param title: figure title
    :return:
    """
    feature = feature.to_numpy()
    # Run classifier with cross-validation and plot ROC curves
    cv_splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=1)

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    fig, ax = plt.subplots()
    for i, (train, test) in enumerate(cv_splitter.split(feature, label)):
        model.fit(feature[train], label[train].ravel())
        # Get the fpr and tpr
        y_pred_proba = model.predict_proba(feature[test])[::, 1]
        fpr, tpr, _ = roc_curve(label[test], y_pred_proba)
        roc_auc = roc_auc_score(label[test], y_pred_proba)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(roc_auc)

        # Get the fpr and tpr (and plotting each fold)
        # viz = RocCurveDisplay.from_estimator(
        #     model,
        #     feature[test],
        #     label[test],
        #     name='_nolegend_',
        #     alpha=0.2,
        #     lw=1,
        #     ax=ax,
        # )
        # interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        # interp_tpr[0] = 0.0
        # tprs.append(interp_tpr)
        # aucs.append(viz.roc_auc)

    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )
    ax.set(
        xlim=[0, 1],
        ylim=[0, 1],
        title=title,
    )
    ax.legend(loc="lower right")
    plt.show()


def compare_roc(feature_list, label_list, model_list, title):
    """
    Plot a list of ROC curves and save the image after plotting each curve in /figures folder.
    :param feature_list: a list of features to perform the classification
    :param label_list: a list of labels to perform the classification
    :param model_list: a list of classification models
    :param title: title of the figure
    :return:
    """
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)
    ax.set(
        xlim=[0, 1],
        ylim=[0, 1],
        title=title,
    )
    line_labels = ['Combined signature', 'Refined signature', 'Exploratory signature']
    for feature, label, model, line_label in zip(feature_list, label_list, model_list, line_labels):
        feature = feature.to_numpy()
        label = label_binarize(label, classes=['FSHD', 'Control'])
        # Run classifier with cross-validation and plot ROC curves
        cv_splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=1)

        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)

        for i, (train, test) in enumerate(cv_splitter.split(feature, label)):
            model.fit(feature[train], label[train].ravel())
            # Get the fpr and tpr
            y_pred_proba = model.predict_proba(feature[test])[::, 1]
            fpr, tpr, _ = roc_curve(label[test], y_pred_proba)
            roc_auc = roc_auc_score(label[test], y_pred_proba)
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(roc_auc)

            # Get the fpr and tpr (and plotting each fold)
            # viz = RocCurveDisplay.from_estimator(
            #     model,
            #     feature[test],
            #     label[test],
            #     name='_nolegend_',
            #     alpha=0.2,
            #     lw=1,
            #     ax=ax,
            # )
            # interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
            # interp_tpr[0] = 0.0
            # tprs.append(interp_tpr)
            # aucs.append(viz.roc_auc)

        # Plot the mean ROC curve
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        ax.plot(
            mean_fpr,
            mean_tpr,
            label=r"%s ROC-AUC=%0.2f $\pm$ %0.2f)" % (line_label, mean_auc, std_auc),
            lw=2,
            alpha=0.8,
        )
        # Plot standard deviation
        # std_tpr = np.std(tprs, axis=0)
        # tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        # tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        # ax.fill_between(
        #     mean_fpr,
        #     tprs_lower,
        #     tprs_upper,
        #     color=color,
        #     alpha=0.2
        # )
        ax.legend(loc="lower right")
        plt.savefig('figures/%s.png' % line_label)
        print('%s figure saved' % line_labels)


def compare_acc(feature_list, label_list, model_list):
    """
    Compare the accuracy
    :param feature_list: a list of features to perform the classification
    :param label_list: a list of labels to perform the classification
    :param model_list: a list of classification models
    :return:
    """
    line_labels = ['Combined signature', 'Refined signature', 'Exploratory signature']
    for feature, label, model, line_label in zip(feature_list, label_list, model_list, line_labels):
        feature = feature.to_numpy()
        label = label_binarize(label, classes=['FSHD', 'Control'])
        # Run classifier with cross-validation and plot ROC curves
        cv_splitter = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=1)
        cv_result = cross_validate(model, feature, label.flatten(),
                                   cv=cv_splitter,
                                   scoring='accuracy',
                                   n_jobs=-1)
        acc = mean(cv_result['test_score'])
        print('Accracy for %s is: ' % line_label, acc)


if __name__ == '__main__':
    # ------------------------ Load Data  ------------------------ #
    # Combined Signature
    GSE_list_0 = ["GSE140261", "GSE115650"]
    sig_list = ['DUX4_target', 'PAX7_target']
    feature_0, label_0 = load_data(GSE_list_0, sig_list, signature_version='v0')
    # Refined Signature
    GSE_list_1 = ["GSE140261", "GSE115650", 'Heuvel_expression']
    feature_1, label_1 = load_data(GSE_list_1, sig_list, signature_version='v0')
    # Exploratory Signature
    common_gene_list = find_common_genes(GSE_list_0)
    feature_2, label_2 = load_data_from_list(GSE_list_0, common_gene_list)

    # ---------------------- Feature Scaling ------------------------- #
    feature_2 = feature_scaling(feature_2)

    # ---------------------- Feature Selection ------------------------- #
    feature_1 = boruta_feature_selection(feature_1, label_1, 99)
    feature_2, label_2 = variance_feature_selection(feature_2, label_2, 30)

    # -------------------------- Model ------------------------------- #
    model_0 = RandomForestClassifier()
    model_1 = RandomForestClassifier()
    model_2 = GaussianNB()

    feature_list = [feature_0, feature_1, feature_2]
    label_list = [label_0, label_1, label_2]
    model_list = [model_0, model_1, model_2]
    # title = 'ROC Curve on Binary Classification (FSHD vs Control)'
    # compare_roc(feature_list, label_list, model_list, title)
    compare_acc(feature_list, label_list, model_list)

    # -------------------- Print Selected Genes ----------------------- #
    # print(show_signature(feature_selected_3.columns))
    # print(*feature_selected_3.columns, sep='\n')
    # show_signature(feature_selected_3.columns).to_csv('tmp.csv')
