import numpy as np
from sklearn.externals import joblib  # Save parameters
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold

from utils import *

# Constant
K = 5


def Train(model, X_train, y_train):
    # Initialize
    oob = 0  # Out-of-bag scores
    fprs, tprs, scores = [], [], []  # ROC curve
    feature_importance = pd.DataFrame(np.zeros((X_train.shape[1], K)),
                                      columns=["Fold_{}".format(i) for i in range(1, K + 1)])
    skf = StratifiedKFold(n_splits=K, random_state=K, shuffle=True)
    # Training
    for fold, (trn_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        print("Fold {}".format(fold))
        # Fitting model
        model.fit(X_train[trn_idx], y_train[trn_idx])
        # Computing train AUC score
        trn_fpr, trn_tpr, trn_thresholds = roc_curve(y_train[trn_idx],
                                                     model.predict_proba(X_train[trn_idx])[:, 1])
        trn_auc_score = auc(trn_fpr, trn_tpr)
        # Computing validation AUC score
        val_fpr, val_tpr, val_thresholds = roc_curve(y_train[val_idx],
                                                     model.predict_proba(X_train[val_idx])[:, 1])
        val_auc_score = auc(val_fpr, val_tpr)
        # Append in list
        scores.append((trn_auc_score, val_auc_score))
        fprs.append(val_fpr)
        tprs.append(val_tpr)
        # Export Importance
        feature_importance.iloc[:, fold - 1] = model.feature_importances_
        # Out of bag score
        oob += model.oob_score_ / K
        print("Fold {} OOB Score: {}".format(fold, model.oob_score_))
        print("Average OOB Score: {}".format(oob))
    # Save model
    joblib.dump(model, "./models/checkpoint.pkl")
    return fprs, tprs, scores, feature_importance
