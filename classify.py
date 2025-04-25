"""
class.py
----
Provides classification metrics vectors using classification scores.

Scores are interpreted as: higher = more likely to be 1.
"""
import numpy as np
from scipy.integrate import trapezoid

class Classifier(object):
    """ Classifier Object """
    # Base Classification Metrics
    @property
    def tpr(self):
        return self.TP_Count / (self.TP_Count + self.FN_Count)
    @property
    def fnr(self):
        return 1 - self.tpr
    @property
    def fpr(self):
        return self.FP_Count / (self.FP_Count + self.TN_Count)
    @property
    def tnr(self):
        return 1 - self.fpr
    # Derived Classification Metrics
    @property
    def recall(self):
        return self.tpr
    @property 
    def precision(self):
        return self.TP_Count / (self.TP_Count + self.FP_Count)
    @property
    def f1(self):
        return 2 * self.precision * self.recall / (self.precision + self.recall)
    @property
    def sensitivity(self):
        return self.tpr
    @property
    def specificity(self):
        return self.tnr
    @property
    def positive_predictive_value(self):
        return self.TP_Count / (self.TP_Count + self.FP_Count)
    @property
    def false_omission_rate(self):
        return self.FN_Count / (self.FN_Count + self.TN_Count)
    @property
    def negative_predictive_value(self):
        return 1 - self.false_omission_rate
    @property
    def false_dioscovery_rate(self):
        return 1 - self.positive_predictive_value
    @property
    def informedness(self):
        return self.tpr + self.tnr - 1
    @property
    def prevalence_threshold(self):
        return (np.sqrt(self.tpr * self.fpr) - self.fpr) / (self.tpr - self.fpr)
    @property
    def markedness(self):
        return self.positive_predictive_value + self.negative_predictive_value - 1
    # descriptive statistics
    @property
    def prevalence(self):
        return self.A / self.N
    @property
    def auroc(self):
        roc = (
            np.hstack((0, self.tpr[::-1], 1)), # true  positive rate (y)
            np.hstack((0, self.fpr[::-1], 1)), # false positive rate (x)
            )
        return trapezoid(*roc)
    @property
    def auprc(self):
        prc = (
            np.hstack((1, self.precision[::-1], 0)),
            np.hstack((0, self.recall[::-1], 1)),
            )
        return trapezoid(*prc)

    def setup(self, score, actual):
        # fill in numerical error scores
        score[np.isposinf(score)] = np.finfo(np.float64).max
        score[np.isneginf(score)] = np.finfo(np.float64).min
        score[np.isnan(score)]    = np.finfo(np.float64).max
        # parse args
        self.actual = actual
        self.score  = score
        # description info
        self.N = self.score.shape[0] # number of observations
        self.A = self.actual.sum()   # number of TRUE's
        # sort scores in descending order
        self.order    = np.argsort(score)
        self.sorted_actual = self.actual[self.order]
        self.sorted_score  = self.score[self.order]
        self.FP_Count = (1 - self.sorted_actual)[::-1].cumsum()[::-1]
        self.TP_Count = self.sorted_actual[::-1].cumsum()[::-1]
        self.FN_Count = self.sorted_actual.cumsum()
        self.TN_Count = (1 - self.sorted_actual).cumsum()
        return

    def __init__(self, score, actual):
        """
        score  : np.array(1d) - classification scores; higher = more likely 1
        actual : np.array(1d, bool/int) - outcome/labels
        """
        self.setup(score, actual)
        return

if __name__ == '__main__':
    pass

# EOF