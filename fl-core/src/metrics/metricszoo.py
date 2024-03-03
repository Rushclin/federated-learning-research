import torch
import numpy as np
import warnings

from sklearn.metrics import  roc_curve, f1_score, precision_score, recall_score, accuracy_score

from .basemetric import BaseMetric

warnings.filterwarnings('ignore')


class Accuracy(BaseMetric):
    def __init__(self):
        self.scores = []
        self.answers = []
        self._use_youdenj = False

    def collect(self, pred, true):
        p, t = pred.detach().cpu(), true.detach().cpu()
        self.scores.append(p)
        self.answers.append(t)

    def summarize(self):
        scores = torch.cat(self.scores)
        answers = torch.cat(self.answers).numpy()

        if scores.size(-1) > 1:
            labels = scores.argmax(-1).numpy()
        else: 
            scores = scores.sigmoid().numpy()
            if self._use_youdenj:
                fpr, tpr, thresholds = roc_curve(answers, scores)
                cutoff = thresholds[np.argmax(tpr - fpr)]
            else:
                cutoff = 0.5
            labels = np.where(scores >= cutoff, 1, 0)
        return accuracy_score(answers, labels)
    
class F1(BaseMetric):
    def __init__(self):
        self.scores = []
        self.answers = []
        self._use_youdenj = False

    def collect(self, pred, true):
        p, t = pred.detach().cpu(), true.detach().cpu()
        self.scores.append(p)
        self.answers.append(t)

    def summarize(self):
        scores = torch.cat(self.scores)
        answers = torch.cat(self.answers).numpy()

        if scores.size(-1) > 1:
            labels = scores.argmax(-1).numpy()
        else: 
            scores = scores.sigmoid().numpy()
            if self._use_youdenj: 
                fpr, tpr, thresholds = roc_curve(answers, scores)
                cutoff = thresholds[np.argmax(tpr - fpr)]
            else:
                cutoff = 0.5
            labels = np.where(scores >= cutoff, 1, 0)
        return f1_score(answers, labels, average='weighted', zero_division=0)
    

class Precision(BaseMetric):
    def __init__(self):
        self.scores = []
        self.answers = []
        self._use_youdenj = False

    def collect(self, pred, true):
        p, t = pred.detach().cpu(), true.detach().cpu()
        self.scores.append(p)
        self.answers.append(t)

    def summarize(self):
        scores = torch.cat(self.scores)
        answers = torch.cat(self.answers).numpy()

        if scores.size(-1) > 1: 
            labels = scores.argmax(-1).numpy()
        else: 
            scores = scores.sigmoid().numpy()
            if self._use_youdenj: 
                fpr, tpr, thresholds = roc_curve(answers, scores)
                cutoff = thresholds[np.argmax(tpr - fpr)]
            else:
                cutoff = 0.5
            labels = np.where(scores >= cutoff, 1, 0)
        return precision_score(answers, labels, average='weighted', zero_division=0)

class Recall(BaseMetric):
    def __init__(self):
        self.scores = []
        self.answers = []
        self._use_youdenj = False

    def collect(self, pred, true):
        p, t = pred.detach().cpu(), true.detach().cpu()
        self.scores.append(p)
        self.answers.append(t)

    def summarize(self):
        scores = torch.cat(self.scores)
        answers = torch.cat(self.answers).numpy()

        if scores.size(-1) > 1:
            labels = scores.argmax(-1).numpy()
        else: 
            scores = scores.sigmoid().numpy()
            if self._use_youdenj:
                fpr, tpr, thresholds = roc_curve(answers, scores)
                cutoff = thresholds[np.argmax(tpr - fpr)]
            else:
                cutoff = 0.5
            labels = np.where(scores >= cutoff, 1, 0)
        return recall_score(answers, labels, average='weighted', zero_division=0)

