from typing import Type

import scipy.optimize as spopt

from estimate_bcm.confusion_matrix import NumericalConfusionMatrix
from estimate_bcm.metrics import Metric

class EstimationResult:

    def __init__(self,
                 metric_classes_set: set[Type[Metric]],
                 cm: NumericalConfusionMatrix,
                 orig_opt_result: spopt.OptimizeResult):

        self.metric_classes_set = metric_classes_set
        self.cm = cm
        self.orig_opt_result = orig_opt_result

        self.metric_values = self.compute_metric_values()


    def compute_metric_values(self):

        metric_values = {}
        for metric_class in self.metric_classes_set:
            metric_values[metric_class.symb_name] = metric_class.get_expr(self.cm)

        return metric_values


    def add_metric_class(self, metric_class: Type[Metric]):
        if metric_class not in self.metric_classes_set:
            self.metric_classes_set.add(metric_class)
            self.metric_values[metric_class.symb_name] = metric_class.get_expr(self.cm)

    def remove_metric_class(self, metric_class: Type[Metric]):
        if metric_class in self.metric_classes_set:
            del self.metric_values[metric_class.symb_name]
            self.metric_classes_set.remove(metric_class)



class EstimationResultCollection:

    def __init__(self):
        self.metric_classes_set = set()
        self.results_list = []

    def add_result(self, result: EstimationResult):
        self.results_list.append(result)

        for c in result.metric_classes_set:
            if c not in self.metric_classes_set:
                self.add_metric_class(c)

    def add_metric_class(self, metric_class: Type[Metric]):
        for r in self.results_list:
            r.add_metric_class(metric_class)

    def remove_metric_class(self, metric_class: Type[Metric]):
        for r in self.results_list:
            r.remove_metric_class(metric_class)
