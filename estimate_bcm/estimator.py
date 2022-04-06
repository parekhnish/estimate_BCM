from copy import deepcopy
import warnings
import itertools

import numpy as np
import scipy.optimize as spopt
from sympy import Matrix as sym_matrix
from sympy import zeros as sym_zeros
from estimate_bcm.estimation_result import EstimationResult, EstimationResultCollection

from estimate_bcm.metrics import Metric, LinearMetric
from estimate_bcm.confusion_matrix import NumericalConfusionMatrix, SymbolicConfusionMatrix


class MetricKeyExists(Exception):
    pass


class Estimator:

    def __init__(self):

        self.all_metrics_dict = {}
        self.linear_metrics_dict = {}
        self.metric_classes_set = set()


    def add_metric(self, metric_obj: Metric, force=False):

        metric_class = metric_obj.__class__
        self.metric_classes_set.add(metric_class)

        if metric_class in self.metric_classes_set:
            if not force:
                raise MetricKeyExists(f"A {metric_class} metric already exists in the estimator! If you want to overwrite it, call `add_metric()` with the argument `force=True`")


        metric_name = metric_obj.symb_name
        self.all_metrics_dict[metric_name] = metric_obj
        if isinstance(metric_obj, LinearMetric):
            self.linear_metrics_dict[metric_name] = metric_obj


    def estimate(self, num_iter_per_metric: int = 11):

        # --- CHECK ---: At least 4 linear metrics are available
        num_linear_metrics = len(self.linear_metrics_dict)
        if num_linear_metrics < 4:
            raise NotImplementedError(f"Found only {num_linear_metrics} linear metrics; Currently code handles only situations with 4 linear metrics or more!")

        # Copy the linear metric information into a list,
        # so that there is no ambiguity of ordering in subsquent code
        linear_metric_obj_list = list(self.linear_metrics_dict.values())

        # Create the symbolic LHS and RHS matrices
        dummy_symbolic_cm = SymbolicConfusionMatrix()
        symbolic_lhs_matrix, symbolic_rhs_vector = create_symbolic_linear_eq_matrices(linear_metric_obj_list, dummy_symbolic_cm)

        # --- CHECK ---: Full rank in the symbolic LHS matrix
        symbolic_lhs_rank = symbolic_lhs_matrix.rank()
        if symbolic_lhs_rank < 4:
            raise NotImplementedError(f"Symbolic rank of LHS matrix is {symbolic_lhs_rank}; Only (rank=4) is handled at the moment!")

        # --- CHECK ---: Homogenous system
        if symbolic_rhs_vector == sym_zeros(4, 1):
            raise NotImplementedError(f"Encountered homogenous system of linear equations; This is not handled at the moment!")


        # Compute the uncertainty values for each linear metric
        uncertainty_values_list = []
        for metric_obj in linear_metric_obj_list:
            if metric_obj.perturb_during_estimation:
                uncertainty_values_list.append(np.linspace(metric_obj.lower_lim, metric_obj.upper_lim, num_iter_per_metric))
            else:
                uncertainty_values_list.append(np.linspace(metric_obj.current_value, metric_obj.current_value, 1))


        # Copy the linear metric obj list
        # (This will be used for the uncertainty iteration
        #  , thus leaving the supplied metric objects intact and unchanged)
        curr_linear_metric_obj_list = deepcopy(linear_metric_obj_list)

        # Init the numeric LHS and RHS matrices
        numeric_lhs_matrix = np.empty((num_linear_metrics, 4))
        numeric_rhs_vector = np.empty((num_linear_metrics,))

        # Init the EstimationResultsCollection result
        er_coll = EstimationResultCollection()


        # Loop over every combination of the uncertain values ...
        for curr_metric_value_tuple in itertools.product(*uncertainty_values_list):

            # Populate the matrices with the required equations and values
            for metric_idx, (metric_obj, metric_value) in enumerate(zip(curr_linear_metric_obj_list, curr_metric_value_tuple)):
                metric_obj.current_value = metric_value
                lhs_row, rhs_val = metric_obj.get_numeric_eqn()

                numeric_lhs_matrix[metric_idx, :] = lhs_row
                numeric_rhs_vector[metric_idx] = rhs_val

            # --- CHECK ---: Full rank for numerical matrix
            numeric_lhs_rank = np.linalg.matrix_rank(numeric_lhs_matrix)
            if numeric_lhs_rank < 4:
                warnings.warn(f"Encountered LHS matrix with rank {numeric_lhs_rank}; Only (rank=4) is handled at the moment! Current iteration skipped...")

            # --- CHECK ---: Homogenous system
            rhs_is_zero = np.all(numeric_rhs_vector == 0)
            if rhs_is_zero:
                warnings.warn(f"Encountered homogenous system of linear equations; This is not handled at the moment! Current iteration skipped...")

            # Get the first estimate
            opt_result = spopt.lsq_linear(numeric_lhs_matrix, numeric_rhs_vector, bounds=(0, np.inf))

            # Round off the results and make a Confusion Matrix
            cm_vector = np.round(opt_result.x).astype("int")
            cm = NumericalConfusionMatrix(tp=cm_vector[0], fn=cm_vector[1], fp=cm_vector[2], tn=cm_vector[3])

            # Create the EstimationResult object
            er = EstimationResult(metric_classes_set=self.metric_classes_set.copy(),
                                  cm=cm,
                                  orig_opt_result=opt_result)

            # Check if the results satisfies uncertainty ranges for all applicable metrics
            is_cm_within_uncertainty = True
            for metric_name, metric_obj in self.all_metrics_dict.items():
                if isinstance(metric_obj, LinearMetric) and (not metric_obj.perturb_during_estimation):
                    continue

                if ((er.metric_values[metric_name] < metric_obj.lower_lim) or (er.metric_values[metric_name] > metric_obj.upper_lim)):
                    is_cm_within_uncertainty = False
                    break

            # If result satisfies uncertainty ranges, add it to the er_coll
            if is_cm_within_uncertainty:
                er_coll.add_result(er)

        return er_coll


def create_symbolic_linear_eq_matrices(list_of_linear_metrics: list[LinearMetric],
                                       cm: SymbolicConfusionMatrix):

    lhs_row_list = []
    rhs_val_list = []
    for metric_obj in list_of_linear_metrics:
        lhs_row, rhs_val = metric_obj.get_symbolic_eqn(cm)
        lhs_row_list.append(lhs_row)
        rhs_val_list.append(rhs_val)

    lhs_matrix = sym_matrix(lhs_row_list)
    rhs_vector = sym_matrix(rhs_val_list)

    return lhs_matrix, rhs_vector







