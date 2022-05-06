from sympy import Symbol
import numpy as np
import mip

from rational_metrics import FractionalMetric, LinearMetric

class SymbolicConfusionMatrix:

    def __init__(self):

        self.tp = Symbol("tp")
        self.fn = Symbol("fn")
        self.fp = Symbol("fp")
        self.tn = Symbol("tn")


class NumericalConfusionMatrix:

    def __init__(self, tp: int, fn: int, fp: int, tn: int):

        self.tp = tp
        self.fn = fn
        self.fp = fp
        self.tn = tn


class MIPConfusionMatrix:

    def __init__(self):

        self.model = mip.Model()
        self.model.verbose = 0

        self.tp = self.model.add_var(name="tp", lb=0, var_type=mip.INTEGER)
        self.fn = self.model.add_var(name="fn", lb=0, var_type=mip.INTEGER)
        self.fp = self.model.add_var(name="fp", lb=0, var_type=mip.INTEGER)
        self.tn = self.model.add_var(name="tn", lb=0, var_type=mip.INTEGER)

    def add_metric_constraint(self, metric_obj: LinearMetric):

        if metric_obj.exact_value:
            eqn_lhs, eqn_rhs = metric_obj.get_mip_eqn(metric_obj.current_value, self)
            constr = (eqn_lhs == eqn_rhs)
            self.model.add_constr(constr, name=f"{metric_obj.symb_name}_exact")

        else:
            lower_lim_eqn_lhs, lower_lim_eqn_rhs = metric_obj.get_mip_eqn(metric_obj.uncertainty_range[0], self)
            constr = (lower_lim_eqn_lhs >= lower_lim_eqn_rhs)
            self.model.add_constr(constr, name=f"{metric_obj.symb_name}_lower_lim")

            upper_lim_eqn_lhs, upper_lim_eqn_rhs = metric_obj.get_mip_eqn(metric_obj.uncertainty_range[1], self)
            constr = (upper_lim_eqn_lhs <= upper_lim_eqn_rhs)
            self.model.add_constr(constr, name=f"{metric_obj.symb_name}_upper_lim")

        return


    def iterative_function_for_optimizing_fractional_metric(self, metric_obj: FractionalMetric,
                                                            optimize_direction: str,
                                                            max_num_iterations: int = 100,
                                                            iter_obj_epsilon: float = 1e-5):
        """
        optimize_direction can be one of "min" or "max"

        I found this algorithm at:
        https://optimization.cbe.cornell.edu/index.php?title=Mixed-integer_linear_fractional_programming_(MILFP)#Parametric_Algorithm
        """

        target_num, target_den = metric_obj.get_fractional_repr(self)

        optimization_status = None
        best_metric_value = None

        iteration_success = False
        q = 0

        for _ in range(max_num_iterations):

            opt_target = target_num - (q * target_den)

            if optimize_direction == "min":
                self.model.objective = mip.minimize(opt_target)
            else:
                self.model.objective = mip.maximize(opt_target)

            status = self.model.optimize()

            if status != mip.OptimizationStatus.OPTIMAL:
                optimization_status = status
                break

            computed_num = target_num.x
            computed_den = target_den.x
            q = computed_num / computed_den

            if self.model.objective_value < iter_obj_epsilon:
                iteration_success = True
                break

        if iteration_success:
            optimization_status = mip.OptimizationStatus.OPTIMAL
            best_metric_value = computed_num / computed_den

        return optimization_status, iteration_success, best_metric_value


    def optimize_for_metric(self, metric_obj: LinearMetric,
                            optimize_direction: str):
        """
        optimize_direction can be one of "min" or "max"

        """

        # -----------
        # TODO: Implement this!
        if not isinstance(metric_obj, FractionalMetric):
            raise NotImplementedError("optimize_for_metric() only supports FractionalMetric for now!")
        # -----------

        optimization_status, iteration_success, best_metric_value = self.iterative_function_for_optimizing_fractional_metric(
            metric_obj, optimize_direction
        )

        return optimization_status, iteration_success, best_metric_value
