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


    def direct_function_for_optimizing_linear_metric(self, metric_class,
                                                     optimize_direction: str):
        """
        optimize_direction can be one of "min" or "max"
        """

        best_metric_value = None

        opt_target = metric_class.get_symbolic_expr(self)

        if optimize_direction == "min":
            self.model.objective = mip.minimize(opt_target)
        else:
            self.model.objective = mip.maximize(opt_target)


        optimization_status = self.model.optimize()
        if optimization_status == mip.OptimizationStatus.OPTIMAL:
            best_metric_value = self.model.objective_value

        return optimization_status, best_metric_value


    def iterative_function_for_optimizing_fractional_metric(self, metric_class,
                                                            optimize_direction: str,
                                                            max_num_iterations: int = 100,
                                                            iter_obj_epsilon: float = 1e-5):
        """
        optimize_direction can be one of "min" or "max"

        I found this algorithm at:
        https://optimization.cbe.cornell.edu/index.php?title=Mixed-integer_linear_fractional_programming_(MILFP)#Parametric_Algorithm
        """

        target_num, target_den = metric_class.get_fractional_repr(self)

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

            if self.model.objective_value < iter_obj_epsilon:
                iteration_success = True
                break

            q = target_num.x / target_den.x


        if iteration_success:
            optimization_status = mip.OptimizationStatus.OPTIMAL

            if target_den.x == 0:
                if target_num.x == 0:
                    best_metric_value = 0
                else:
                    raise ZeroDivisionError
            else:
                best_metric_value = target_num.x / target_den.x

        else:
            if status == mip.OptimizationStatus.OPTIMAL:    # I.e. OPTIMAL, but not iteration_success --> Iteration ended without error, but without reaching optimal value
                optimization_status = mip.OptimizationStatus.OTHER

        return optimization_status, best_metric_value


    def optimize_for_metric(self, metric_class,
                            optimize_direction: str):
        """
        optimize_direction can be one of "min" or "max"
        """

        if issubclass(metric_class, FractionalMetric):
            optimization_status, best_metric_value = self.iterative_function_for_optimizing_fractional_metric(
                metric_class, optimize_direction
            )

        elif issubclass(metric_class, LinearMetric):
            optimization_status, best_metric_value = self.direct_function_for_optimizing_linear_metric(
                metric_class, optimize_direction
            )

        else:
            raise NotImplementedError("optimize_for_metric() only supports LinearMetric and subclasses for now!")

        return optimization_status, best_metric_value
