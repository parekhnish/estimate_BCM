from sympy import Symbol
import numpy as np
import mip

from rational_metrics import LinearMetric

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
