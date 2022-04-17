from typing import Union, Tuple
from decimal import Decimal

import numpy as np
import sympy

from rational_utils import _POS_INFINITY, _NEG_INFINITY

class Metric:
    symb_name = ""  # These are dummy placeholders! Replace with actual values in the metric implementations!
    lower_lim = 0   # ' ' '
    upper_lim = 1   # ' ' '

    def __init__(self, supplied_value: Decimal,
                 exact_value: bool = False):


        self.supplied_value = supplied_value
        self.current_value = sympy.Rational(str(supplied_value))

        self.exact_value = exact_value

        if exact_value:
            self.uncertainty_range = (self.current_value, self.current_value)
        else:
            prec_exp = supplied_value.as_tuple().exponent
            prec_step = sympy.Rational(5, 10**(-prec_exp + 1))
            self.uncertainty_range = (self.current_value - prec_step, self.current_value + prec_step)

        self.uncertainty_range = (sympy.Max(self.lower_lim, self.uncertainty_range[0]), sympy.Min(self.upper_lim, self.uncertainty_range[1]))


    @staticmethod
    def get_symbolic_expr(cm):
        raise NotImplementedError

    @staticmethod
    def get_numeric_expr(cm):
        raise NotImplementedError


class LinearMetric(Metric):

    def __init__(self, supplied_value: Decimal,
                 perturb_during_estimation=True,
                 **metric_kwargs):

        super().__init__(supplied_value, **metric_kwargs)
        self.perturb_during_estimation = perturb_during_estimation

    @staticmethod
    def get_eqn(v):
        raise NotImplementedError

    def get_symbolic_eqn(self, cm):
        return self.get_eqn(self.get_symbolic_expr(cm))

    def get_numeric_eqn(self):
        return self.get_eqn(self.current_value)


class Accuracy(LinearMetric):
    symb_name = "acc"
    lower_lim = 0
    upper_lim = 1

    def __init__(self, supplied_value: Decimal, **kwargs):
        super().__init__(supplied_value, **kwargs)

    @staticmethod
    def get_eqn(v):
        return ([(1-v), -v, -v, (1-v)], 0)

    @staticmethod
    def get_symbolic_expr(cm):
        return ((cm.tp + cm.tn) / (cm.tp + cm.fn + cm.fp + cm.tn))

    @staticmethod
    def get_numeric_expr(cm):
        return sympy.Rational((cm.tp + cm.tn), (cm.tp + cm.fn + cm.fp + cm.tn))


class F1_Score(LinearMetric):
    symb_name = "f1"
    lower_lim = 0
    upper_lim = 1

    def __init__(self, supplied_value: Decimal, **kwargs):
        super().__init__(supplied_value, **kwargs)

    @staticmethod
    def get_eqn(v):
        return ([2*(1-v), -v, -v, 0], 0)

    @staticmethod
    def get_symbolic_expr(cm):
        return ((2*cm.tp) / ((2*cm.tp) + cm.fn + cm.fp))

    @staticmethod
    def get_numeric_expr(cm):
        return sympy.Rational((2*cm.tp), ((2*cm.tp) + cm.fn + cm.fp))


class PPV(LinearMetric):
    symb_name = "ppv"
    lower_lim = 0
    upper_lim = 1

    def __init__(self, supplied_value: Decimal, **kwargs):
        super().__init__(supplied_value, **kwargs)

    @staticmethod
    def get_eqn(v):
        return ([(1-v), 0, -v, 0], 0)

    @staticmethod
    def get_symbolic_expr(cm):
        return (cm.tp / (cm.tp + cm.fp))

    @staticmethod
    def get_numeric_expr(cm):
        return sympy.Rational(cm.tp, (cm.tp + cm.fp))


class NPV(LinearMetric):
    symb_name = "npv"
    lower_lim = 0
    upper_lim = 1

    def __init__(self, supplied_value: Decimal, **kwargs):
        super().__init__(supplied_value, **kwargs)

    @staticmethod
    def get_eqn(v):
        return ([0, -v, 0, (1-v)], 0)

    @staticmethod
    def get_symbolic_expr(cm):
        return (cm.tn / (cm.tn + cm.fn))

    @staticmethod
    def get_numeric_expr(cm):
        return sympy.Rational(cm.tn, (cm.tn + cm.fn))


class TPR(LinearMetric):
    symb_name = "tpr"
    lower_lim = 0
    upper_lim = 1

    def __init__(self, supplied_value: Decimal, **kwargs):
        super().__init__(supplied_value, **kwargs)

    @staticmethod
    def get_eqn(v):
        return ([(1-v), -v, 0, 0], 0)

    @staticmethod
    def get_symbolic_expr(cm):
        return (cm.tp / (cm.tp + cm.fn))

    @staticmethod
    def get_numeric_expr(cm):
        return sympy.Rational(cm.tp, (cm.tp + cm.fn))


class TNR(LinearMetric):
    symb_name = "tnr"
    lower_lim = 0
    upper_lim = 1

    def __init__(self, supplied_value: Decimal, **kwargs):
        super().__init__(supplied_value, **kwargs)

    @staticmethod
    def get_eqn(v):
        return ([0, 0, -v, (1-v)], 0)

    @staticmethod
    def get_symbolic_expr(cm):
        return (cm.tn / (cm.tn + cm.fp))

    @staticmethod
    def get_numeric_expr(cm):
        return sympy.Rational(cm.tn, (cm.tn + cm.fp))


class FNR(LinearMetric):
    symb_name = "fnr"
    lower_lim = 0
    upper_lim = 1

    def __init__(self, supplied_value: Decimal, **kwargs):
        super().__init__(supplied_value, **kwargs)

    @staticmethod
    def get_eqn(v):
        return ([-v, (1-v), 0, 0], 0)

    @staticmethod
    def get_symbolic_expr(cm):
        return (cm.fn / (cm.fn + cm.tp))

    @staticmethod
    def get_numeric_expr(cm):
        return sympy.Rational(cm.fn, (cm.fn + cm.tp))


class FPR(LinearMetric):
    symb_name = "fpr"
    lower_lim = 0
    upper_lim = 1

    def __init__(self, supplied_value: Decimal, **kwargs):
        super().__init__(supplied_value, **kwargs)

    @staticmethod
    def get_eqn(v):
        return ([0, 0, (1-v), -v], 0)

    @staticmethod
    def get_symbolic_expr(cm):
        return (cm.fp / (cm.fp + cm.tn))

    @staticmethod
    def get_numeric_expr(cm):
        return sympy.Rational(cm.fp, (cm.fp + cm.tn))


class FDR(LinearMetric):
    symb_name = "fdr"
    lower_lim = 0
    upper_lim = 1

    def __init__(self, supplied_value: Decimal, **kwargs):
        super().__init__(supplied_value, **kwargs)

    @staticmethod
    def get_eqn(v):
        return ([-v, 0, (1-v), 0], 0)

    @staticmethod
    def get_symbolic_expr(cm):
        return (cm.fp / (cm.fp + cm.tp))

    @staticmethod
    def get_numeric_expr(cm):
        return sympy.Rational(cm.fp, (cm.fp + cm.tp))


class FOR(LinearMetric):
    symb_name = "for"
    lower_lim = 0
    upper_lim = 1

    def __init__(self, supplied_value: Decimal, **kwargs):
        super().__init__(supplied_value, **kwargs)

    @staticmethod
    def get_eqn(v):
        return ([0, (1-v), 0, -v], 0)

    @staticmethod
    def get_symbolic_expr(cm):
        return (cm.fn / (cm.fn + cm.tn))

    @staticmethod
    def get_numeric_expr(cm):
        return sympy.Rational(cm.fn, (cm.fn + cm.tn))


class CSI(LinearMetric):
    symb_name = "csi"
    lower_lim = 0
    upper_lim = 1

    def __init__(self, supplied_value: Decimal, **kwargs):
        super().__init__(supplied_value, **kwargs)

    @staticmethod
    def get_eqn(v):
        return ([(1-v), -v, -v, 0], 0)

    @staticmethod
    def get_symbolic_expr(cm):
        return (cm.tp / (cm.tp + cm.fn + cm.fp))

    @staticmethod
    def get_numeric_expr(cm):
        return sympy.Rational(cm.tp, (cm.tp + cm.fn + cm.fp))


class Prevalence(LinearMetric):
    symb_name = "prevalence"
    lower_lim = 0
    upper_lim = 1

    def __init__(self, supplied_value: Decimal, **kwargs):
        super().__init__(supplied_value, **kwargs)

    @staticmethod
    def get_eqn(v):
        return ([(1-v), (1-v), -v, -v], 0)

    @staticmethod
    def get_symbolic_expr(cm):
        return ((cm.tp + cm.fn) / (cm.tp + cm.fn + cm.fp + cm.tn))

    @staticmethod
    def get_numeric_expr(cm):
        return sympy.Rational((cm.tp + cm.fn), (cm.tp + cm.fn + cm.fp + cm.tn))


class ActualPositive(LinearMetric):
    symb_name = "actual_p"
    lower_lim = 0
    upper_lim = _POS_INFINITY

    def __init__(self, supplied_value: Decimal, **kwargs):
        super().__init__(supplied_value, **kwargs)

    @staticmethod
    def get_eqn(v):
        return ([1, 1, 0, 0], v)

    @staticmethod
    def get_symbolic_expr(cm):
        return (cm.tp + cm.fn)

    @staticmethod
    def get_numeric_expr(cm):
        return sympy.Rational(cm.tp + cm.fn)


class ActualNegative(LinearMetric):
    symb_name = "actual_n"
    lower_lim = 0
    upper_lim = _POS_INFINITY

    def __init__(self, supplied_value: Decimal, **kwargs):
        super().__init__(supplied_value, **kwargs)

    @staticmethod
    def get_eqn(v):
        return ([0, 0, 1, 1], v)

    @staticmethod
    def get_symbolic_expr(cm):
        return (cm.tn + cm.fp)

    @staticmethod
    def get_numeric_expr(cm):
        return sympy.Rational(cm.tn + cm.fp)


class PredictedPositive(LinearMetric):
    symb_name = "pred_p"
    lower_lim = 0
    upper_lim = _POS_INFINITY

    def __init__(self, supplied_value: Decimal, **kwargs):
        super().__init__(supplied_value, **kwargs)

    @staticmethod
    def get_eqn(v):
        return ([1, 0, 1, 0], v)

    @staticmethod
    def get_symbolic_expr(cm):
        return (cm.tp + cm.fp)

    @staticmethod
    def get_numeric_expr(cm):
        return sympy.Rational(cm.tp + cm.fp)


class PredictedNegative(LinearMetric):
    symb_name = "pred_n"
    lower_lim = 0
    upper_lim = _POS_INFINITY

    def __init__(self, supplied_value: Decimal, **kwargs):
        super().__init__(supplied_value, **kwargs)

    @staticmethod
    def get_eqn(v):
        return ([0, 1, 0, 1], v)

    @staticmethod
    def get_symbolic_expr(cm):
        return (cm.tn + cm.fn)

    @staticmethod
    def get_numeric_expr(cm):
        return sympy.Rational(cm.tn + cm.fn)


class TotalNumberOfItems(LinearMetric):
    symb_name = "total"
    lower_lim = 0
    upper_lim = _POS_INFINITY

    def __init__(self, supplied_value: Decimal, **kwargs):
        super().__init__(supplied_value, **kwargs)

    @staticmethod
    def get_eqn(v):
        return ([1, 1, 1, 1], v)

    @staticmethod
    def get_symbolic_expr(cm):
        return (cm.tp + cm.fn + cm.fp + cm.tn)

    @staticmethod
    def get_numeric_expr(cm):
        return sympy.Rational(cm.tp + cm.fn + cm.fp + cm.tn)
