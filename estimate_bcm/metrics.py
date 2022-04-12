from typing import Union, Tuple
from decimal import Decimal

import numpy as np


class Metric:
    symb_name = ""      # These are dummy placeholders! Replace with actual values in the metric implementations!
    lower_lim = 0.0     # ' ' '
    upper_lim = 1.0     # ' ' '

    def __init__(self, supplied_value: Decimal,
                 exact_value: bool = False,
                 uncertainty_range_param: Union[str, float, tuple, None] = "round"):

        self.supplied_value = supplied_value
        self.current_value = float(self.supplied_value)

        self.exact_value = exact_value

        self.uncertainty_range_param = uncertainty_range_param
        self.uncertainty_range = self._compute_uncertainty_range(
            uncertainty_range_param=self.uncertainty_range_param,
            current_value=self.current_value,
            supplied_value=self.supplied_value,
            lower_lim=self.lower_lim,
            upper_lim=self.upper_lim
        )


    @staticmethod
    def get_expr(cm):
        raise NotImplementedError


    @staticmethod
    def _compute_uncertainty_range(uncertainty_range_param: Union[str, float, tuple, None],
                                   current_value: float,
                                   supplied_value: Decimal,
                                   lower_lim: float, upper_lim: float):

        if uncertainty_range_param is None:
            uncertainty_range = (current_value, current_value)

        elif isinstance(uncertainty_range_param, str):
            if uncertainty_range_param.endswith("%"):
                uncertainty_range = Metric._compute_metric_uncertainty_range_from_tuple(
                    t=(uncertainty_range_param, uncertainty_range_param),
                    orig_val=current_value
                )

            else:
                prec_exp = supplied_value.as_tuple().exponent
                prec_step = float(f"1e{prec_exp}")

                if uncertainty_range_param == "round":
                    uncertainty_range = (current_value - (prec_step/2.0), current_value + (prec_step/2.0))
                elif uncertainty_range_param == "floor":
                    uncertainty_range = (current_value - prec_step, current_value)
                elif uncertainty_range_param == "ceil":
                    uncertainty_range = (current_value, current_value + prec_step)
                else:
                    raise ValueError(f"{uncertainty_range_param} is not a valid string for uncertainty_range_param; Expected one of 'round', 'floor', 'ceil'")


        elif isinstance(uncertainty_range_param, (float, int)):
            uncertainty_range = Metric._compute_metric_uncertainty_range_from_tuple(
                t=(uncertainty_range_param, uncertainty_range_param),
                orig_val=current_value
            )

        elif isinstance(uncertainty_range_param, tuple):

            if len(uncertainty_range_param) != 2:
                raise SyntaxError(f"{uncertainty_range_param} is not a valid tuple for uncertainty_range_param; Expected a tuple of length 2")

            uncertainty_range = Metric._compute_metric_uncertainty_range_from_tuple(
                t=uncertainty_range_param,
                orig_val=current_value
            )

        else:
            raise SyntaxError(f"{uncertainty_range_param} (type: {type(uncertainty_range_param)}) is not a valid uncertainty_range_param; Expected a str, float, tuple, or None")

        # Ensure the range is not outside the allowed range
        uncertainty_range = (max(lower_lim, uncertainty_range[0]), min(upper_lim, uncertainty_range[1]))

        return uncertainty_range


    @staticmethod
    def _compute_metric_uncertainty_range_from_tuple(t: Tuple[Union[str, float], Union[str, float]],
                                                     orig_val: float):

        diff_tuple = (None, None)
        for i in range(2):

            if isinstance(t[i], str):
                if not t[i].endswith("%"):
                    raise ValueError(f"Element {i} of uncertainty_range_param tuple {t} must be a string ending with %")

                ratio = float(t[i][:-1]) / 100.0
                diff_tuple[i] = orig_val * ratio

            else:
                diff_tuple[i] = t[i]

        uncertainty_range = (orig_val - diff_tuple[0], orig_val + diff_tuple[1])
        return uncertainty_range


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
        return self.get_eqn(self.get_expr(cm))

    def get_numeric_eqn(self):
        return self.get_eqn(self.current_value)


class Accuracy(LinearMetric):
    symb_name = "acc"
    lower_lim = 0.0
    upper_lim = 1.0

    def __init__(self, supplied_value: Decimal, **kwargs):
        super().__init__(supplied_value, **kwargs)

    @staticmethod
    def get_eqn(v):
        return ([1-v, -v, -v, 1-v], 0)

    @staticmethod
    def get_expr(cm):
        return (cm.tp + cm.tn) / (cm.tp + cm.fn + cm.fp + cm.tn)


class F1_Score(LinearMetric):
    symb_name = "f1"
    lower_lim = 0.0
    upper_lim = 1.0

    def __init__(self, supplied_value: Decimal, **kwargs):
        super().__init__(supplied_value, **kwargs)

    @staticmethod
    def get_eqn(v):
        return ([2*(1-v), -v, -v, 0], 0)

    @staticmethod
    def get_expr(cm):
        return (2*cm.tp) / ((2*cm.tp) + cm.fn + cm.fp)


class PPV(LinearMetric):
    symb_name = "ppv"
    lower_lim = 0.0
    upper_lim = 1.0

    def __init__(self, supplied_value: Decimal, **kwargs):
        super().__init__(supplied_value, **kwargs)

    @staticmethod
    def get_eqn(v):
        return ([(1-v), 0, -v, 0], 0)

    @staticmethod
    def get_expr(cm):
        return cm.tp / (cm.tp + cm.fp)


class ActualPositive(LinearMetric):
    symb_name = "actual_p"
    lower_lim = 0
    upper_lim = np.inf

    def __init__(self, supplied_value: Decimal, **kwargs):
        super().__init__(supplied_value, **kwargs)

    @staticmethod
    def get_eqn(v):
        return ([1, 1, 0, 0], v)

    @staticmethod
    def get_expr(cm):
        return cm.tp + cm.fn


class ActualNegative(LinearMetric):
    symb_name = "actual_n"
    lower_lim = 0
    upper_lim = np.inf

    def __init__(self, supplied_value: Decimal, **kwargs):
        super().__init__(supplied_value, **kwargs)

    @staticmethod
    def get_eqn(v):
        return ([0, 0, 1, 1], v)

    @staticmethod
    def get_expr(cm):
        return cm.tn + cm.fp


class PredictedPositive(LinearMetric):
    symb_name = "pred_p"
    lower_lim = 0
    upper_lim = np.inf

    def __init__(self, supplied_value: Decimal, **kwargs):
        super().__init__(supplied_value, **kwargs)

    @staticmethod
    def get_eqn(v):
        return ([1, 0, 1, 0], v)

    @staticmethod
    def get_expr(cm):
        return cm.tp + cm.fp


class PredictedNegative(LinearMetric):
    symb_name = "pred_n"
    lower_lim = 0
    upper_lim = np.inf

    def __init__(self, supplied_value: Decimal, **kwargs):
        super().__init__(supplied_value, **kwargs)

    @staticmethod
    def get_eqn(v):
        return ([0, 1, 0, 1], v)

    @staticmethod
    def get_expr(cm):
        return cm.tn + cm.fn


class TotalNumberOfItems(LinearMetric):
    symb_name = "total"
    lower_lim = 0
    upper_lim = np.inf

    def __init__(self, supplied_value: Decimal, **kwargs):
        super().__init__(supplied_value, **kwargs)

    @staticmethod
    def get_eqn(v):
        return ([1, 1, 1, 1], v)

    @staticmethod
    def get_expr(cm):
        return cm.tp + cm.fn + cm.fp + cm.tn
