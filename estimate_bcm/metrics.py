from sympy import Symbol


class Metric:
    symb_name = "placeholder_metric"

    def __init__(self, supplied_value):
        self._symbol = Symbol(self.symb_name)
        self.supplied_value = supplied_value

    @property
    def symbol(self):
        return self._symbol

    @staticmethod
    def get_expr(cm):
        raise NotImplementedError


class LinearMetric(Metric):
    symb_name = "placeholder_linear_metric"

    def __init__(self, supplied_value):
        super().__init__(supplied_value)

    @staticmethod
    def get_eqn(v):
        raise NotImplementedError

    def get_symbolic_eqn(self):
        return self.get_eqn_for_linear_system(self.symbol)

    def get_numeric_eqn(self):
        return self.get_eqn(self.supplied_value)


class Accuracy(LinearMetric):
    symb_name = "acc"

    def __init__(self, supplied_value):
        super().__init__(supplied_value)

    @staticmethod
    def get_eqn(v):
        return ([1-v, -v, -v, 1-v], 0)

    @staticmethod
    def get_expr(cm):
        return (cm.tp + cm.tn) / (cm.tp + cm.fn + cm.fp + cm.tn)


class F1_Score(LinearMetric):
    symb_name = "f1"

    def __init__(self, supplied_value):
        super().__init__(supplied_value)

    @staticmethod
    def get_eqn(v):
        return ([2*(1-v), -v, -v, 0], 0)

    @staticmethod
    def get_expr(cm):
        return (2*cm.tp) / ((2*cm.tp) + cm.fn + cm.fp)


class PPV(LinearMetric):
    symb_name = "ppv"

    def __init__(self, supplied_value):
        super().__init__(supplied_value)

    @staticmethod
    def get_eqn(v):
        return ([(1-v), 0, -v, 0], 0)

    @staticmethod
    def get_expr(cm):
        return cm.tp / (cm.tp + cm.fp)


class ActualPositive(LinearMetric):
    symb_name = "actual_p"

    def __init__(self, supplied_value):
        super().__init__(supplied_value)

    @staticmethod
    def get_eqn(v):
        return ([1, 1, 0, 0], v)

    @staticmethod
    def get_expr(cm):
        return cm.tp + cm.fn


class ActualNegative(LinearMetric):
    symb_name = "actual_n"

    def __init__(self, supplied_value):
        super().__init__(supplied_value)

    @staticmethod
    def get_eqn(v):
        return ([0, 0, 1, 1], v)

    @staticmethod
    def get_expr(cm):
        return cm.tn + cm.fp


class PredictedPositive(LinearMetric):
    symb_name = "pred_p"

    def __init__(self, supplied_value):
        super().__init__(supplied_value)

    @staticmethod
    def get_eqn(v):
        return ([1, 0, 1, 0], v)

    @staticmethod
    def get_expr(cm):
        return cm.tp + cm.fp


class PredictedNegative(LinearMetric):
    symb_name = "pred_n"

    def __init__(self, supplied_value):
        super().__init__(supplied_value)

    @staticmethod
    def get_eqn(v):
        return ([0, 1, 0, 1], v)

    @staticmethod
    def get_expr(cm):
        return cm.tn + cm.fn
