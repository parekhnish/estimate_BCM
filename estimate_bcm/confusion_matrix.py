from sympy import Symbol
import numpy as np

class SymbolicConfusionMatrix:

    def __init__(self):

        self.tp = Symbol("tp")
        self.fn = Symbol("fn")
        self.fp = Symbol("fp")
        self.tn = Symbol("tn")


class NumericalConfusionMatrix:

    def __init__(self, tp=np.nan, fn=np.nan, fp=np.nan, tn=np.nan):

        self.tp = tp
        self.fn = fn
        self.fp = fp
        self.tn = tn
