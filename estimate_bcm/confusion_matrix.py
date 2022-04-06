from sympy import Symbol
import numpy as np

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
