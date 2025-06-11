from utils.system import PassiveSurveillanceSystem

class DirectionFinder(PassiveSurveillanceSystem):
    do_2d_aoa: bool = False
    bias = None
    def __init__(self,x, cov, do_2d_aoa=False, bias=None):
        super().__init__(x, cov)

        self.do_2d_aoa = do_2d_aoa
        self.bias = bias

    # Imported methods
    import model
    import perf
    import solvers

    # static methods need to be set
    # from ._static_example import something
    # something = staticmethod(something)

    # Some more small functions
    # def printHi(self):
    #     print("Hello world")