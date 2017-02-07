from NonlinearTMM import _SecondOrderNLTMMCython  # @UnresolvedImport

__all__ = ["NonlinearTMM"]
    
class NonlinearTMM(_SecondOrderNLTMMCython.NonlinearTMM): # @UndefinedVariable
    
    def GetWaveFields2D(self, wave, th0, zs, xs, direction = "total"):
        wave.Solve(self.wl, th0)
        res = super().GetWaveFields2D(wave.betas, wave.expansionCoefsKx, zs, xs, direction)
        return res
        

if __name__ == "__main__":
    pass