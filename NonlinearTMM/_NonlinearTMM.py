from NonlinearTMM import _SecondOrderNLTMMCython  # @UnresolvedImport

__all__ = ["NonlinearTMM"]
    
class NonlinearTMM(_SecondOrderNLTMMCython.NonlinearTMM): # @UndefinedVariable
    
    def GetFields2DWaves(self, wave, th0, zs, xs, direction = "total"):
        wave.Solve(self.wl, th0)
        res = self.IntegrateFields2D("beta", wave.betas, wave.expansionCoefsPhi, wave.phis, zs, xs)
        return res
        

if __name__ == "__main__":
    pass