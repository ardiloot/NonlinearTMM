from  NonlinearTMM import _SecondOrderNLTMMCython  # @UnresolvedImport

__all__ = ["SecondOrderNLTMM"]
    
class SecondOrderNLTMM(_SecondOrderNLTMMCython.SecondOrderNLTMM): 
    
    def GetGenWaveFields2D(self, waveP1, waveP2, th0P1, th0P2, zs, xs, direction = "total"):
        waveP1.Solve(self.P1.wl, th0P1)
        waveP2.Solve(self.P2.wl, th0P2)
        res = super().GetGenWaveFields2D(waveP1.betas, waveP2.betas, \
            waveP1.expansionCoefsKx, waveP2.expansionCoefsKx, zs, xs, direction)
        return res
    
if __name__ == "__main__":
    pass