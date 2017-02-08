from  NonlinearTMM import _SecondOrderNLTMMCython  # @UnresolvedImport

__all__ = ["SecondOrderNLTMM"]
    
class SecondOrderNLTMM(_SecondOrderNLTMMCython.SecondOrderNLTMM): 
    
    def GetGenWaveFields2D(self, waveP1, waveP2, th0P1, th0P2, zs, xs, direction = "total"):
        waveP1.Solve(th0P1, wl = self.P1.wl)
        waveP2.Solve(th0P2, wl = self.P2.wl)
        res = super().GetGenWaveFields2D(waveP1.betas, waveP2.betas, \
            waveP1.expansionCoefsKx, waveP2.expansionCoefsKx, zs, xs, direction)
        return res
    
    def SetLayerParams(self, layerNr, **kwargs):    
        if layerNr < 0 or layerNr >= len(self.P1.layers):
            raise ValueError("LayerNr invalid")
        
        if "d" in kwargs:
            d = kwargs.pop("d")
            self.P1.layers[layerNr].d = d
            self.P2.layers[layerNr].d = d
            self.Gen.layers[layerNr].d = d
        
        for k in kwargs:
            raise ValueError("Unknown kwarg (%s)" % (k))
    
    
if __name__ == "__main__":
    pass