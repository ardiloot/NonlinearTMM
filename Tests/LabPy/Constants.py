"""This module contains physical constants and functions to convert between 
different units.

Attributes:
    c (float): speed of light (m/s)
    h (float): Planck's constant (J*s)
    hp (float): Reduced Planck's constant (J*s)
    qe (float): Elementary charge (C)
    eps0 (float): Vacuum permittivity (F/m)
    mu0 (float): Vacuum permeability (A/m)

"""
import math
import numpy as np
from LabPy import Core  # @UnresolvedImport

#===============================================================================
# Definition of constants
#===============================================================================


c = 299792458.0
h = 6.62606957e-34
hp = h / (2.0 * math.pi)
qe = 1.602176565e-19 
eps0 = 8.854187817e-12
mu0 = 1.2566370614e-6
pi = math.pi
kB = 1.38064852e-23


#===============================================================================
# Wavelength to ... conversion functions
#===============================================================================


def WlToJoule(wl):
    """Converts wavelength (m) to photon energy (J).
    
    Args:
        wl (float): wavelength (m)
        
    Returns: 
        float: energy (J)
    
    """
    E = h * c / wl
    return E

def WlToEv(wl):
    """Converts wavelength (m) to energy (eV).
    
    Args:
        wl (float): wavelength
    
    Returns: 
        float: energy (eV)
    
    """
    ev = h * c / (wl * qe)
    return ev

def WlToFreq(wl):
    """Converts wavelength to frequency (Hz).
    
    Args:
    wl (float): wavelength (m)
    
    Returns: 
        float: frequency (Hz)
    
    """
    freq = c / wl
    return freq 


def WlToOmega(wl):
    """Converts wavelength to angular frequency (rad/s).
    
    Args:
        wl (float): wavelength (m)
    
    Returns:
        float: angular frequency (rad/s)
    
    """
    omega = 2.0 * math.pi * c / wl
    return omega

#===============================================================================
# ... to wavelength conversion functions
#===============================================================================

def JouleToWl(E):
    """Converts energy of a photon (J) to wavelength (m).
    
    Args:
        E (float): energy (J)
    
    Returns: 
        float: wavelength (nm)
    
    """
    wl = h * c / E
    return wl


def EvToWl(ev):
    """Converts energy (eV) to wavelength (m).
    
    Args:
        ev (float): energy (eV)
    
    Returns: 
        float: wavelength (m)
    
    """
    wl = h * c / (ev * qe)
    return wl


def FreqToWl(freq):
    """Converts frequency (1/s) to wavelength (m).
    
    Args:
        freq (float): frequency (1/s)
        
    Returns: 
        float: wavelength (m)
        
    """
    wl = c / freq
    return wl


def OmegaToWl(omega):
    """Converts angular frequency (rad/s) to wavelength (m).
    
    Args:
        omega (float): angular frequency (rad/s)

    Returns: 
        float: wavelength (m)
    
    """
    wl = 2.0 * math.pi * c / omega
    return wl

#===============================================================================
# Other conversion functions
#===============================================================================
 
def EvToJoule(ev):
    """Converts eV-s to joules.
    
    Args:
        ev (float): Energy in eV
        
    Returns:
        float: Energy in J
        
    """
    
    return ev * qe
 
def JouleToEv(E):
    """Converts joules to eV-s.
    
    Args:
        E (float): Energy in J
        
    Returns: 
        float: Energy in eV
        
    """
    
    return E / qe

def EvToOmega(ev):
    """Converts eV-s to angular frequency.
    
    Args:
        ev (float): Energy in eV
        
    Returns: 
        float: Photon angular frequency (rad/s)
        
    """
    return 2.0 * math.pi * ev * qe / h

def OmegaToEv(omega):
    """Converts angular frequency to eV-s.
    
    Args:
        omega (float): photon angular frequency (rad/s)
        
    Returns: 
        float: Photon energy (eV)
        
    """
    return omega * h / (2.0 * pi * qe)

#===============================================================================
# Distributions
#===============================================================================     
    

def GetDistributionNames():
    res = ["DetlaDistribution", "NormalDistribution", \
                     "LogNormalDistribution", "LogNormalDistributionLocal"]
    return res

    
def Distribution(name, **kwargs):
    if name == "DetlaDistribution":
        return DetlaDistribution(**kwargs)
    elif name == "NormalDistribution":
        return NormalDistribution(**kwargs)
    elif name == "LogNormalDistribution":
        return LogNormalDistribution(**kwargs)
    elif name == "LogNormalDistributionLocal":
        return LogNormalDistributionLocal(**kwargs)
    else:
        raise ValueError("Unknown distribution name")
    
class DetlaDistribution(Core.ParamsBaseClass):
    """This class presents delta function distribution.
    
    Delta function distribution is zero except at x=x0 where it is 1.0.
    
    Kwargs:
        x0=0.0 (float): The position of delta function
    
    """
    
    def __init__(self, **kwargs):
        self._params = ["x0"]
        self.x0 = 0.0
        
        super(DetlaDistribution, self).__init__(**kwargs)
        
    def __call__(self, x):
        """Returns distribution value at position x.
        
        Args:
            x (float): The point were to evaluate the distribution
        
        Returns:
            float: Distribution value at position x
            
        """
        if x == self.x0:
            return 1.0
        else:
            return 0.0
        
    def GetDist(self):
        """Function returns (np.array([x0]), np.array([1.0])).
        
        Returns:
            tuple_of_arrays: Distribution positions and values
            
        """
        return (np.array([self.x0]), np.array([1.0]))
        
class NormalDistribution(Core.ParamsBaseClass):
    """This class presents normal distribution.
    
    Kwargs:
        x0=0.0 (float): mean
        std=1.0 (float): standard deviation
        nPoints=30 (int): number of points in distribution
        rangeB=3.0 (float): beginning of sampling range is x0 - ramgeB * std
        rangeE=3.0 (float): end of sampling range is x0 + ramgeE * std
    
    """
    
    def __init__(self, **kwargs):
        self._params = ["x0", "std", "nPoints", "rangeB", "rangeE"]
        self.x0 = 50e-9
        self.std = 10e-9
        self.nPoints = 30
        self.rangeB = 3.0
        self.rangeE = 3.0
        
        super(NormalDistribution, self).__init__(**kwargs)

    def __call__(self, x):
        """Returns distribution value at position x.
        
        Args:
            x (float): The point were to evaluate the distribution
        
        Returns:
            float: Distribution value at position x
            
        """
        res = 1.0 / (self.std * math.sqrt(2.0 * math.pi)) * \
            np.exp(- (x - self.x0) ** 2.0 / (2.0 * self.std ** 2.0))
        
        return res

    def GetDist(self):
        """Function returns distribution at nPoints sampling points.
        
        The sampling range is from max(1e-16, x0 - rangeB * std) to \
        x0 + rangeE * std where nPoints sampling points are taken.
        
        Returns:
            tuple_of_arrays: Distribution positions and values
            
        """
        xs = np.linspace(max(1e-12, self.x0 - self.rangeB * self.std), \
                         self.x0 + self.rangeE * self.std, self.nPoints)

        
        ws = self.__call__(xs)
        return xs, ws
            
class LogNormalDistribution(NormalDistribution):
    """This class presents lognormal distribution.
    
    Kwargs:
        x0=0.0 (float): mean
        std=1.0 (float): standard deviation
        nPoints=30 (int): number of points in distribution
        rangeB=0.0 (float): beginning of sampling range
        rangeE=10.0 (float): end of sampling range
    
    """
    def __init__(self, **kwargs):
        super(LogNormalDistribution, self).__init__(**kwargs)
   
    def __call__(self, x):
        """Returns distribution value at position x.
        
        Args:
            x (float): The point were to evaluate the distribution
        
        Returns:
            float: Distribution value at position x
            
        """
        #x02 = math.log(self.x0 ** 2.0 / math.sqrt(self.std + self.x0 ** 2.0))
        #std2 = math.sqrt(math.log(1.0 + self.std / self.x0 ** 2.0))
        x02 = np.log(self.x0)
        std2 = self.std
        
        #res = 1.0 / (std2 * math.sqrt(2.0 * math.pi) * x) * \
        #    np.exp(- (np.log(x) - x02) ** 2.0 / (2.0 * std2 ** 2.0))
        
        res = 1.0 / (std2 * math.sqrt(2.0 * math.pi)) * \
            np.exp(- (np.log(x) - x02) ** 2.0 / (2.0 * std2 ** 2.0))
        
       
        return res

    def GetDist(self):
        """Function returns distribution at nPoints sampling points.
        
        The sampling range is from max(1e-16, x0 - rangeB * std) to \
        x0 + rangeE * std where nPoints sampling points are taken.
        
        Returns:
            tuple_of_arrays: Distribution positions and values
            
        """
        xs = np.linspace(self.rangeB, self.rangeE, self.nPoints)

        ws = self.__call__(xs)
        return xs, ws

class LogNormalDistributionLocal(Core.ParamsBaseClass):
    """This class presents lognormal distribution with usual local parameters.
    
    Kwargs:
        xCoef (float): x-scale linear transformation parameter
        xOffset (float): x-offset value (new_x = x + xOffset)
        mu (float): location parameter
        sigma (float): scale parameter
        nPoint (int): number of points in distribution
        rangeB (float): beginning of sampling range
        rangeE (float): end of sampling range
        
    
    """
    def __init__(self, **kwargs):
        self._params = ["xCoef", "xOffset", "mu", "sigma", "nPoints", "distB", "distE"]
        self.xCoef = 1.0
        self.xOffset = 0.0
        self.mu = 0.0
        self.sigma = 1.0
        self.nPoints = 30
        self.distB = 0.0
        self.distE = 3.0
        
        super(LogNormalDistributionLocal, self).__init__(**kwargs)

    def __call__(self, x):
        """Returns distribution value at position x.
        
        Args:
            x (float): The point were to evaluate the distribution
        
        Returns:
            float: Distribution value at position x
            
        """
        
        x = self.xCoef * (x - self.xOffset)
        
        res = 1.0 / (self.sigma * math.sqrt(2.0 * math.pi) * x) * \
            np.exp(- (np.log(x) - self.mu) ** 2.0 / (2.0 * self.sigma ** 2.0))
        
        return res
    
    def GetDist(self):
        """Function returns distribution at nPoints sampling points.
        
        The sampling range is from max(1e-16, distB) to \
        distE where nPoints sampling points are taken.
        
        Returns:
            tuple_of_arrays: Distribution positions and values
            
        """
        xs = np.linspace(max(self.xOffset + 1e-16, self.distB), \
                         self.distE, self.nPoints)
        
        ws = self.__call__(xs)
        return xs, ws
        

if __name__ == "__main__":
    pass
#     import pylab as plt
#     dist = LogNormalDistribution(x0 = 10.0, std = 5.0)
#     print dist.GetParams()
#     xs, ws = dist.GetDist()
#     
#     plt.figure()
#     plt.plot(xs, ws, "x-")
#     plt.show()
#     
    