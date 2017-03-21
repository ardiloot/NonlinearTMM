import numpy as np
import pytest
import LabPy
from NonlinearTMM import Material

@pytest.fixture(scope = "module", params = ["main/Ag/Johnson", "main/Au/Johnson"])
def materials(request):    
    labpyMaterial = LabPy.Material(request.param)
    tmmMaterial = Material.FromLabPy(labpyMaterial) 
    return (tmmMaterial, labpyMaterial)

@pytest.fixture(scope = "module")
def chi2s():
    dValues = {"d11": 1e-12, "d22": 2e-12, "d33": 3e-12, "d12": 4e-12, "d23": 5e-12}
    materialCpp = Material.FromLabPy(LabPy.Material("Static", n = 1.0))
    chi2Cpp = materialCpp.chi2
    chi2Cpp.Update(**dValues)
    chi2Py = LabPy.Chi2Tensor(**dValues)
    return chi2Cpp, chi2Py

@pytest.fixture(params = np.linspace(300e-9, 1500e-9, 100))
def wl(request):
    return request.param

@pytest.fixture(params = [0.0, 0.5, 1.2])
def phiX(request):
    return request.param

@pytest.fixture(params = [0.0, 0.5, 1.2])
def phiY(request):
    return request.param

@pytest.fixture(params = [0.0, 0.5, 1.2])
def phiZ(request):
    return request.param

def test_RefractiveIndex(materials, wl):
    tmmMaterial, labpyMaterial = materials
    np.testing.assert_almost_equal(tmmMaterial.GetN(wl), labpyMaterial(wl))
    
def test_Chi2Tensor(chi2s, phiX, phiY, phiZ):
    chi2Cpp, chi2Py = chi2s
    chi2Cpp.Update(phiX = phiX, phiY = phiY, phiZ = phiZ)
    chi2Py.Update(phiX = phiX, phiY = phiY, phiZ = phiZ)
    
    tensorPy = chi2Py.GetChi2().real
    tensorCpp = chi2Cpp.GetChi2Tensor()
    np.testing.assert_allclose(tensorPy, tensorCpp)
    
    