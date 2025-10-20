"""
Element-based solver for orbital elements ELM.
"""

import numpy as np
from scipy.optimize import least_squares
from .elm_elements import OrbitalElementsELM
from .loss_elements import residual_elements, evaluate_solution_elements

def fit_elm_elements(t0, t1, L=8, N_colloc=20, lam_f=1.0,
                    obs=None, t_obs=None, station_eci=None, lam_th=1.0, seed=42):
    """
    Fit orbital elements ELM using nonlinear least squares.
    
    Parameters:
    -----------
    t0, t1 : float
        Time bounds
    L : int
        Number of hidden neurons
    N_colloc : int
        Number of collocation points
    lam_f : float
        Physics residual weight
    obs : ndarray, optional
        Observations
    t_obs : ndarray, optional
        Observation times
    station_eci : ndarray, optional
        Station positions
    lam_th : float
        Measurement residual weight
    seed : int
        Random seed
        
    Returns:
    --------
    beta : ndarray
        Fitted parameters
    model : OrbitalElementsELM
        The fitted model
    result : OptimizeResult
        Optimization result
    """
    
    # Create time grids
    t_colloc = np.linspace(t0, t1, N_colloc)
    
    # Create model
    model = OrbitalElementsELM(L=L, t_phys=np.array([t0, t1]), seed=seed)
    
    # Initialize beta with reasonable GEO values
    beta0 = np.array([
        42164000.0,  # a (GEO altitude)
        0.0,         # e (circular)
        0.0,         # i (equatorial)
        0.0,         # Omega
        0.0,         # omega
        0.0,         # M0
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # ELM weights (small)
    ])
    
    # Add small random variations to initial guess
    beta0[6:] = np.random.randn(6) * 0.01
    
    def fun(beta):
        return residual_elements(beta, model, t_colloc, lam_f, obs, t_obs, station_eci, lam_th)
    
    # Optimize
    result = least_squares(fun, beta0, method="trf", max_nfev=5000, 
                          ftol=1e-10, xtol=1e-10, gtol=1e-10)
    
    return result.x, model, result


def test_solver_elements():
    """Test the element-based solver."""
    print("Testing element-based solver...")
    
    # Test parameters
    t0, t1 = 0.0, 7200.0  # 2 hours
    L = 8
    N_colloc = 20
    
    # Test physics-only fitting
    print("Testing physics-only fitting...")
    beta, model, result = fit_elm_elements(t0, t1, L=L, N_colloc=N_colloc, lam_f=1.0)
    
    print(f"Optimization success: {result.success}")
    print(f"Function evaluations: {result.nfev}")
    print(f"Final cost: {result.cost:.2e}")
    
    # Evaluate solution
    t_eval = np.linspace(t0, t1, 50)
    r, v, a, physics_rms, measurement_rms = evaluate_solution_elements(beta, model, t_eval)
    
    print(f"Physics RMS: {physics_rms:.6f}")
    print(f"Measurement RMS: {measurement_rms:.6f}")
    
    # Check orbital elements
    mean_elements, elm_weights = model.elements_from_beta(beta)
    print(f"Mean elements: a={mean_elements[0]/1000:.1f}km, e={mean_elements[1]:.6f}")
    
    print("✓ Element-based solver test completed")
    return beta, model, result


if __name__ == "__main__":
    test_solver_elements()
