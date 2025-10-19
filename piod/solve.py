"""
Solver module for physics-informed ELM using nonlinear least squares.
"""

import numpy as np
from scipy.optimize import least_squares
from .elm import GeoELM
from .loss import residual, physics_residual_rms, measurement_residual_rms


def fit_elm(t0, t1, L=48, N_colloc=80, lam_f=10.0,
            obs=None, t_obs=None, station_eci=None, lam_th=1.0, seed=42,
            max_nfev=20000, ftol=1e-10, xtol=1e-10, gtol=1e-10):
    """
    Fit ELM using nonlinear least squares on physics and measurement residuals.
    
    Parameters:
    -----------
    t0, t1 : float
        Time bounds for the arc (seconds)
    L : int
        Number of hidden neurons (default: 48)
    N_colloc : int
        Number of collocation points (default: 80)
    lam_f : float
        Weight for physics residuals (default: 10.0)
    obs : array_like, shape (3, N_obs), optional
        Observed trig components [sin(RA), cos(RA), sin(DEC)]
    t_obs : array_like, optional
        Observation time points
    station_eci : array_like, shape (3, N_obs), optional
        Station position in ECI frame at observation times
    lam_th : float
        Weight for measurement residuals (default: 1.0)
    seed : int
        Random seed for ELM initialization (default: 42)
    max_nfev : int
        Maximum number of function evaluations (default: 20000)
    ftol : float
        Function tolerance for convergence (default: 1e-10)
    xtol : float
        Parameter tolerance for convergence (default: 1e-10)
    gtol : float
        Gradient tolerance for convergence (default: 1e-10)
        
    Returns:
    --------
    beta : ndarray, shape (3*L,)
        Optimized output weights
    model : GeoELM
        ELM model instance
    result : scipy.optimize.OptimizeResult
        Optimization result
    """
    # Create time grids
    t_colloc = np.linspace(t0, t1, N_colloc)
    
    # Initialize ELM
    model = GeoELM(L=L, t_phys=np.array([t0, t1]), seed=seed)
    
    # Initialize output weights with small random values instead of zeros
    # This prevents division by zero in dynamics when position is zero
    beta0 = np.random.randn(3 * L) * 0.01
    
    # Define residual function
    def fun(beta):
        return residual(beta, model, t_colloc, lam_f,
                       obs, t_obs, station_eci, lam_th)
    
    # Solve using Levenberg-Marquardt (trust-region reflective)
    result = least_squares(fun, beta0, method="trf",
                          max_nfev=max_nfev, ftol=ftol, xtol=xtol, gtol=gtol)
    
    return result.x, model, result


def fit_physics_only(t0, t1, L=48, N_colloc=80, lam_f=10.0, seed=42):
    """
    Convenience function to fit ELM with physics residuals only.
    
    Parameters:
    -----------
    t0, t1 : float
        Time bounds for the arc (seconds)
    L : int
        Number of hidden neurons
    N_colloc : int
        Number of collocation points
    lam_f : float
        Weight for physics residuals
    seed : int
        Random seed for ELM initialization
        
    Returns:
    --------
    beta : ndarray, shape (3*L,)
        Optimized output weights
    model : GeoELM
        ELM model instance
    result : scipy.optimize.OptimizeResult
        Optimization result
    """
    return fit_elm(t0, t1, L=L, N_colloc=N_colloc, lam_f=lam_f, seed=seed)


def evaluate_solution(beta, model, t_eval, obs=None, t_obs=None, station_eci=None):
    """
    Evaluate the fitted ELM solution and compute residuals.
    
    Parameters:
    -----------
    beta : array_like, shape (3*L,)
        ELM output weights
    model : GeoELM
        ELM model instance
    t_eval : array_like
        Time points for evaluation
    obs : array_like, shape (3, N_obs), optional
        Observed trig components
    t_obs : array_like, optional
        Observation time points
    station_eci : array_like, shape (3, N_obs), optional
        Station position in ECI frame
        
    Returns:
    --------
    r : ndarray, shape (3, N)
        Position vectors
    v : ndarray, shape (3, N)
        Velocity vectors
    a : ndarray, shape (3, N)
        Acceleration vectors
    physics_rms : float
        RMS of physics residuals
    measurement_rms : float or None
        RMS of measurement residuals in arcseconds (if observations provided)
    """
    # Compute trajectory
    r, v, a = model.r_v_a(t_eval, beta)
    
    # Compute physics residual RMS
    physics_rms = physics_residual_rms(beta, model, t_eval)
    
    # Compute measurement residual RMS if observations provided
    measurement_rms = None
    if obs is not None and t_obs is not None and station_eci is not None:
        measurement_rms = measurement_residual_rms(beta, model, obs, t_obs, station_eci)
    
    return r, v, a, physics_rms, measurement_rms


def test_solver():
    """
    Test function to verify solver implementation.
    """
    # Test parameters
    t0, t1 = 0.0, 3600.0  # 1 hour arc
    L = 32
    N_colloc = 40
    
    print("Testing physics-only fit...")
    beta, model, result = fit_physics_only(t0, t1, L=L, N_colloc=N_colloc)
    
    print(f"Optimization success: {result.success}")
    print(f"Function evaluations: {result.nfev}")
    print(f"Final cost: {result.cost:.6f}")
    print(f"Final residual RMS: {np.sqrt(result.cost/len(result.fun)):.6f}")
    
    # Evaluate solution
    t_eval = np.linspace(t0, t1, 100)
    r, v, a, physics_rms, _ = evaluate_solution(beta, model, t_eval)
    
    print(f"Position range: {np.min(np.linalg.norm(r, axis=0))/1000:.1f} - {np.max(np.linalg.norm(r, axis=0))/1000:.1f} km")
    print(f"Velocity range: {np.min(np.linalg.norm(v, axis=0)):.1f} - {np.max(np.linalg.norm(v, axis=0)):.1f} m/s")
    print(f"Physics residual RMS: {physics_rms:.6f}")
    
    return beta, model, result


if __name__ == "__main__":
    test_solver()
