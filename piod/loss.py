"""
Loss function module for physics-informed ELM.
Implements physics residuals and optional measurement residuals.
"""

import numpy as np
from .dynamics import accel_2body_J2
from .observe import trig_ra_dec


def residual(beta, model, t_colloc, lam_f,
             obs=None, t_obs=None, station_eci=None, lam_th=1.0):
    """
    Compute stacked residual vector for physics-informed ELM.
    
    Parameters:
    -----------
    beta : array_like, shape (3*L,)
        ELM output weights [βx; βy; βz]
    model : GeoELM
        ELM model instance
    t_colloc : array_like
        Collocation time points for physics residuals
    lam_f : float
        Weight for physics residuals
    obs : array_like, shape (3, N_obs), optional
        Observed trig components [sin(RA), cos(RA), sin(DEC)]
    t_obs : array_like, optional
        Observation time points
    station_eci : array_like, shape (3, N_obs), optional
        Station position in ECI frame at observation times
    lam_th : float
        Weight for measurement residuals
        
    Returns:
    --------
    residual : ndarray
        Stacked residual vector [√λ_f * L_f; √λ_th * L_th]
    """
    # Physics residuals: L_f = a_NN - a_model
    r, v, a_nn = model.r_v_a(t_colloc, beta)
    
    # Compute model acceleration at collocation points
    a_mod = np.apply_along_axis(accel_2body_J2, 0, r)
    
    # Physics residual: difference between ELM acceleration and model acceleration
    Lf = (a_nn - a_mod).ravel()
    
    if obs is None:
        # Physics-only case
        return np.sqrt(lam_f) * Lf
    
    # Measurement residuals (if observations provided)
    # Get ELM position at observation times
    r_obs, _, _ = model.r_v_a(t_obs, beta)
    
    # Compute topocentric vectors: ρ = r_sat - r_station
    r_topo = r_obs - station_eci
    
    # Convert to trig components
    theta_nn = np.apply_along_axis(trig_ra_dec, 0, r_topo)
    
    # Measurement residual: difference between observed and predicted angles
    Lth = (obs - theta_nn).ravel()
    
    # Stack residuals with weights
    return np.hstack([np.sqrt(lam_f) * Lf, np.sqrt(lam_th) * Lth])


def physics_residual_rms(beta, model, t_colloc):
    """
    Compute RMS of physics residuals for monitoring convergence.
    
    Parameters:
    -----------
    beta : array_like, shape (3*L,)
        ELM output weights
    model : GeoELM
        ELM model instance
    t_colloc : array_like
        Collocation time points
        
    Returns:
    --------
    rms : float
        RMS of physics residuals
    """
    r, v, a_nn = model.r_v_a(t_colloc, beta)
    a_mod = np.apply_along_axis(accel_2body_J2, 0, r)
    Lf = (a_nn - a_mod).ravel()
    return np.sqrt(np.mean(Lf**2))


def measurement_residual_rms(beta, model, obs, t_obs, station_eci):
    """
    Compute RMS of measurement residuals in arcseconds.
    
    Parameters:
    -----------
    beta : array_like, shape (3*L,)
        ELM output weights
    model : GeoELM
        ELM model instance
    obs : array_like, shape (3, N_obs)
        Observed trig components
    t_obs : array_like
        Observation time points
    station_eci : array_like, shape (3, N_obs)
        Station position in ECI frame
        
    Returns:
    --------
    rms_arcsec : float
        RMS of measurement residuals in arcseconds
    """
    r_obs, _, _ = model.r_v_a(t_obs, beta)
    r_topo = r_obs - station_eci
    theta_nn = np.apply_along_axis(trig_ra_dec, 0, r_topo)
    
    # Convert trig residuals back to angular residuals
    # For small residuals: δRA ≈ δ(sin(RA))/cos(RA), δDEC ≈ δ(sin(DEC))/cos(DEC)
    sin_ra_obs, cos_ra_obs, sin_dec_obs = obs
    sin_ra_nn, cos_ra_nn, sin_dec_nn = theta_nn
    
    # Avoid division by zero
    cos_ra_obs = np.maximum(np.abs(cos_ra_obs), 1e-8)
    cos_dec_obs = np.maximum(np.sqrt(1 - sin_dec_obs**2), 1e-8)
    
    dra = (sin_ra_nn - sin_ra_obs) / cos_ra_obs
    ddec = (sin_dec_nn - sin_dec_obs) / cos_dec_obs
    
    # Convert to arcseconds
    rms_ra = np.sqrt(np.mean(dra**2)) * 180 / np.pi * 3600
    rms_dec = np.sqrt(np.mean(ddec**2)) * 180 / np.pi * 3600
    
    return np.sqrt(rms_ra**2 + rms_dec**2)


def test_residual():
    """
    Test function to verify residual computation.
    """
    from .elm import GeoELM
    
    # Create test ELM
    L = 32
    t_phys = np.array([0.0, 3600.0])
    model = GeoELM(L, t_phys, seed=42)
    
    # Test collocation points
    t_colloc = np.linspace(0, 3600, 20)
    
    # Random output weights
    beta = np.random.randn(3 * L) * 0.1
    
    # Test physics-only residual
    res_physics = residual(beta, model, t_colloc, lam_f=1.0)
    print(f"Physics residual shape: {res_physics.shape}")
    print(f"Physics residual RMS: {np.sqrt(np.mean(res_physics**2)):.6f}")
    
    # Test with mock observations
    t_obs = np.array([1800.0, 2700.0])  # 2 observation times
    station_eci = np.array([[6378136.3, 0.0, 0.0],
                           [6378136.3, 0.0, 0.0]]).T  # Station at Greenwich
    
    # Mock observations (trig components)
    obs = np.array([[0.0, 1.0, 0.0],    # RA=0, DEC=0
                    [1.0, 0.0, 0.0]]).T  # RA=π/2, DEC=0
    
    res_full = residual(beta, model, t_colloc, lam_f=1.0,
                       obs=obs, t_obs=t_obs, station_eci=station_eci, lam_th=1.0)
    print(f"Full residual shape: {res_full.shape}")
    print(f"Full residual RMS: {np.sqrt(np.mean(res_full**2)):.6f}")
    
    return res_physics, res_full


if __name__ == "__main__":
    test_residual()
