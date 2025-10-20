"""
Enhanced loss function with position magnitude constraint for orbital elements ELM.
"""

import numpy as np
from .dynamics import accel_2body_J2
from .observe import trig_ra_dec

def residual_elements_enhanced(beta, model, t_colloc, lam_f, obs=None, t_obs=None, station_eci=None, lam_th=1.0, lam_r=1.0):
    """
    Enhanced residual function with position magnitude constraint.
    
    Parameters:
    -----------
    beta : ndarray, shape (12,)
        [a, e, i, Omega, omega, M0, beta_a, beta_e, beta_i, beta_Omega, beta_omega, beta_M]
    model : OrbitalElementsELM
        The orbital elements ELM model
    t_colloc : ndarray, shape (N_colloc,)
        Collocation time points
    lam_f : float
        Physics residual weight
    obs : ndarray, shape (3, N_obs), optional
        Trigonometric observation components
    t_obs : ndarray, shape (N_obs,), optional
        Observation time points
    station_eci : ndarray, shape (3, N_obs), optional
        Station positions in ECI frame
    lam_th : float
        Measurement residual weight
    lam_r : float
        Position magnitude residual weight
        
    Returns:
    --------
    residuals : ndarray
        Stacked residual vector with position constraint
    """
    
    # Physics residuals
    physics_residuals = []
    
    for t in t_colloc:
        # Get position, velocity, acceleration from ELM
        r, v, a_nn = model.r_v_a(t, beta)
        
        # Compute model acceleration
        a_mod = accel_2body_J2(r)
        
        # Physics residual
        physics_residuals.extend((a_nn - a_mod).tolist())
    
    physics_residuals = np.array(physics_residuals)
    
    # Position magnitude residuals
    position_residuals = []
    geo_altitude = 42164000  # meters
    
    for t in t_colloc:
        # Get position from ELM
        r, _, _ = model.r_v_a(t, beta)
        
        # Position magnitude residual
        r_mag = np.linalg.norm(r)
        position_residuals.append(r_mag - geo_altitude)
    
    position_residuals = np.array(position_residuals)
    
    if obs is None:
        return np.hstack([np.sqrt(lam_f) * physics_residuals, 
                         np.sqrt(lam_r) * position_residuals])
    
    # Measurement residuals
    measurement_residuals = []
    
    for i, t in enumerate(t_obs):
        # Get position from ELM
        r, _, _ = model.r_v_a(t, beta)
        
        # Compute topocentric vector
        r_topo = r - station_eci[:, i]
        
        # Convert to trigonometric components
        theta_nn = trig_ra_dec(r_topo)
        
        # Measurement residual
        measurement_residuals.extend((obs[:, i] - theta_nn).tolist())
    
    measurement_residuals = np.array(measurement_residuals)
    
    # Stack residuals
    return np.hstack([np.sqrt(lam_f) * physics_residuals, 
                     np.sqrt(lam_r) * position_residuals,
                     np.sqrt(lam_th) * measurement_residuals])


def evaluate_solution_elements_enhanced(beta, model, t_eval, obs=None, t_obs=None, station_eci=None):
    """
    Enhanced evaluation with position magnitude analysis.
    """
    
    # Evaluate solution
    r = np.zeros((3, len(t_eval)))
    v = np.zeros((3, len(t_eval)))
    a = np.zeros((3, len(t_eval)))
    
    for i, t in enumerate(t_eval):
        r[:, i], v[:, i], a[:, i] = model.r_v_a(t, beta)
    
    # Compute physics RMS
    physics_residuals = []
    for i, t in enumerate(t_eval):
        a_mod = accel_2body_J2(r[:, i])
        physics_residuals.extend((a[:, i] - a_mod).tolist())
    
    physics_rms = np.sqrt(np.mean(np.array(physics_residuals)**2))
    
    # Compute position magnitude RMS
    r_mag = np.linalg.norm(r, axis=0)
    geo_altitude = 42164000
    position_magnitude_rms = np.sqrt(np.mean((r_mag - geo_altitude)**2))/1000  # km
    
    # Compute measurement RMS
    if obs is not None and t_obs is not None and station_eci is not None:
        measurement_residuals = []
        for i, t in enumerate(t_obs):
            r_topo = r[:, i] - station_eci[:, i]
            theta_nn = trig_ra_dec(r_topo)
            measurement_residuals.extend((obs[:, i] - theta_nn).tolist())
        
        measurement_rms = np.sqrt(np.mean(np.array(measurement_residuals)**2))
    else:
        measurement_rms = 0.0
    
    return r, v, a, physics_rms, measurement_rms, position_magnitude_rms


def test_enhanced_residual():
    """Test the enhanced residual function."""
    print("Testing enhanced residual function...")
    
    from .elm_elements import OrbitalElementsELM
    
    # Create model
    t_phys = np.array([0.0, 7200.0])
    model = OrbitalElementsELM(L=8, t_phys=t_phys)
    
    # Test beta vector
    beta = np.array([
        42164000.0,  # a
        0.0,         # e
        0.0,         # i
        0.0,         # Omega
        0.0,         # omega
        0.0,         # M0
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # ELM weights
    ])
    
    # Test collocation points
    t_colloc = np.linspace(0.0, 7200.0, 10)
    
    # Test enhanced residual
    residual = residual_elements_enhanced(beta, model, t_colloc, lam_f=1.0, lam_r=1.0)
    print(f"Enhanced residual shape: {residual.shape}")
    print(f"Enhanced residual RMS: {np.sqrt(np.mean(residual**2)):.6f}")
    
    print("✓ Enhanced residual test completed")
    return True


if __name__ == "__main__":
    test_enhanced_residual()
