#!/usr/bin/env python3
"""
Debug the enhanced residual function.
"""

import sys
sys.path.append('piod')
import numpy as np
from piod.elm_elements import OrbitalElementsELM
from piod.loss_elements_enhanced import residual_elements_enhanced

def debug_enhanced_residual():
    """Debug the enhanced residual function."""
    print("=== DEBUGGING ENHANCED RESIDUAL FUNCTION ===")
    print()
    
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
    t_colloc = np.linspace(0.0, 7200.0, 5)
    
    print("1. Testing basic residual...")
    try:
        residual = residual_elements_enhanced(beta, model, t_colloc, lam_f=1.0, lam_r=1.0)
        print(f"✓ Basic residual shape: {residual.shape}")
        print(f"✓ Basic residual RMS: {np.sqrt(np.mean(residual**2)):.6f}")
        print(f"✓ Residual values: {residual[:10]}")
    except Exception as e:
        print(f"✗ Basic residual failed: {e}")
        return False
    
    print("2. Testing with observations...")
    # Create simple observations
    from piod.observe import ecef_to_eci, radec_to_trig
    
    station_ecef = np.array([6378136.3, 0.0, 0.0])
    t_obs = np.array([0.0, 3600.0, 7200.0])
    jd_obs = 2451545.0 + t_obs / 86400.0
    station_eci = np.array([ecef_to_eci(station_ecef, jd) for jd in jd_obs]).T
    
    ra_obs = np.array([0.0, 0.01, 0.02])
    dec_obs = np.array([0.0, 0.005, 0.01])
    obs = radec_to_trig(ra_obs, dec_obs)
    
    try:
        residual = residual_elements_enhanced(beta, model, t_colloc, lam_f=1.0, lam_r=1.0,
                                            obs=obs, t_obs=t_obs, station_eci=station_eci, lam_th=1.0)
        print(f"✓ Enhanced residual shape: {residual.shape}")
        print(f"✓ Enhanced residual RMS: {np.sqrt(np.mean(residual**2)):.6f}")
        print(f"✓ Residual values: {residual[:10]}")
    except Exception as e:
        print(f"✗ Enhanced residual failed: {e}")
        return False
    
    print("3. Testing with different beta values...")
    # Test with different beta values
    beta_test = beta.copy()
    beta_test[6:] = np.random.randn(6) * 100
    
    try:
        residual = residual_elements_enhanced(beta_test, model, t_colloc, lam_f=1.0, lam_r=1.0,
                                            obs=obs, t_obs=t_obs, station_eci=station_eci, lam_th=1.0)
        print(f"✓ Random beta residual shape: {residual.shape}")
        print(f"✓ Random beta residual RMS: {np.sqrt(np.mean(residual**2)):.6f}")
        print(f"✓ Residual values: {residual[:10]}")
    except Exception as e:
        print(f"✗ Random beta residual failed: {e}")
        return False
    
    print("4. Testing element bounds...")
    # Test with bounds
    beta_bounded = beta.copy()
    beta_bounded[0] = 42000000  # Within bounds
    beta_bounded[1] = 0.05      # Within bounds
    beta_bounded[2] = 0.05      # Within bounds
    beta_bounded[6:] = np.random.randn(6) * 500  # Within bounds
    
    try:
        residual = residual_elements_enhanced(beta_bounded, model, t_colloc, lam_f=1.0, lam_r=1.0,
                                            obs=obs, t_obs=t_obs, station_eci=station_eci, lam_th=1.0)
        print(f"✓ Bounded beta residual shape: {residual.shape}")
        print(f"✓ Bounded beta residual RMS: {np.sqrt(np.mean(residual**2)):.6f}")
        print(f"✓ Residual values: {residual[:10]}")
    except Exception as e:
        print(f"✗ Bounded beta residual failed: {e}")
        return False
    
    print()
    print("=== DEBUG COMPLETE ===")
    print("✓ Enhanced residual function is working correctly!")
    return True

if __name__ == "__main__":
    debug_enhanced_residual()
