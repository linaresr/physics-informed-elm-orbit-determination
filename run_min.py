#!/usr/bin/env python3
"""
Main script for physics-informed ELM orbit determination.
Demonstrates physics-only fitting and optional angle-aided fitting.
"""

import numpy as np
import sys
import os

# Add piod module to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'piod'))

from piod.solve import fit_elm, fit_physics_only, evaluate_solution
from piod.utils import create_time_grid, print_solution_summary, save_solution
from piod.observe import ecef_to_eci, radec_to_trig


def demo_physics_only():
    """
    Demonstrate physics-only ELM fitting.
    """
    print("="*60)
    print("PHYSICS-ONLY ELM FITTING DEMO")
    print("="*60)
    
    # Define time arc (12 hours)
    t0, t1 = 0.0, 12 * 3600.0  # seconds
    
    # ELM parameters
    L = 48          # hidden neurons
    N_colloc = 80   # collocation points
    lam_f = 10.0    # physics weight
    
    print(f"Time arc: {t0/3600:.1f} to {t1/3600:.1f} hours")
    print(f"Hidden neurons: {L}")
    print(f"Collocation points: {N_colloc}")
    print(f"Physics weight: {lam_f}")
    
    # Fit ELM
    print("\nFitting ELM...")
    beta, model, result = fit_physics_only(t0, t1, L=L, N_colloc=N_colloc, lam_f=lam_f)
    
    print(f"Optimization success: {result.success}")
    print(f"Function evaluations: {result.nfev}")
    print(f"Final cost: {result.cost:.6f}")
    
    # Evaluate solution
    t_eval = create_time_grid(t0, t1, 200, 'linear')
    r, v, a, physics_rms, _ = evaluate_solution(beta, model, t_eval)
    
    # Print summary
    print_solution_summary(beta, model, physics_rms)
    
    # Save solution
    save_solution('data/physics_only_solution.npz', beta, model, t_eval, r, v, a, physics_rms)
    
    return beta, model, t_eval, r, v, a


def demo_with_observations():
    """
    Demonstrate ELM fitting with angle observations.
    """
    print("\n" + "="*60)
    print("ANGLE-AIDED ELM FITTING DEMO")
    print("="*60)
    
    # Define time arc (6 hours)
    t0, t1 = 0.0, 6 * 3600.0  # seconds
    
    # ELM parameters
    L = 48
    N_colloc = 60
    lam_f = 10.0
    lam_th = 1.0
    
    # Create mock observations
    # Station at Greenwich (ECEF)
    station_ecef = np.array([6378136.3, 0.0, 0.0])  # m
    
    # Observation times (every 30 minutes)
    t_obs = np.arange(t0, t1, 30 * 60)  # every 30 minutes
    N_obs = len(t_obs)
    
    # Convert station to ECI at observation times
    jd_obs = 2451545.0 + t_obs / 86400.0  # J2000.0 + time in days
    station_eci = np.array([ecef_to_eci(station_ecef, jd) for jd in jd_obs]).T
    
    # Mock observations: GEO satellite with small perturbations
    # True RA/DEC (simplified)
    ra_true = np.linspace(0, 0.1, N_obs)  # Small RA change
    dec_true = np.linspace(0, 0.05, N_obs)  # Small DEC change
    
    # Add noise
    noise_level = 0.001  # radians (~3.4 arcmin)
    ra_obs = ra_true + np.random.normal(0, noise_level, N_obs)
    dec_obs = dec_true + np.random.normal(0, noise_level, N_obs)
    
    # Convert to trig components
    obs = radec_to_trig(ra_obs, dec_obs)
    
    print(f"Time arc: {t0/3600:.1f} to {t1/3600:.1f} hours")
    print(f"Observations: {N_obs} points")
    print(f"Station: Greenwich (ECEF)")
    print(f"Noise level: {noise_level*180/np.pi*3600:.1f} arcsec")
    
    # Fit ELM with observations (use optimal weights for better measurement residuals)
    print("\nFitting ELM with observations...")
    beta, model, result = fit_elm(t0, t1, L=L, N_colloc=N_colloc, lam_f=1.0,
                                obs=obs, t_obs=t_obs, station_eci=station_eci, lam_th=10.0)
    
    print(f"Optimization success: {result.success}")
    print(f"Function evaluations: {result.nfev}")
    print(f"Final cost: {result.cost:.6f}")
    
    # Evaluate solution
    t_eval = create_time_grid(t0, t1, 200, 'linear')
    r, v, a, physics_rms, measurement_rms = evaluate_solution(
        beta, model, t_eval, obs, t_obs, station_eci)
    
    # Print summary
    print_solution_summary(beta, model, physics_rms, measurement_rms)
    
    # Save solution
    save_solution('data/angle_aided_solution.npz', beta, model, t_eval, r, v, a, 
                 physics_rms, measurement_rms)
    
    return beta, model, t_eval, r, v, a


def main():
    """
    Main function demonstrating both physics-only and angle-aided fitting.
    """
    print("Physics-Informed ELM for Angle-Only Orbit Determination")
    print("Based on the paper: Physics-Informed Extreme Learning Machine")
    print()
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    try:
        # Run physics-only demo
        demo_physics_only()
        
        # Run angle-aided demo
        demo_with_observations()
        
        print("\n" + "="*60)
        print("DEMO COMPLETED SUCCESSFULLY")
        print("="*60)
        print("Check the 'data/' directory for saved solutions.")
        print("You can load them with:")
        print("  data = np.load('data/physics_only_solution.npz')")
        print("  data = np.load('data/angle_aided_solution.npz')")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
