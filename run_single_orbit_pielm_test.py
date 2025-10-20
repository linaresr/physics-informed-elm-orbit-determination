#!/usr/bin/env python3
"""
Comprehensive test of the single orbit ELM approach with PIELM-compliant parameters.
This script runs the single orbit case and generates updated performance metrics.
"""

import sys
sys.path.append('piod')
import numpy as np
import matplotlib.pyplot as plt
from piod.elm import GeoELM
from piod.loss import residual, physics_residual_rms, measurement_residual_rms
from piod.solve import fit_elm, evaluate_solution
from piod.dynamics import accel_2body_J2, eom
from piod.observe import radec_to_trig, trig_ra_dec, ecef_to_eci, vec_to_radec
from scipy.integrate import solve_ivp
import json
import os
from datetime import datetime


def run_single_orbit_test():
    """
    Run comprehensive single orbit ELM test with PIELM-compliant parameters.
    """
    print("=== SINGLE ORBIT ELM TEST (PIELM-COMPLIANT) ===")
    print()
    
    # Test parameters (verified PIELM-compliant)
    t0, t1 = 0.0, 7200.0  # 2 hour arc
    L = 24  # Hidden neurons
    N_colloc = 80  # Collocation points
    lam_f = 1.0  # Physics weight
    lam_th = 10000.0  # Measurement weight
    n_obs = 20  # Number of observations
    
    print(f"Test Parameters:")
    print(f"  • Time arc: {t0} to {t1} seconds ({t1/3600:.1f} hours)")
    print(f"  • Hidden neurons: L = {L}")
    print(f"  • Collocation points: N_colloc = {N_colloc}")
    print(f"  • Physics weight: λ_f = {lam_f}")
    print(f"  • Measurement weight: λ_th = {lam_th}")
    print(f"  • Observations: {n_obs} points")
    print()
    
    # Generate reference orbit
    print("1. Generating reference orbit...")
    r0 = np.array([42164000.0, 0.0, 0.0])  # GEO-like position
    v0 = np.array([0.0, 3074.0, 0.0])      # Circular velocity
    
    # Integrate true orbit
    sol = solve_ivp(eom, [t0, t1], np.hstack([r0, v0]), 
                    t_eval=np.linspace(t0, t1, 200),
                    rtol=1e-8, atol=1e-8)
    
    if not sol.success:
        print("✗ Failed to generate reference orbit")
        return None
    
    r_true = sol.y[:3]
    t_true = sol.t
    print(f"✓ Generated reference orbit with {len(t_true)} points")
    print(f"  Position range: {np.min(np.linalg.norm(r_true, axis=0))/1000:.1f} - {np.max(np.linalg.norm(r_true, axis=0))/1000:.1f} km")
    print()
    
    # Generate observations
    print("2. Generating observations...")
    station_ecef = np.array([6378136.3, 0.0, 0.0])  # Greenwich
    jd_start = 2451545.0  # J2000.0
    
    # Select observation times evenly spaced
    obs_indices = np.linspace(0, len(t_true)-1, n_obs, dtype=int)
    t_obs = t_true[obs_indices]
    r_obs_true = r_true[:, obs_indices]
    
    jd_obs = jd_start + t_obs / 86400.0
    station_eci = np.array([ecef_to_eci(station_ecef, jd) for jd in jd_obs]).T
    
    # Generate observations with realistic noise
    noise_level = 0.0001  # ~0.02 arcsec
    true_ra, true_dec = [], []
    for i in range(len(t_obs)):
        r_topo = r_obs_true[:, i] - station_eci[:, i]
        ra, dec = vec_to_radec(r_topo)
        true_ra.append(ra)
        true_dec.append(dec)
    
    true_ra = np.array(true_ra)
    true_dec = np.array(true_dec)
    
    # Add noise
    np.random.seed(42)  # For reproducibility
    ra_noisy = true_ra + np.random.normal(0, noise_level, len(true_ra))
    dec_noisy = true_dec + np.random.normal(0, noise_level, len(true_dec))
    
    obs = radec_to_trig(ra_noisy, dec_noisy)
    
    print(f"✓ Generated {len(t_obs)} observations")
    print(f"  Noise level: {noise_level*180/np.pi*3600:.2f} arcsec")
    print(f"  Observation span: {t_obs[0]/3600:.1f} to {t_obs[-1]/3600:.1f} hours")
    print()
    
    # Train ELM
    print("3. Training ELM...")
    try:
        beta, model, result = fit_elm(t0, t1, L=L, N_colloc=N_colloc,
                                     obs=obs, t_obs=t_obs,
                                     station_eci=station_eci,
                                     lam_f=lam_f, lam_th=lam_th, seed=42)
        
        print(f"✓ Training completed successfully")
        print(f"  Success: {result.success}")
        print(f"  Function evaluations: {result.nfev}")
        print(f"  Final cost: {result.cost:.6f}")
        print(f"  Convergence: {'✓' if result.success else '✗'}")
        print()
        
    except Exception as e:
        print(f"✗ Training failed: {e}")
        return None
    
    # Evaluate performance
    print("4. Evaluating performance...")
    t_eval = np.linspace(t0, t1, 200)
    r_eval, v_eval, a_eval, physics_rms, measurement_rms = evaluate_solution(
        beta, model, t_eval, obs, t_obs, station_eci)
    
    # Calculate position error
    r_true_interp = np.zeros_like(r_eval)
    for i in range(3):
        r_true_interp[i] = np.interp(t_eval, t_true, r_true[i])
    
    r_error = np.linalg.norm(r_eval - r_true_interp, axis=0)
    position_error_rms = np.sqrt(np.mean(r_error**2))/1000  # Convert to km
    
    # Calculate measurement error
    measurement_residuals = []
    for i, t in enumerate(t_obs):
        r_obs, _, _ = model.r_v_a(t, beta)
        r_topo = r_obs - station_eci[:, i]
        theta_nn = trig_ra_dec(r_topo)
        residual = obs[:, i] - theta_nn
        measurement_residuals.extend(residual.tolist())
    
    measurement_residuals = np.array(measurement_residuals)
    measurement_rms_detailed = np.sqrt(np.mean(measurement_residuals**2)) * 180/np.pi * 3600
    
    print(f"✓ Performance evaluation completed")
    print(f"  Position Error RMS: {position_error_rms:.1f} km")
    print(f"  Measurement RMS: {measurement_rms:.2f} arcsec")
    print(f"  Physics RMS: {physics_rms:.6f}")
    print()
    
    # Check target achievement
    position_target = position_error_rms < 10.0
    measurement_target = measurement_rms < 5.0
    physics_target = physics_rms < 0.01
    
    print("5. Target Achievement:")
    print(f"  Position target (<10 km): {'✓ ACHIEVED' if position_target else '✗ NOT ACHIEVED'} ({position_error_rms:.1f} km)")
    print(f"  Measurement target (<5 arcsec): {'✓ ACHIEVED' if measurement_target else '✗ NOT ACHIEVED'} ({measurement_rms:.2f} arcsec)")
    print(f"  Physics target (<0.01): {'✓ ACHIEVED' if physics_target else '✗ NOT ACHIEVED'} ({physics_rms:.6f})")
    print()
    
    # Create results summary
    results = {
        'method': 'Single-Orbit Cartesian ELM (PIELM-Compliant)',
        'parameters': {
            'L': L,
            'N_colloc': N_colloc,
            'lam_f': lam_f,
            'lam_th': lam_th,
            'n_obs': n_obs,
            'time_arc_hours': t1/3600
        },
        'performance': {
            'position_error_rms_km': position_error_rms,
            'measurement_rms_arcsec': measurement_rms,
            'physics_rms': physics_rms,
            'position_target_achieved': position_target,
            'measurement_target_achieved': measurement_target,
            'physics_target_achieved': physics_target
        },
        'optimization': {
            'success': result.success,
            'nfev': result.nfev,
            'cost': result.cost
        },
        'trajectory': {
            'r_eval': r_eval,
            'v_eval': v_eval,
            'a_eval': a_eval,
            't_eval': t_eval,
            'r_true': r_true_interp
        },
        'observations': {
            't_obs': t_obs,
            'obs': obs,
            'station_eci': station_eci,
            'noise_level_arcsec': noise_level*180/np.pi*3600
        },
        'model': model,
        'beta': beta
    }
    
    return results


def create_performance_plot(results):
    """
    Create a simple performance visualization.
    """
    print("6. Creating performance visualization...")
    
    # Create results directory
    os.makedirs("results/single_orbit_pielm", exist_ok=True)
    
    # Extract data
    r_eval = results['trajectory']['r_eval']
    r_true = results['trajectory']['r_true']
    t_eval = results['trajectory']['t_eval']
    
    # Calculate position error over time
    r_error = np.linalg.norm(r_eval - r_true, axis=0)
    
    # Create simple plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Position error over time
    ax1.plot(t_eval/3600, r_error/1000, 'r-', linewidth=2)
    ax1.set_xlabel('Time (hours)')
    ax1.set_ylabel('Position Error (km)')
    ax1.set_title('Position Error Over Time')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=10, color='g', linestyle='--', alpha=0.7, label='Target (10 km)')
    ax1.legend()
    
    # Plot 2: Performance summary
    ax2.axis('off')
    
    # Create performance summary text
    perf_text = f"""
PERFORMANCE SUMMARY
==================
Method: Single-Orbit Cartesian ELM (PIELM-Compliant)

Parameters:
• Hidden neurons: L = {results['parameters']['L']}
• Collocation points: N_colloc = {results['parameters']['N_colloc']}
• Physics weight: λ_f = {results['parameters']['lam_f']}
• Measurement weight: λ_th = {results['parameters']['lam_th']}
• Observations: {results['parameters']['n_obs']} points
• Time arc: {results['parameters']['time_arc_hours']:.1f} hours

Performance:
• Position Error RMS: {results['performance']['position_error_rms_km']:.1f} km
• Measurement RMS: {results['performance']['measurement_rms_arcsec']:.2f} arcsec
• Physics RMS: {results['performance']['physics_rms']:.6f}

Target Achievement:
• Position (<10 km): {'✓' if results['performance']['position_target_achieved'] else '✗'}
• Measurement (<5 arcsec): {'✓' if results['performance']['measurement_target_achieved'] else '✗'}
• Physics (<0.01): {'✓' if results['performance']['physics_target_achieved'] else '✗'}

Optimization:
• Success: {'✓' if results['optimization']['success'] else '✗'}
• Function evaluations: {results['optimization']['nfev']}
• Final cost: {results['optimization']['cost']:.6f}
"""
    
    ax2.text(0.05, 0.95, perf_text, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig("results/single_orbit_pielm/single_orbit_performance.png", 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Performance visualization saved")


def save_results(results):
    """
    Save detailed results to JSON file.
    """
    print("7. Saving detailed results...")
    
    # Create a copy without numpy arrays and convert numpy types for JSON serialization
    results_save = {
        'method': results['method'],
        'parameters': results['parameters'],
        'performance': {
            'position_error_rms_km': float(results['performance']['position_error_rms_km']),
            'measurement_rms_arcsec': float(results['performance']['measurement_rms_arcsec']),
            'physics_rms': float(results['performance']['physics_rms']),
            'position_target_achieved': bool(results['performance']['position_target_achieved']),
            'measurement_target_achieved': bool(results['performance']['measurement_target_achieved']),
            'physics_target_achieved': bool(results['performance']['physics_target_achieved'])
        },
        'optimization': {
            'success': bool(results['optimization']['success']),
            'nfev': int(results['optimization']['nfev']),
            'cost': float(results['optimization']['cost'])
        },
        'trajectory': {
            'r_eval': results['trajectory']['r_eval'].tolist(),
            'v_eval': results['trajectory']['v_eval'].tolist(),
            'a_eval': results['trajectory']['a_eval'].tolist(),
            't_eval': results['trajectory']['t_eval'].tolist(),
            'r_true': results['trajectory']['r_true'].tolist()
        },
        'observations': {
            't_obs': results['observations']['t_obs'].tolist(),
            'obs': results['observations']['obs'].tolist(),
            'station_eci': results['observations']['station_eci'].tolist(),
            'noise_level_arcsec': float(results['observations']['noise_level_arcsec'])
        },
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'test_type': 'single_orbit_pielm_compliant',
            'version': '1.0'
        }
    }
    
    with open("results/single_orbit_pielm/single_orbit_results.json", 'w') as f:
        json.dump(results_save, f, indent=2)
    
    print("✓ Detailed results saved")


def main():
    """
    Main function to run single orbit test and generate updated results.
    """
    print("=== SINGLE ORBIT ELM TEST (PIELM-COMPLIANT) ===")
    print("Running comprehensive test with verified parameters")
    print()
    
    # Run test
    results = run_single_orbit_test()
    
    if results is None:
        print("✗ Test failed")
        return
    
    # Create visualization
    create_performance_plot(results)
    
    # Save results
    save_results(results)
    
    print()
    print("=== TEST COMPLETE ===")
    print("📁 Results saved in: results/single_orbit_pielm/")
    print("📊 Generated files:")
    print("  • single_orbit_performance.png - Performance visualization")
    print("  • single_orbit_results.json - Detailed results")
    print()
    
    # Final summary
    print("🎯 FINAL RESULTS:")
    print(f"  Position Error RMS: {results['performance']['position_error_rms_km']:.1f} km")
    print(f"  Measurement RMS: {results['performance']['measurement_rms_arcsec']:.2f} arcsec")
    print(f"  Physics RMS: {results['performance']['physics_rms']:.6f}")
    print()
    
    targets_achieved = sum([
        results['performance']['position_target_achieved'],
        results['performance']['measurement_target_achieved'],
        results['performance']['physics_target_achieved']
    ])
    
    print(f"Targets Achieved: {targets_achieved}/3")
    if targets_achieved == 3:
        print("🎉 ALL TARGETS ACHIEVED!")
    elif targets_achieved >= 2:
        print("🎯 MOST TARGETS ACHIEVED!")
    else:
        print("⚠️ Some targets need improvement")
    
    return results


if __name__ == "__main__":
    main()
