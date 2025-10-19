#!/usr/bin/env python3
"""
Systematic optimization study to find the best ELM parameters.
Focuses on minimizing measurement and position residuals.
"""

import sys
sys.path.append('piod')
import numpy as np
import json
from piod.solve import fit_elm, evaluate_solution
from piod.observe import ecef_to_eci, radec_to_trig, vec_to_radec
from piod.utils import propagate_state
from piod.dynamics import eom
from scipy.integrate import solve_ivp

def generate_true_orbit():
    """Generate a realistic GEO orbit using numerical integration."""
    # Initial conditions for GEO orbit
    r0 = np.array([42164000.0, 0.0, 0.0])  # GEO altitude
    v0 = np.array([0.0, 3074.0, 0.0])     # Circular orbit velocity
    
    # Time span
    t0, t1 = 0.0, 4 * 3600.0  # 4 hours
    
    # Integrate using scipy
    sol = solve_ivp(eom, [t0, t1], np.hstack([r0, v0]), 
                   t_eval=np.linspace(t0, t1, 1000), rtol=1e-8, atol=1e-8)
    
    if not sol.success:
        print(f"Integration failed: {sol.message}")
        return None
    
    return sol.t, sol.y[:3], sol.y[3:]

def create_observations_from_true_orbit(t_true, r_true, noise_level=0.001):
    """Create realistic observations from true orbit."""
    t0, t1 = t_true[0], t_true[-1]
    
    # Create observations
    station_ecef = np.array([6378136.3, 0.0, 0.0])  # Greenwich
    t_obs = np.linspace(t0, t1, 10)  # 10 observations
    
    # Find closest true orbit points to observation times
    obs_indices = []
    for t in t_obs:
        idx = np.argmin(np.abs(t_true - t))
        obs_indices.append(idx)
    
    r_obs_true = r_true[:, obs_indices]
    jd_obs = 2451545.0 + t_obs / 86400.0
    station_eci = np.array([ecef_to_eci(station_ecef, jd) for jd in jd_obs]).T
    
    # Convert true positions to observed angles
    r_topo_true = r_obs_true - station_eci
    obs_true = np.array([vec_to_radec(r_topo_true[:, i]) for i in range(len(t_obs))])
    ra_obs_true, dec_obs_true = obs_true[:, 0], obs_true[:, 1]
    
    # Add realistic noise to observations
    ra_obs = ra_obs_true + np.random.normal(0, noise_level, len(t_obs))
    dec_obs = dec_obs_true + np.random.normal(0, noise_level, len(t_obs))
    obs = radec_to_trig(ra_obs, dec_obs)
    
    return t_obs, obs, station_eci, ra_obs_true, dec_obs_true

def evaluate_elm_performance(beta, model, t_eval, t_true, r_true, v_true, 
                           obs, t_obs, station_eci):
    """Evaluate ELM performance comprehensively."""
    # Get ELM predictions
    r_elm, v_elm, a_elm, physics_rms, measurement_rms = evaluate_solution(
        beta, model, t_eval, obs, t_obs, station_eci)
    
    # Interpolate true orbit to ELM evaluation times
    r_true_interp = np.zeros_like(r_elm)
    v_true_interp = np.zeros_like(v_elm)
    for i in range(3):
        r_true_interp[i] = np.interp(t_eval, t_true, r_true[i])
        v_true_interp[i] = np.interp(t_eval, t_true, v_true[i])
    
    # Calculate errors
    r_error = np.linalg.norm(r_elm - r_true_interp, axis=0)
    v_error = np.linalg.norm(v_elm - v_true_interp, axis=0)
    
    # Calculate measurement residuals at observation times
    r_elm_obs, _, _ = model.r_v_a(t_obs, beta)
    r_topo_elm = r_elm_obs - station_eci
    obs_elm = np.array([vec_to_radec(r_topo_elm[:, i]) for i in range(len(t_obs))])
    
    # Convert to angular residuals
    ra_residual = (obs[:, 0] - obs_elm[:, 0]) * 180/np.pi * 3600  # arcsec
    dec_residual = (obs[:, 1] - obs_elm[:, 1]) * 180/np.pi * 3600  # arcsec
    
    return {
        'position_rms_error': np.sqrt(np.mean(r_error**2)),
        'velocity_rms_error': np.sqrt(np.mean(v_error**2)),
        'position_max_error': np.max(r_error),
        'velocity_max_error': np.max(v_error),
        'measurement_rms': measurement_rms,
        'ra_rms_residual': np.sqrt(np.mean(ra_residual**2)),
        'dec_rms_residual': np.sqrt(np.mean(dec_residual**2)),
        'physics_rms': physics_rms,
        'position_range': (np.min(np.linalg.norm(r_elm, axis=0))/1000, 
                         np.max(np.linalg.norm(r_elm, axis=0))/1000)
    }

def systematic_optimization_study():
    """Run systematic optimization study."""
    print("=== SYSTEMATIC ELM OPTIMIZATION STUDY ===")
    print("Focus: Minimizing measurement and position residuals")
    print()
    
    # Generate true orbit
    print("1. Generating true GEO orbit...")
    t_true, r_true, v_true = generate_true_orbit()
    if t_true is None:
        return
    
    print(f"✓ Generated true orbit: {len(t_true)} points over {t_true[-1]/3600:.1f} hours")
    
    # Create observations
    print("2. Creating observations from true orbit...")
    t_obs, obs, station_eci, ra_obs_true, dec_obs_true = create_observations_from_true_orbit(
        t_true, r_true, noise_level=0.0005)  # Reduced noise for better results
    
    print(f"✓ Created {len(t_obs)} observations with {0.0005*180/np.pi*3600:.1f} arcsec noise")
    
    # Define parameter space to explore
    print("3. Defining parameter space...")
    
    # Network sizes to test
    network_sizes = [8, 12, 16, 20, 24, 28, 32, 40, 48]
    
    # Weight combinations to test
    weight_combinations = [
        (1.0, 1.0),    # Equal weights
        (1.0, 10.0),   # Favor measurements
        (1.0, 100.0),  # Strong measurements
        (1.0, 1000.0), # Very strong measurements
        (10.0, 1.0),   # Favor physics
        (100.0, 1.0),  # Strong physics
    ]
    
    # Collocation point strategies
    collocation_strategies = ['proportional', 'fixed_low', 'fixed_high']
    
    print(f"   Network sizes: {network_sizes}")
    print(f"   Weight combinations: {len(weight_combinations)}")
    print(f"   Collocation strategies: {collocation_strategies}")
    print(f"   Total configurations: {len(network_sizes) * len(weight_combinations) * len(collocation_strategies)}")
    print()
    
    # Run optimization study
    print("4. Running optimization study...")
    results = []
    total_configs = len(network_sizes) * len(weight_combinations) * len(collocation_strategies)
    config_count = 0
    
    for L in network_sizes:
        for lam_f, lam_th in weight_combinations:
            for strategy in collocation_strategies:
                config_count += 1
                
                # Determine collocation points
                if strategy == 'proportional':
                    N_colloc = max(L, 20)
                elif strategy == 'fixed_low':
                    N_colloc = 30
                else:  # fixed_high
                    N_colloc = 60
                
                print(f"   [{config_count}/{total_configs}] L={L}, λ_f={lam_f}, λ_th={lam_th}, N_colloc={N_colloc}...", end=" ")
                
                try:
                    # Train ELM
                    beta, model, result = fit_elm(t_true[0], t_true[-1], L=L, N_colloc=N_colloc,
                                                obs=obs, t_obs=t_obs, station_eci=station_eci,
                                                lam_f=lam_f, lam_th=lam_th)
                    
                    if result.success:
                        # Evaluate performance
                        t_eval = np.linspace(t_true[0], t_true[-1], 200)
                        performance = evaluate_elm_performance(
                            beta, model, t_eval, t_true, r_true, v_true, 
                            obs, t_obs, station_eci)
                        
                        # Store results
                        config_result = {
                            'L': L,
                            'lam_f': lam_f,
                            'lam_th': lam_th,
                            'N_colloc': N_colloc,
                            'strategy': strategy,
                            'success': True,
                            'nfev': result.nfev,
                            'cost': result.cost,
                            **performance
                        }
                        
                        results.append(config_result)
                        
                        print(f"✓ Success")
                        print(f"      Position RMS: {performance['position_rms_error']/1000:.1f} km")
                        print(f"      Measurement RMS: {performance['measurement_rms']:.1f} arcsec")
                        print(f"      Physics RMS: {performance['physics_rms']:.6f}")
                        
                    else:
                        print(f"✗ Failed: {result.message}")
                        results.append({
                            'L': L,
                            'lam_f': lam_f,
                            'lam_th': lam_th,
                            'N_colloc': N_colloc,
                            'strategy': strategy,
                            'success': False,
                            'nfev': result.nfev,
                            'cost': result.cost
                        })
                        
                except Exception as e:
                    print(f"✗ Error: {e}")
                    results.append({
                        'L': L,
                        'lam_f': lam_f,
                        'lam_th': lam_th,
                        'N_colloc': N_colloc,
                        'strategy': strategy,
                        'success': False,
                        'error': str(e)
                    })
    
    print()
    print("5. Analyzing results...")
    
    # Filter successful results
    successful_results = [r for r in results if r.get('success', False)]
    
    if not successful_results:
        print("✗ No successful configurations found!")
        return
    
    print(f"✓ {len(successful_results)} successful configurations out of {len(results)}")
    
    # Find best configurations
    best_position = min(successful_results, key=lambda x: x['position_rms_error'])
    best_measurement = min(successful_results, key=lambda x: x['measurement_rms'])
    best_combined = min(successful_results, key=lambda x: x['position_rms_error'] + x['measurement_rms']/1000)
    
    print()
    print("🏆 BEST CONFIGURATIONS:")
    print()
    print(f"Best Position Accuracy:")
    print(f"  L={best_position['L']}, λ_f={best_position['lam_f']}, λ_th={best_position['lam_th']}, N_colloc={best_position['N_colloc']}")
    print(f"  Position RMS: {best_position['position_rms_error']/1000:.1f} km")
    print(f"  Measurement RMS: {best_position['measurement_rms']:.1f} arcsec")
    print(f"  Physics RMS: {best_position['physics_rms']:.6f}")
    print()
    print(f"Best Measurement Accuracy:")
    print(f"  L={best_measurement['L']}, λ_f={best_measurement['lam_f']}, λ_th={best_measurement['lam_th']}, N_colloc={best_measurement['N_colloc']}")
    print(f"  Position RMS: {best_measurement['position_rms_error']/1000:.1f} km")
    print(f"  Measurement RMS: {best_measurement['measurement_rms']:.1f} arcsec")
    print(f"  Physics RMS: {best_measurement['physics_rms']:.6f}")
    print()
    print(f"Best Combined Performance:")
    print(f"  L={best_combined['L']}, λ_f={best_combined['lam_f']}, λ_th={best_combined['lam_th']}, N_colloc={best_combined['N_colloc']}")
    print(f"  Position RMS: {best_combined['position_rms_error']/1000:.1f} km")
    print(f"  Measurement RMS: {best_combined['measurement_rms']:.1f} arcsec")
    print(f"  Physics RMS: {best_combined['physics_rms']:.6f}")
    
    # Save results
    print()
    print("6. Saving results...")
    
    optimization_results = {
        'all_results': results,
        'successful_results': successful_results,
        'best_position': best_position,
        'best_measurement': best_measurement,
        'best_combined': best_combined,
        'metadata': {
            'total_configurations': len(results),
            'successful_configurations': len(successful_results),
            'true_orbit_duration_hours': t_true[-1]/3600,
            'observations_count': len(t_obs),
            'noise_level_arcsec': 0.0005*180/np.pi*3600
        }
    }
    
    with open('data/optimization_study.json', 'w') as f:
        json.dump(optimization_results, f, indent=2)
    
    print("✓ Results saved to: data/optimization_study.json")
    
    # Print summary statistics
    print()
    print("📊 SUMMARY STATISTICS:")
    print(f"Position RMS Error Range: {min(r['position_rms_error'] for r in successful_results)/1000:.1f} - {max(r['position_rms_error'] for r in successful_results)/1000:.1f} km")
    print(f"Measurement RMS Range: {min(r['measurement_rms'] for r in successful_results):.1f} - {max(r['measurement_rms'] for r in successful_results):.1f} arcsec")
    print(f"Physics RMS Range: {min(r['physics_rms'] for r in successful_results):.6f} - {max(r['physics_rms'] for r in successful_results):.6f}")
    
    print()
    print("=== OPTIMIZATION STUDY COMPLETE ===")
    print("Use the best configurations for improved performance!")

if __name__ == "__main__":
    systematic_optimization_study()
