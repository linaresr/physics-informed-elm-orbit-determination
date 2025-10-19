#!/usr/bin/env python3
"""
Training data collection script with progress indication.
Saves all training results for later plotting.
"""

import sys
sys.path.append('piod')
import numpy as np
import json
from piod.solve import fit_elm, evaluate_solution
from piod.observe import ecef_to_eci, radec_to_trig, vec_to_radec
from piod.utils import create_time_grid

def train_and_save_data():
    """Train ELM with different configurations and save results."""
    
    print("=== ELM TRAINING DATA COLLECTION ===")
    print("This will train multiple ELM configurations and save results for plotting.")
    print()
    
    # Configuration 1: Learning curve analysis
    print("1. Learning Curve Analysis (6 configurations)")
    print("   Testing different network sizes...")
    
    network_sizes = [8, 12, 16, 20, 24, 28]
    learning_curve_results = []
    
    for i, L in enumerate(network_sizes):
        print(f"   [{i+1}/6] Testing L={L}...", end=" ")
        
        # Use proportional collocation points (smaller for faster training)
        N_colloc = max(L, 20)
        
        # Create observations (shorter arc for faster training)
        t0, t1 = 0.0, 2 * 3600.0  # 2 hour arc
        station_ecef = np.array([6378136.3, 0.0, 0.0])
        t_obs = np.linspace(t0, t1, 6)  # Fewer observations
        jd_obs = 2451545.0 + t_obs / 86400.0
        station_eci = np.array([ecef_to_eci(station_ecef, jd) for jd in jd_obs]).T
        
        ra_obs = np.linspace(0.0, 0.06, len(t_obs))
        dec_obs = np.linspace(0.0, 0.04, len(t_obs))
        obs = radec_to_trig(ra_obs, dec_obs)
        
        try:
            beta, model, result = fit_elm(t0, t1, L=L, N_colloc=N_colloc,
                                        obs=obs, t_obs=t_obs, station_eci=station_eci,
                                        lam_f=1.0, lam_th=10.0)
            
            if result.success:
                t_eval = np.linspace(t0, t1, 100)
                r, v, a, physics_rms, measurement_rms = evaluate_solution(
                    beta, model, t_eval, obs, t_obs, station_eci)
                
                learning_curve_results.append({
                    'L': L,
                    'N_colloc': N_colloc,
                    'success': True,
                    'nfev': result.nfev,
                    'cost': result.cost,
                    'physics_rms': physics_rms,
                    'measurement_rms': measurement_rms,
                    'position_range': (np.min(np.linalg.norm(r, axis=0))/1000, 
                                     np.max(np.linalg.norm(r, axis=0))/1000)
                })
                print(f"✓ Success (eval={result.nfev}, meas_rms={measurement_rms:.1f} arcsec)")
            else:
                learning_curve_results.append({
                    'L': L,
                    'N_colloc': N_colloc,
                    'success': False,
                    'nfev': result.nfev,
                    'cost': result.cost,
                    'physics_rms': np.nan,
                    'measurement_rms': np.nan,
                    'position_range': (np.nan, np.nan)
                })
                print(f"✗ Failed: {result.message}")
        except Exception as e:
            print(f"✗ Error: {e}")
            learning_curve_results.append({
                'L': L,
                'N_colloc': N_colloc,
                'success': False,
                'nfev': 0,
                'cost': np.nan,
                'physics_rms': np.nan,
                'measurement_rms': np.nan,
                'position_range': (np.nan, np.nan)
            })
    
    print("   Learning curve analysis complete!")
    print()
    
    # Configuration 2: Weight comparison analysis
    print("2. Weight Comparison Analysis (2 configurations)")
    print("   Testing different weight combinations...")
    
    weight_combinations = [
        (10.0, 1.0, 'Original (Poor)'),
        (1.0, 10.0, 'Optimal (Good)'),
    ]
    
    weight_comparison_results = []
    t0, t1 = 0.0, 4 * 3600.0  # 4 hour arc
    L = 40
    N_colloc = 50
    
    # Create consistent observations
    station_ecef = np.array([6378136.3, 0.0, 0.0])
    t_obs = np.linspace(t0, t1, 8)
    jd_obs = 2451545.0 + t_obs / 86400.0
    station_eci = np.array([ecef_to_eci(station_ecef, jd) for jd in jd_obs]).T
    
    ra_obs = np.linspace(0.0, 0.06, len(t_obs))
    dec_obs = np.linspace(0.0, 0.04, len(t_obs))
    obs = radec_to_trig(ra_obs, dec_obs)
    
    for i, (lam_f, lam_th, label) in enumerate(weight_combinations):
        print(f"   [{i+1}/2] Testing {label}: λ_f={lam_f}, λ_th={lam_th}...", end=" ")
        
        try:
            beta, model, result = fit_elm(t0, t1, L=L, N_colloc=N_colloc,
                                        obs=obs, t_obs=t_obs, station_eci=station_eci,
                                        lam_f=lam_f, lam_th=lam_th)
            
            if result.success:
                t_eval = np.linspace(t0, t1, 100)
                r, v, a, physics_rms, measurement_rms = evaluate_solution(
                    beta, model, t_eval, obs, t_obs, station_eci)
                
                weight_comparison_results.append({
                    'label': label,
                    'lam_f': lam_f,
                    'lam_th': lam_th,
                    'success': True,
                    'nfev': result.nfev,
                    'cost': result.cost,
                    'physics_rms': physics_rms,
                    'measurement_rms': measurement_rms,
                    'position_range': (np.min(np.linalg.norm(r, axis=0))/1000, 
                                     np.max(np.linalg.norm(r, axis=0))/1000),
                    'r': r.tolist(),  # Convert to list for JSON serialization
                    't_eval': t_eval.tolist()
                })
                print(f"✓ Success (eval={result.nfev}, meas_rms={measurement_rms:.1f} arcsec)")
            else:
                weight_comparison_results.append({
                    'label': label,
                    'lam_f': lam_f,
                    'lam_th': lam_th,
                    'success': False,
                    'nfev': result.nfev,
                    'cost': result.cost,
                    'physics_rms': np.nan,
                    'measurement_rms': np.nan,
                    'position_range': (np.nan, np.nan),
                    'r': None,
                    't_eval': None
                })
                print(f"✗ Failed: {result.message}")
        except Exception as e:
            print(f"✗ Error: {e}")
            weight_comparison_results.append({
                'label': label,
                'lam_f': lam_f,
                'lam_th': lam_th,
                'success': False,
                'nfev': 0,
                'cost': np.nan,
                'physics_rms': np.nan,
                'measurement_rms': np.nan,
                'position_range': (np.nan, np.nan),
                'r': None,
                't_eval': None
            })
    
    print("   Weight comparison analysis complete!")
    print()
    
    # Configuration 3: Detailed training analysis
    print("3. Detailed Training Analysis (1 configuration)")
    print("   Training with optimal parameters for detailed analysis...")
    
    # Use optimal parameters
    t0, t1 = 0.0, 6 * 3600.0  # 6 hour arc
    L = 48
    N_colloc = 60
    lam_f, lam_th = 1.0, 10.0
    
    # Create detailed observations
    station_ecef = np.array([6378136.3, 0.0, 0.0])
    t_obs = np.linspace(t0, t1, 10)
    jd_obs = 2451545.0 + t_obs / 86400.0
    station_eci = np.array([ecef_to_eci(station_ecef, jd) for jd in jd_obs]).T
    
    ra_obs = np.linspace(0.0, 0.08, len(t_obs))
    dec_obs = np.linspace(0.0, 0.05, len(t_obs))
    obs = radec_to_trig(ra_obs, dec_obs)
    
    print(f"   Training with L={L}, N_colloc={N_colloc}, λ_f={lam_f}, λ_th={lam_th}...", end=" ")
    
    try:
        beta, model, result = fit_elm(t0, t1, L=L, N_colloc=N_colloc,
                                    obs=obs, t_obs=t_obs, station_eci=station_eci,
                                    lam_f=lam_f, lam_th=lam_th)
        
        if result.success:
            t_eval = np.linspace(t0, t1, 200)
            r, v, a, physics_rms, measurement_rms = evaluate_solution(
                beta, model, t_eval, obs, t_obs, station_eci)
            
            # Create residual data for plotting
            t_colloc = create_time_grid(t0, t1, N_colloc, 'linear')
            r_colloc, v_colloc, a_colloc = model.r_v_a(t_colloc, beta)
            from piod.dynamics import accel_2body_J2
            a_mod = np.apply_along_axis(accel_2body_J2, 0, r_colloc)
            physics_residuals = np.linalg.norm(a_colloc - a_mod, axis=0)
            
            detailed_results = {
                't0': t0,
                't1': t1,
                'L': L,
                'N_colloc': N_colloc,
                'lam_f': lam_f,
                'lam_th': lam_th,
                'success': True,
                'nfev': result.nfev,
                'cost': result.cost,
                'physics_rms': physics_rms,
                'measurement_rms': measurement_rms,
                'position_range': (np.min(np.linalg.norm(r, axis=0))/1000, 
                                 np.max(np.linalg.norm(r, axis=0))/1000),
                'velocity_range': (np.min(np.linalg.norm(v, axis=0)), 
                                 np.max(np.linalg.norm(v, axis=0))),
                'r': r.tolist(),
                'v': v.tolist(),
                't_eval': t_eval.tolist(),
                't_obs': t_obs.tolist(),
                'ra_obs': ra_obs.tolist(),
                'dec_obs': dec_obs.tolist(),
                'station_eci': station_eci.tolist(),
                't_colloc': t_colloc.tolist(),
                'physics_residuals': physics_residuals.tolist()
            }
            print(f"✓ Success (eval={result.nfev}, meas_rms={measurement_rms:.1f} arcsec)")
        else:
            detailed_results = {
                'success': False,
                'message': result.message
            }
            print(f"✗ Failed: {result.message}")
    except Exception as e:
        print(f"✗ Error: {e}")
        detailed_results = {
            'success': False,
            'error': str(e)
        }
    
    print("   Detailed training analysis complete!")
    print()
    
    # Save all results
    print("4. Saving Results...")
    
    all_results = {
        'learning_curve': learning_curve_results,
        'weight_comparison': weight_comparison_results,
        'detailed_analysis': detailed_results,
        'metadata': {
            'timestamp': np.datetime64('now').astype(str),
            'total_configurations': len(network_sizes) + len(weight_combinations) + 1,
            'successful_configurations': sum(1 for r in learning_curve_results if r['success']) + 
                                       sum(1 for r in weight_comparison_results if r['success']) + 
                                       (1 if detailed_results.get('success', False) else 0)
        }
    }
    
    # Save as JSON
    with open('data/training_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("   Results saved to: data/training_results.json")
    print()
    
    # Print summary
    print("=== TRAINING COMPLETE ===")
    print(f"Total configurations tested: {all_results['metadata']['total_configurations']}")
    print(f"Successful configurations: {all_results['metadata']['successful_configurations']}")
    print()
    print("Key findings:")
    
    # Best measurement residual
    best_meas_rms = min([r['measurement_rms'] for r in learning_curve_results if r['success']] + 
                       [r['measurement_rms'] for r in weight_comparison_results if r['success']])
    print(f"• Best measurement residual: {best_meas_rms:.1f} arcsec")
    
    # Best network size
    best_L = min([r['L'] for r in learning_curve_results if r['success'] and r['measurement_rms'] < 2000])
    print(f"• Best network size: L={best_L}")
    
    print("• All data ready for plotting!")
    print()
    print("Next step: Run the plotting script to visualize results.")

if __name__ == "__main__":
    train_and_save_data()
