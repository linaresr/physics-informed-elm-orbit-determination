#!/usr/bin/env python3
"""
Focused optimization study to find the best ELM parameters.
Simplified approach focusing on measurement and position residuals.
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
    t0, t1 = 0.0, 3 * 3600.0  # 3 hours (shorter for faster training)
    
    # Integrate using scipy
    sol = solve_ivp(eom, [t0, t1], np.hstack([r0, v0]), 
                   t_eval=np.linspace(t0, t1, 500), rtol=1e-8, atol=1e-8)
    
    if not sol.success:
        print(f"Integration failed: {sol.message}")
        return None
    
    return sol.t, sol.y[:3], sol.y[3:]

def create_simple_observations(t0, t1, noise_level=0.0005):
    """Create simple observations for testing."""
    # Create observations
    station_ecef = np.array([6378136.3, 0.0, 0.0])  # Greenwich
    t_obs = np.linspace(t0, t1, 8)  # 8 observations
    
    jd_obs = 2451545.0 + t_obs / 86400.0
    station_eci = np.array([ecef_to_eci(station_ecef, jd) for jd in jd_obs]).T
    
    # Simple observation pattern
    ra_obs = np.linspace(0.0, 0.05, len(t_obs))  # Small RA change
    dec_obs = np.linspace(0.0, 0.03, len(t_obs))  # Small DEC change
    
    # Add noise
    ra_obs += np.random.normal(0, noise_level, len(t_obs))
    dec_obs += np.random.normal(0, noise_level, len(t_obs))
    
    obs = radec_to_trig(ra_obs, dec_obs)
    
    return t_obs, obs, station_eci

def test_configuration(L, lam_f, lam_th, N_colloc, t0, t1, obs, t_obs, station_eci):
    """Test a single configuration."""
    try:
        # Train ELM
        beta, model, result = fit_elm(t0, t1, L=L, N_colloc=N_colloc,
                                    obs=obs, t_obs=t_obs, station_eci=station_eci,
                                    lam_f=lam_f, lam_th=lam_th)
        
        if not result.success:
            return None
        
        # Evaluate solution
        t_eval = np.linspace(t0, t1, 100)
        r, v, a, physics_rms, measurement_rms = evaluate_solution(
            beta, model, t_eval, obs, t_obs, station_eci)
        
        # Calculate position range
        r_mag = np.linalg.norm(r, axis=0)
        position_range = (np.min(r_mag)/1000, np.max(r_mag)/1000)
        
        return {
            'L': L,
            'lam_f': lam_f,
            'lam_th': lam_th,
            'N_colloc': N_colloc,
            'success': True,
            'nfev': result.nfev,
            'cost': result.cost,
            'physics_rms': physics_rms,
            'measurement_rms': measurement_rms,
            'position_range': position_range,
            'position_rms': np.sqrt(np.mean((r_mag - 42164000)**2))/1000  # Distance from GEO
        }
        
    except Exception as e:
        return {
            'L': L,
            'lam_f': lam_f,
            'lam_th': lam_th,
            'N_colloc': N_colloc,
            'success': False,
            'error': str(e)
        }

def focused_optimization_study():
    """Run focused optimization study."""
    print("=== FOCUSED ELM OPTIMIZATION STUDY ===")
    print("Focus: Finding optimal parameters for measurement and position accuracy")
    print()
    
    # Generate true orbit
    print("1. Generating true GEO orbit...")
    t_true, r_true, v_true = generate_true_orbit()
    if t_true is None:
        return
    
    t0, t1 = t_true[0], t_true[-1]
    print(f"✓ Generated true orbit: {len(t_true)} points over {t1/3600:.1f} hours")
    
    # Create observations
    print("2. Creating observations...")
    t_obs, obs, station_eci = create_simple_observations(t0, t1, noise_level=0.0003)
    print(f"✓ Created {len(t_obs)} observations with {0.0003*180/np.pi*3600:.1f} arcsec noise")
    
    # Define focused parameter space
    print("3. Testing focused parameter combinations...")
    
    # Focus on promising configurations
    configurations = [
        # Small networks with strong measurement weights
        (8, 1.0, 100.0, 20),
        (12, 1.0, 100.0, 25),
        (16, 1.0, 100.0, 30),
        (20, 1.0, 100.0, 35),
        (24, 1.0, 100.0, 40),
        
        # Medium networks with balanced weights
        (16, 1.0, 10.0, 30),
        (20, 1.0, 10.0, 35),
        (24, 1.0, 10.0, 40),
        (28, 1.0, 10.0, 45),
        (32, 1.0, 10.0, 50),
        
        # Larger networks with strong measurement weights
        (32, 1.0, 100.0, 50),
        (40, 1.0, 100.0, 60),
        (48, 1.0, 100.0, 70),
        
        # Very strong measurement weights
        (16, 1.0, 1000.0, 30),
        (20, 1.0, 1000.0, 35),
        (24, 1.0, 1000.0, 40),
        
        # Physics-focused (for comparison)
        (24, 100.0, 1.0, 40),
        (32, 100.0, 1.0, 50),
    ]
    
    print(f"   Testing {len(configurations)} focused configurations...")
    print()
    
    # Test configurations
    results = []
    for i, (L, lam_f, lam_th, N_colloc) in enumerate(configurations):
        print(f"   [{i+1}/{len(configurations)}] L={L}, λ_f={lam_f}, λ_th={lam_th}, N_colloc={N_colloc}...", end=" ")
        
        result = test_configuration(L, lam_f, lam_th, N_colloc, t0, t1, obs, t_obs, station_eci)
        
        if result and result['success']:
            print(f"✓ Success")
            print(f"      Measurement RMS: {result['measurement_rms']:.1f} arcsec")
            print(f"      Position RMS: {result['position_rms']:.1f} km")
            print(f"      Physics RMS: {result['physics_rms']:.6f}")
            print(f"      Position range: {result['position_range'][0]:.1f} - {result['position_range'][1]:.1f} km")
        else:
            print(f"✗ Failed")
            if result:
                print(f"      Error: {result.get('error', 'Unknown')}")
        
        results.append(result)
        print()
    
    # Analyze results
    print("4. Analyzing results...")
    
    successful_results = [r for r in results if r and r.get('success', False)]
    
    if not successful_results:
        print("✗ No successful configurations found!")
        return
    
    print(f"✓ {len(successful_results)} successful configurations out of {len(results)}")
    
    # Find best configurations
    best_measurement = min(successful_results, key=lambda x: x['measurement_rms'])
    best_position = min(successful_results, key=lambda x: x['position_rms'])
    best_combined = min(successful_results, key=lambda x: x['measurement_rms'] + x['position_rms'])
    
    print()
    print("🏆 BEST CONFIGURATIONS:")
    print()
    print(f"Best Measurement Accuracy:")
    print(f"  L={best_measurement['L']}, λ_f={best_measurement['lam_f']}, λ_th={best_measurement['lam_th']}, N_colloc={best_measurement['N_colloc']}")
    print(f"  Measurement RMS: {best_measurement['measurement_rms']:.1f} arcsec")
    print(f"  Position RMS: {best_measurement['position_rms']:.1f} km")
    print(f"  Physics RMS: {best_measurement['physics_rms']:.6f}")
    print(f"  Position range: {best_measurement['position_range'][0]:.1f} - {best_measurement['position_range'][1]:.1f} km")
    print()
    print(f"Best Position Accuracy:")
    print(f"  L={best_position['L']}, λ_f={best_position['lam_f']}, λ_th={best_position['lam_th']}, N_colloc={best_position['N_colloc']}")
    print(f"  Measurement RMS: {best_position['measurement_rms']:.1f} arcsec")
    print(f"  Position RMS: {best_position['position_rms']:.1f} km")
    print(f"  Physics RMS: {best_position['physics_rms']:.6f}")
    print(f"  Position range: {best_position['position_range'][0]:.1f} - {best_position['position_range'][1]:.1f} km")
    print()
    print(f"Best Combined Performance:")
    print(f"  L={best_combined['L']}, λ_f={best_combined['lam_f']}, λ_th={best_combined['lam_th']}, N_colloc={best_combined['N_colloc']}")
    print(f"  Measurement RMS: {best_combined['measurement_rms']:.1f} arcsec")
    print(f"  Position RMS: {best_combined['position_rms']:.1f} km")
    print(f"  Physics RMS: {best_combined['physics_rms']:.6f}")
    print(f"  Position range: {best_combined['position_range'][0]:.1f} - {best_combined['position_range'][1]:.1f} km")
    
    # Save results
    print()
    print("5. Saving results...")
    
    optimization_results = {
        'all_results': results,
        'successful_results': successful_results,
        'best_measurement': best_measurement,
        'best_position': best_position,
        'best_combined': best_combined,
        'metadata': {
            'total_configurations': len(results),
            'successful_configurations': len(successful_results),
            'orbit_duration_hours': t1/3600,
            'observations_count': len(t_obs),
            'noise_level_arcsec': 0.0003*180/np.pi*3600
        }
    }
    
    with open('data/focused_optimization.json', 'w') as f:
        json.dump(optimization_results, f, indent=2)
    
    print("✓ Results saved to: data/focused_optimization.json")
    
    # Print summary statistics
    print()
    print("📊 SUMMARY STATISTICS:")
    print(f"Measurement RMS Range: {min(r['measurement_rms'] for r in successful_results):.1f} - {max(r['measurement_rms'] for r in successful_results):.1f} arcsec")
    print(f"Position RMS Range: {min(r['position_rms'] for r in successful_results):.1f} - {max(r['position_rms'] for r in successful_results):.1f} km")
    print(f"Physics RMS Range: {min(r['physics_rms'] for r in successful_results):.6f} - {max(r['physics_rms'] for r in successful_results):.6f}")
    
    print()
    print("=== FOCUSED OPTIMIZATION STUDY COMPLETE ===")
    print("Use the best configurations for improved performance!")

if __name__ == "__main__":
    focused_optimization_study()
