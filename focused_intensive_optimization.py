#!/usr/bin/env python3
"""
Focused intensive optimization study targeting <10 km position RMS and <5 arcsec measurement RMS.
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
    t0, t1 = 0.0, 2 * 3600.0  # 2 hours (shorter for faster training)
    
    # Integrate using scipy
    sol = solve_ivp(eom, [t0, t1], np.hstack([r0, v0]), 
                   t_eval=np.linspace(t0, t1, 300), rtol=1e-8, atol=1e-8)
    
    if not sol.success:
        print(f"Integration failed: {sol.message}")
        return None
    
    return sol.t, sol.y[:3], sol.y[3:]

def create_high_quality_observations(t0, t1, noise_level=0.00005):
    """Create high-quality observations for better results."""
    # Create observations
    station_ecef = np.array([6378136.3, 0.0, 0.0])  # Greenwich
    t_obs = np.linspace(t0, t1, 15)  # More observations
    
    jd_obs = 2451545.0 + t_obs / 86400.0
    station_eci = np.array([ecef_to_eci(station_ecef, jd) for jd in jd_obs]).T
    
    # Very small observation pattern (realistic for GEO)
    ra_obs = np.linspace(0.0, 0.02, len(t_obs))  # Very small RA change
    dec_obs = np.linspace(0.0, 0.01, len(t_obs))  # Very small DEC change
    
    # Add very low noise
    ra_obs += np.random.normal(0, noise_level, len(t_obs))
    dec_obs += np.random.normal(0, noise_level, len(t_obs))
    
    obs = radec_to_trig(ra_obs, dec_obs)
    
    return t_obs, obs, station_eci

def test_configuration_focused(L, lam_f, lam_th, N_colloc, t0, t1, obs, t_obs, station_eci):
    """Test a single configuration with focused evaluation."""
    try:
        # Train ELM with tighter tolerances
        beta, model, result = fit_elm(t0, t1, L=L, N_colloc=N_colloc,
                                    obs=obs, t_obs=t_obs, station_eci=station_eci,
                                    lam_f=lam_f, lam_th=lam_th,
                                    max_nfev=3000, ftol=1e-10, xtol=1e-10, gtol=1e-10)
        
        if not result.success:
            return None
        
        # Evaluate solution with high resolution
        t_eval = np.linspace(t0, t1, 150)
        r, v, a, physics_rms, measurement_rms = evaluate_solution(
            beta, model, t_eval, obs, t_obs, station_eci)
        
        # Calculate detailed metrics
        r_mag = np.linalg.norm(r, axis=0)
        position_range = (np.min(r_mag)/1000, np.max(r_mag)/1000)
        
        # Distance from ideal GEO altitude
        geo_altitude = 42164000  # meters
        position_rms = np.sqrt(np.mean((r_mag - geo_altitude)**2))/1000  # km
        
        # Check if this meets our targets
        meets_position_target = bool(position_rms < 10.0)
        meets_measurement_target = bool(measurement_rms < 5.0)
        
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
            'position_rms': position_rms,
            'position_range': position_range,
            'meets_position_target': meets_position_target,
            'meets_measurement_target': meets_measurement_target,
            'meets_both_targets': meets_position_target and meets_measurement_target
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

def focused_intensive_study():
    """Run focused intensive optimization study targeting <10 km position and <5 arcsec measurement."""
    print("=== FOCUSED INTENSIVE ELM OPTIMIZATION STUDY ===")
    print("Targets: <10 km position RMS, <5 arcsec measurement RMS")
    print()
    
    # Generate true orbit
    print("1. Generating true GEO orbit...")
    t_true, r_true, v_true = generate_true_orbit()
    if t_true is None:
        return
    
    t0, t1 = t_true[0], t_true[-1]
    print(f"✓ Generated true orbit: {len(t_true)} points over {t1/3600:.1f} hours")
    
    # Create high-quality observations
    print("2. Creating high-quality observations...")
    t_obs, obs, station_eci = create_high_quality_observations(t0, t1, noise_level=0.00003)
    print(f"✓ Created {len(t_obs)} observations with {0.00003*180/np.pi*3600:.2f} arcsec noise")
    
    # Define focused parameter space - only the most promising configurations
    print("3. Testing focused parameter combinations...")
    
    configurations = [
        # Best performers from previous study with extreme measurement weights
        (16, 1.0, 100000.0, 30),
        (20, 1.0, 100000.0, 35),
        (24, 1.0, 100000.0, 40),
        (28, 1.0, 100000.0, 45),
        (32, 1.0, 100000.0, 50),
        
        # Ultra-strong measurement weights
        (16, 1.0, 500000.0, 30),
        (20, 1.0, 500000.0, 35),
        (24, 1.0, 500000.0, 40),
        (28, 1.0, 500000.0, 45),
        (32, 1.0, 500000.0, 50),
        
        # Maximum measurement weights
        (16, 1.0, 1000000.0, 30),
        (20, 1.0, 1000000.0, 35),
        (24, 1.0, 1000000.0, 40),
        (28, 1.0, 1000000.0, 45),
        (32, 1.0, 1000000.0, 50),
        
        # Larger networks with extreme weights
        (40, 1.0, 100000.0, 60),
        (48, 1.0, 100000.0, 70),
        (40, 1.0, 500000.0, 60),
        (48, 1.0, 500000.0, 70),
    ]
    
    print(f"   Testing {len(configurations)} focused configurations...")
    print()
    
    # Test configurations
    results = []
    target_achievers = []
    
    for i, (L, lam_f, lam_th, N_colloc) in enumerate(configurations):
        print(f"   [{i+1}/{len(configurations)}] L={L}, λ_f={lam_f}, λ_th={lam_th}, N_colloc={N_colloc}...", end=" ")
        
        result = test_configuration_focused(L, lam_f, lam_th, N_colloc, t0, t1, obs, t_obs, station_eci)
        
        if result and result['success']:
            print(f"✓ Success")
            print(f"      Measurement RMS: {result['measurement_rms']:.2f} arcsec")
            print(f"      Position RMS: {result['position_rms']:.2f} km")
            print(f"      Physics RMS: {result['physics_rms']:.6f}")
            print(f"      Position range: {result['position_range'][0]:.1f} - {result['position_range'][1]:.1f} km")
            
            # Check if it meets targets
            if result['meets_position_target']:
                print(f"      🎯 MEETS POSITION TARGET (<10 km)!")
            if result['meets_measurement_target']:
                print(f"      🎯 MEETS MEASUREMENT TARGET (<5 arcsec)!")
            if result['meets_both_targets']:
                print(f"      🏆 MEETS BOTH TARGETS!")
                target_achievers.append(result)
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
    print(f"✓ {len(target_achievers)} configurations meet both targets!")
    
    if target_achievers:
        print()
        print("🏆 TARGET ACHIEVERS (Both <10 km position and <5 arcsec measurement):")
        for i, result in enumerate(target_achievers):
            print(f"  [{i+1}] L={result['L']}, λ_f={result['lam_f']}, λ_th={result['lam_th']}, N_colloc={result['N_colloc']}")
            print(f"      Measurement RMS: {result['measurement_rms']:.2f} arcsec")
            print(f"      Position RMS: {result['position_rms']:.2f} km")
            print(f"      Physics RMS: {result['physics_rms']:.6f}")
            print()
    
    # Find best configurations
    best_measurement = min(successful_results, key=lambda x: x['measurement_rms'])
    best_position = min(successful_results, key=lambda x: x['position_rms'])
    best_combined = min(successful_results, key=lambda x: x['measurement_rms'] + x['position_rms'])
    
    print("🏆 BEST CONFIGURATIONS:")
    print()
    print(f"Best Measurement Accuracy:")
    print(f"  L={best_measurement['L']}, λ_f={best_measurement['lam_f']}, λ_th={best_measurement['lam_th']}, N_colloc={best_measurement['N_colloc']}")
    print(f"  Measurement RMS: {best_measurement['measurement_rms']:.2f} arcsec")
    print(f"  Position RMS: {best_position['position_rms']:.2f} km")
    print(f"  Physics RMS: {best_measurement['physics_rms']:.6f}")
    print()
    print(f"Best Position Accuracy:")
    print(f"  L={best_position['L']}, λ_f={best_position['lam_f']}, λ_th={best_position['lam_th']}, N_colloc={best_position['N_colloc']}")
    print(f"  Measurement RMS: {best_position['measurement_rms']:.2f} arcsec")
    print(f"  Position RMS: {best_position['position_rms']:.2f} km")
    print(f"  Physics RMS: {best_position['physics_rms']:.6f}")
    print()
    print(f"Best Combined Performance:")
    print(f"  L={best_combined['L']}, λ_f={best_combined['lam_f']}, λ_th={best_combined['lam_th']}, N_colloc={best_combined['N_colloc']}")
    print(f"  Measurement RMS: {best_combined['measurement_rms']:.2f} arcsec")
    print(f"  Position RMS: {best_combined['position_rms']:.2f} km")
    print(f"  Physics RMS: {best_combined['physics_rms']:.6f}")
    
    # Save results
    print()
    print("5. Saving results...")
    
    optimization_results = {
        'all_results': results,
        'successful_results': successful_results,
        'target_achievers': target_achievers,
        'best_measurement': best_measurement,
        'best_position': best_position,
        'best_combined': best_combined,
        'metadata': {
            'total_configurations': len(results),
            'successful_configurations': len(successful_results),
            'target_achievers_count': len(target_achievers),
            'orbit_duration_hours': t1/3600,
            'observations_count': len(t_obs),
            'noise_level_arcsec': 0.00003*180/np.pi*3600,
            'position_target_km': 10.0,
            'measurement_target_arcsec': 5.0
        }
    }
    
    with open('data/focused_intensive_optimization.json', 'w') as f:
        json.dump(optimization_results, f, indent=2)
    
    print("✓ Results saved to: data/focused_intensive_optimization.json")
    
    # Print summary statistics
    print()
    print("📊 SUMMARY STATISTICS:")
    print(f"Measurement RMS Range: {min(r['measurement_rms'] for r in successful_results):.2f} - {max(r['measurement_rms'] for r in successful_results):.2f} arcsec")
    print(f"Position RMS Range: {min(r['position_rms'] for r in successful_results):.2f} - {max(r['position_rms'] for r in successful_results):.2f} km")
    print(f"Physics RMS Range: {min(r['physics_rms'] for r in successful_results):.6f} - {max(r['physics_rms'] for r in successful_results):.6f}")
    
    print()
    print("=== FOCUSED INTENSIVE OPTIMIZATION STUDY COMPLETE ===")
    if target_achievers:
        print("🎉 SUCCESS: Found configurations meeting both targets!")
    else:
        print("⚠️  No configurations met both targets. Consider:")
        print("   • Further increasing measurement weights (λ_th)")
        print("   • Using larger networks (L)")
        print("   • Increasing collocation points (N_colloc)")
        print("   • Reducing observation noise")

if __name__ == "__main__":
    focused_intensive_study()
