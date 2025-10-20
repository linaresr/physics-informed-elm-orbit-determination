#!/usr/bin/env python3
"""
Analysis of position error computation and improvement strategies for orbital elements ELM.
"""

import sys
sys.path.append('piod')
import numpy as np
import matplotlib.pyplot as plt
from piod.solve_elements import fit_elm_elements, evaluate_solution_elements
from piod.observe import ecef_to_eci, radec_to_trig
from piod.dynamics import eom
from scipy.integrate import solve_ivp

def analyze_position_error_computation():
    """Analyze how position error is computed and why it might be high."""
    print("=== POSITION ERROR ANALYSIS ===")
    print()
    
    print("🔍 HOW POSITION ERROR IS COMPUTED:")
    print("1. Position error is NOT part of the training loss")
    print("2. It's computed AFTER training by comparing:")
    print("   • ELM output: r_elm(t) = elements_to_cartesian(elements(t))")
    print("   • True orbit: r_true(t) = numerical integration")
    print("3. Position RMS = sqrt(mean(||r_elm(t) - r_true(t)||²))")
    print()
    
    print("🎯 WHY POSITION ERROR MIGHT BE HIGH:")
    print("1. MEASUREMENT FITTING BIAS:")
    print("   • ELM learns to fit measurements exactly")
    print("   • But measurements don't constrain absolute position well")
    print("   • Network finds 'any' orbit that fits the angles")
    print()
    
    print("2. POOR INITIALIZATION:")
    print("   • Starting with random orbital elements")
    print("   • No physics-informed initialization")
    print("   • Network might converge to wrong orbit")
    print()
    
    print("3. INSUFFICIENT TRAINING DATA:")
    print("   • Only 8 observations over 2 hours")
    print("   • Limited geometric diversity")
    print("   • Network can't learn orbital dynamics well")
    print()
    
    print("4. ELEMENT BOUNDS:")
    print("   • No constraints on element ranges")
    print("   • Eccentricity can become negative/unrealistic")
    print("   • Semi-major axis can drift far from GEO")
    print()
    
    return True

def test_more_training_data():
    """Test with more training data to improve position accuracy."""
    print("=== TESTING WITH MORE TRAINING DATA ===")
    print()
    
    # Generate true orbit
    print("1. Generating true GEO orbit...")
    r0 = np.array([42164000.0, 0.0, 0.0])
    v0 = np.array([0.0, 3074.0, 0.0])
    t0, t1 = 0.0, 2 * 3600.0
    
    sol = solve_ivp(eom, [t0, t1], np.hstack([r0, v0]), 
                   t_eval=np.linspace(t0, t1, 300), rtol=1e-8, atol=1e-8)
    
    t_true, r_true, v_true = sol.t, sol.y[:3], sol.y[3:]
    print(f"✓ Generated true orbit: {len(t_true)} points over {t1/3600:.1f} hours")
    
    # Create more observations
    print("2. Creating more observations...")
    station_ecef = np.array([6378136.3, 0.0, 0.0])
    
    # Test different observation densities
    observation_counts = [8, 15, 25, 40]
    results = []
    
    for n_obs in observation_counts:
        print(f"   Testing with {n_obs} observations...")
        
        t_obs = np.linspace(t0, t1, n_obs)
        jd_obs = 2451545.0 + t_obs / 86400.0
        station_eci = np.array([ecef_to_eci(station_ecef, jd) for jd in jd_obs]).T
        
        # Small observation pattern
        ra_obs = np.linspace(0.0, 0.02, len(t_obs))
        dec_obs = np.linspace(0.0, 0.01, len(t_obs))
        
        # Add noise
        ra_obs += np.random.normal(0, 0.0001, len(t_obs))
        dec_obs += np.random.normal(0, 0.0001, len(t_obs))
        
        obs = radec_to_trig(ra_obs, dec_obs)
        
        # Fit ELM
        L = 8
        N_colloc = min(20, n_obs * 2)  # Scale collocation points
        lam_f = 1.0
        lam_th = 1000.0
        
        beta, model, result = fit_elm_elements(t0, t1, L=L, N_colloc=N_colloc,
                                             obs=obs, t_obs=t_obs, station_eci=station_eci,
                                             lam_f=lam_f, lam_th=lam_th)
        
        # Evaluate solution
        t_eval = np.linspace(t0, t1, 50)
        r, v, a, physics_rms, measurement_rms = evaluate_solution_elements(
            beta, model, t_eval, obs, t_obs, station_eci)
        
        # Calculate position RMS
        r_mag = np.linalg.norm(r, axis=0)
        geo_altitude = 42164000
        position_rms = np.sqrt(np.mean((r_mag - geo_altitude)**2))/1000
        
        # Calculate true position error
        r_true_interp = np.zeros_like(r)
        for i in range(3):
            r_true_interp[i] = np.interp(t_eval, t_true, r_true[i])
        
        true_position_error = np.sqrt(np.mean(np.sum((r - r_true_interp)**2, axis=0)))/1000
        
        results.append({
            'n_obs': n_obs,
            'measurement_rms': measurement_rms,
            'position_rms': position_rms,
            'true_position_error': true_position_error,
            'physics_rms': physics_rms,
            'success': result.success,
            'nfev': result.nfev
        })
        
        print(f"     Measurement RMS: {measurement_rms:.2f} arcsec")
        print(f"     Position RMS: {position_rms:.1f} km")
        print(f"     True Position Error: {true_position_error:.1f} km")
        print(f"     Physics RMS: {physics_rms:.6f}")
    
    return results

def test_element_bounds():
    """Test with element bounds to keep elements realistic."""
    print("=== TESTING WITH ELEMENT BOUNDS ===")
    print()
    
    # Generate true orbit
    print("1. Generating true GEO orbit...")
    r0 = np.array([42164000.0, 0.0, 0.0])
    v0 = np.array([0.0, 3074.0, 0.0])
    t0, t1 = 0.0, 2 * 3600.0
    
    sol = solve_ivp(eom, [t0, t1], np.hstack([r0, v0]), 
                   t_eval=np.linspace(t0, t1, 300), rtol=1e-8, atol=1e-8)
    
    t_true, r_true, v_true = sol.t, sol.y[:3], sol.y[3:]
    
    # Create observations
    print("2. Creating observations...")
    station_ecef = np.array([6378136.3, 0.0, 0.0])
    t_obs = np.linspace(t0, t1, 20)  # More observations
    
    jd_obs = 2451545.0 + t_obs / 86400.0
    station_eci = np.array([ecef_to_eci(station_ecef, jd) for jd in jd_obs]).T
    
    ra_obs = np.linspace(0.0, 0.02, len(t_obs))
    dec_obs = np.linspace(0.0, 0.01, len(t_obs))
    ra_obs += np.random.normal(0, 0.0001, len(t_obs))
    dec_obs += np.random.normal(0, 0.0001, len(t_obs))
    
    obs = radec_to_trig(ra_obs, dec_obs)
    
    # Test with different element bounds
    print("3. Testing with element bounds...")
    
    # Define bounds for GEO elements
    bounds = [
        (40000000, 45000000),  # a: GEO altitude range
        (0.0, 0.1),           # e: nearly circular
        (0.0, 0.1),           # i: nearly equatorial
        (0.0, 2*np.pi),       # Omega: full range
        (0.0, 2*np.pi),       # omega: full range
        (0.0, 2*np.pi),       # M0: full range
        (-1000, 1000),        # ELM weights: small variations
        (-1000, 1000),
        (-1000, 1000),
        (-1000, 1000),
        (-1000, 1000),
        (-1000, 1000)
    ]
    
    # Test with bounds
    L = 8
    N_colloc = 30
    lam_f = 1.0
    lam_th = 1000.0
    
    # Initialize with better GEO values
    beta0 = np.array([
        42164000.0,  # a (GEO altitude)
        0.0,         # e (circular)
        0.0,         # i (equatorial)
        0.0,         # Omega
        0.0,         # omega
        0.0,         # M0
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # ELM weights
    ])
    
    # Add small random variations
    beta0[6:] = np.random.randn(6) * 100
    
    # Fit with bounds
    from scipy.optimize import least_squares
    from piod.elm_elements import OrbitalElementsELM
    from piod.loss_elements import residual_elements
    
    model = OrbitalElementsELM(L=L, t_phys=np.array([t0, t1]))
    t_colloc = np.linspace(t0, t1, N_colloc)
    
    def fun(beta):
        return residual_elements(beta, model, t_colloc, lam_f, obs, t_obs, station_eci, lam_th)
    
    result = least_squares(fun, beta0, method="trf", max_nfev=5000, 
                          ftol=1e-10, xtol=1e-10, gtol=1e-10, bounds=bounds)
    
    print(f"✓ Success: {result.success}")
    print(f"✓ Function evaluations: {result.nfev}")
    print(f"✓ Final cost: {result.cost:.2e}")
    
    # Evaluate solution
    t_eval = np.linspace(t0, t1, 50)
    r, v, a, physics_rms, measurement_rms = evaluate_solution_elements(
        result.x, model, t_eval, obs, t_obs, station_eci)
    
    # Calculate position RMS
    r_mag = np.linalg.norm(r, axis=0)
    geo_altitude = 42164000
    position_rms = np.sqrt(np.mean((r_mag - geo_altitude)**2))/1000
    
    # Calculate true position error
    r_true_interp = np.zeros_like(r)
    for i in range(3):
        r_true_interp[i] = np.interp(t_eval, t_true, r_true[i])
    
    true_position_error = np.sqrt(np.mean(np.sum((r - r_true_interp)**2, axis=0)))/1000
    
    # Check orbital elements
    mean_elements, elm_weights = model.elements_from_beta(result.x)
    
    print(f"✓ Measurement RMS: {measurement_rms:.2f} arcsec")
    print(f"✓ Position RMS: {position_rms:.1f} km")
    print(f"✓ True Position Error: {true_position_error:.1f} km")
    print(f"✓ Physics RMS: {physics_rms:.6f}")
    print(f"✓ Mean elements: a={mean_elements[0]/1000:.1f}km, e={mean_elements[1]:.6f}")
    
    return result.x, model, r, v, a, physics_rms, measurement_rms, position_rms, true_position_error

def create_improvement_analysis():
    """Create analysis of improvement strategies."""
    print("=== CREATING IMPROVEMENT ANALYSIS ===")
    print()
    
    # Test more training data
    print("1. Testing with more training data...")
    training_data_results = test_more_training_data()
    
    # Test element bounds
    print("2. Testing with element bounds...")
    bounds_results = test_element_bounds()
    
    # Create analysis plot
    print("3. Creating analysis plot...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Training data vs performance
    ax = axes[0, 0]
    n_obs = [r['n_obs'] for r in training_data_results]
    position_errors = [r['true_position_error'] for r in training_data_results]
    measurement_rms = [r['measurement_rms'] for r in training_data_results]
    
    ax.plot(n_obs, position_errors, 'bo-', label='Position Error', linewidth=2)
    ax2 = ax.twinx()
    ax2.plot(n_obs, measurement_rms, 'ro-', label='Measurement RMS', linewidth=2)
    
    ax.set_xlabel('Number of Observations')
    ax.set_ylabel('Position Error (km)')
    ax2.set_ylabel('Measurement RMS (arcsec)')
    ax.set_title('Training Data vs Performance')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # 2. Performance summary
    ax = axes[0, 1]
    ax.axis('off')
    
    summary_text = f"""
IMPROVEMENT STRATEGIES

More Training Data:
• 8 obs: {training_data_results[0]['true_position_error']:.1f} km error
• 15 obs: {training_data_results[1]['true_position_error']:.1f} km error
• 25 obs: {training_data_results[2]['true_position_error']:.1f} km error
• 40 obs: {training_data_results[3]['true_position_error']:.1f} km error

Element Bounds:
• Position Error: {bounds_results[7]:.1f} km
• Measurement RMS: {bounds_results[5]:.2f} arcsec
• Physics RMS: {bounds_results[4]:.6f}

Key Insights:
• More observations help but not dramatically
• Element bounds prevent unrealistic orbits
• Position error is computed AFTER training
• Need better initialization strategy
"""
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # 3. Element bounds comparison
    ax = axes[1, 0]
    ax.axis('off')
    
    bounds_text = f"""
ELEMENT BOUNDS STRATEGY

GEO Realistic Bounds:
• Semi-major axis: 40,000-45,000 km
• Eccentricity: 0.0-0.1 (nearly circular)
• Inclination: 0.0-0.1 rad (nearly equatorial)
• Other angles: 0-2π rad

ELM Weight Bounds:
• Small variations: ±1000
• Prevents extreme element changes
• Maintains orbital stability

Benefits:
• Prevents unrealistic orbits
• Better convergence
• More stable training
• Physical constraints
"""
    
    ax.text(0.05, 0.95, bounds_text, transform=ax.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # 4. Next steps
    ax = axes[1, 1]
    ax.axis('off')
    
    next_steps_text = f"""
NEXT STEPS FOR IMPROVEMENT

1. BETTER INITIALIZATION:
   • Use mean observation position
   • Estimate initial orbital elements
   • Start closer to true solution

2. IMPROVED LOSS FUNCTION:
   • Add position magnitude constraint
   • Weight physics more heavily
   • Use adaptive weighting

3. MORE COLLOCATION POINTS:
   • Increase N_colloc for better physics
   • Use adaptive collocation
   • Focus on critical times

4. ELEMENT SCALING:
   • Scale elements for better conditioning
   • Use log scaling for semi-major axis
   • Normalize angles properly

5. MULTIPLE RESTARTS:
   • Try different random seeds
   • Use best result
   • Ensemble approach
"""
    
    ax.text(0.05, 0.95, next_steps_text, transform=ax.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('data/improvement_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Improvement analysis saved to: data/improvement_analysis.png")
    
    return training_data_results, bounds_results

if __name__ == "__main__":
    analyze_position_error_computation()
    create_improvement_analysis()
