#!/usr/bin/env python3
"""
Comprehensive position error improvement analysis and implementation.
"""

import sys
sys.path.append('piod')
import numpy as np
import matplotlib.pyplot as plt
from piod.solve_elements_enhanced import fit_elm_elements_enhanced, evaluate_solution_elements_enhanced
from piod.observe import ecef_to_eci, radec_to_trig
from piod.dynamics import eom
from scipy.integrate import solve_ivp

def generate_true_orbit():
    """Generate a realistic GEO orbit using numerical integration."""
    r0 = np.array([42164000.0, 0.0, 0.0])  # GEO altitude
    v0 = np.array([0.0, 3074.0, 0.0])     # Circular orbit velocity
    t0, t1 = 0.0, 2 * 3600.0  # 2 hours
    
    sol = solve_ivp(eom, [t0, t1], np.hstack([r0, v0]), 
                   t_eval=np.linspace(t0, t1, 300), rtol=1e-8, atol=1e-8)
    
    if not sol.success:
        print(f"Integration failed: {sol.message}")
        return None
    
    return sol.t, sol.y[:3], sol.y[3:]

def create_observations(t0, t1, noise_level=0.0001, n_obs=15):
    """Create observations for testing."""
    station_ecef = np.array([6378136.3, 0.0, 0.0])  # Greenwich
    t_obs = np.linspace(t0, t1, n_obs)
    
    jd_obs = 2451545.0 + t_obs / 86400.0
    station_eci = np.array([ecef_to_eci(station_ecef, jd) for jd in jd_obs]).T
    
    # Small observation pattern
    ra_obs = np.linspace(0.0, 0.02, len(t_obs))
    dec_obs = np.linspace(0.0, 0.01, len(t_obs))
    
    # Add noise
    ra_obs += np.random.normal(0, noise_level, len(t_obs))
    dec_obs += np.random.normal(0, noise_level, len(t_obs))
    
    obs = radec_to_trig(ra_obs, dec_obs)
    
    return t_obs, obs, station_eci

def test_position_weight_tuning(t0, t1, obs, t_obs, station_eci):
    """Test different position magnitude weights."""
    print("=== TESTING POSITION WEIGHT TUNING ===")
    print()
    
    # Test different position weights
    position_weights = [1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]
    results = []
    
    for lam_r in position_weights:
        print(f"Testing λ_r = {lam_r:.0f}...")
        
        try:
            beta, model, result = fit_elm_elements_enhanced(t0, t1, L=8, N_colloc=20,
                                                           obs=obs, t_obs=t_obs, station_eci=station_eci,
                                                           lam_f=1.0, lam_r=lam_r, lam_th=1000.0)
            
            # Evaluate solution
            t_eval = np.linspace(t0, t1, 50)
            r, v, a, physics_rms, measurement_rms, position_magnitude_rms = evaluate_solution_elements_enhanced(
                beta, model, t_eval, obs, t_obs, station_eci)
            
            results.append({
                'lam_r': lam_r,
                'position_rms': position_magnitude_rms,
                'measurement_rms': measurement_rms,
                'physics_rms': physics_rms,
                'success': result.success,
                'nfev': result.nfev,
                'cost': result.cost
            })
            
            print(f"  Position RMS: {position_magnitude_rms:.1f} km")
            print(f"  Measurement RMS: {measurement_rms:.2f} arcsec")
            print(f"  Physics RMS: {physics_rms:.6f}")
            print(f"  Function evals: {result.nfev}")
            
        except Exception as e:
            print(f"  Failed: {e}")
            results.append({
                'lam_r': lam_r,
                'position_rms': float('inf'),
                'measurement_rms': float('inf'),
                'physics_rms': float('inf'),
                'success': False,
                'nfev': 0,
                'cost': float('inf')
            })
    
    return results

def test_better_initialization(t0, t1, obs, t_obs, station_eci):
    """Test better initialization strategies."""
    print("=== TESTING BETTER INITIALIZATION ===")
    print()
    
    # Strategy 1: Estimate initial position from observations
    print("Strategy 1: Estimate position from observations...")
    
    # Use first observation to estimate initial position
    station_pos = station_eci[:, 0]  # First station position
    
    # Convert observation to unit vector
    sin_ra, cos_ra, sin_dec = obs[0], obs[1], obs[2]  # sin(ra), cos(ra), sin(dec)
    ra = np.arctan2(sin_ra, cos_ra) % (2*np.pi)
    dec = np.arcsin(sin_dec)
    
    # Estimate position (simplified)
    estimated_distance = 42164000  # GEO altitude
    r_estimated = estimated_distance * np.array([
        np.cos(dec) * np.cos(ra),
        np.cos(dec) * np.sin(ra),
        np.sin(dec)
    ])
    
    # Convert to orbital elements (simplified)
    r_mag = np.linalg.norm(r_estimated)
    a_est = r_mag
    e_est = 0.0  # Assume circular
    i_est = 0.0  # Assume equatorial
    
    # Estimate other elements from position
    Omega_est = np.arctan2(r_estimated[1], r_estimated[0]) % (2*np.pi)
    omega_est = 0.0
    M_est = 0.0
    
    print(f"  Estimated elements: a={a_est/1000:.1f}km, e={e_est:.3f}, i={i_est:.3f}")
    
    # Test with estimated initialization
    try:
        # Create custom initial guess
        beta0_custom = np.array([
            a_est, e_est, i_est, Omega_est, omega_est, M_est,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # Small ELM weights
        ])
        
        # Use custom initialization in enhanced solver
        from scipy.optimize import least_squares
        from piod.elm_elements import OrbitalElementsELM
        from piod.loss_elements_enhanced import residual_elements_enhanced
        
        model = OrbitalElementsELM(L=8, t_phys=np.array([t0, t1]))
        t_colloc = np.linspace(t0, t1, 20)
        
        def fun(beta):
            return residual_elements_enhanced(beta, model, t_colloc, 1.0, obs, t_obs, station_eci, 1000.0, 1000.0)
        
        bounds = (
            np.array([40000000, 0.0, 0.0, 0.0, 0.0, 0.0, -1000, -1000, -1000, -1000, -1000, -1000]),
            np.array([45000000, 0.1, 0.1, 2*np.pi, 2*np.pi, 2*np.pi, 1000, 1000, 1000, 1000, 1000, 1000])
        )
        
        result = least_squares(fun, beta0_custom, method="trf", max_nfev=5000, 
                              ftol=1e-10, xtol=1e-10, gtol=1e-10, bounds=bounds)
        
        # Evaluate solution
        t_eval = np.linspace(t0, t1, 50)
        r, v, a, physics_rms, measurement_rms, position_magnitude_rms = evaluate_solution_elements_enhanced(
            result.x, model, t_eval, obs, t_obs, station_eci)
        
        print(f"  Custom init results:")
        print(f"    Position RMS: {position_magnitude_rms:.1f} km")
        print(f"    Measurement RMS: {measurement_rms:.2f} arcsec")
        print(f"    Physics RMS: {physics_rms:.6f}")
        print(f"    Success: {result.success}")
        print(f"    Function evals: {result.nfev}")
        
        return position_magnitude_rms, measurement_rms, physics_rms, result.nfev
        
    except Exception as e:
        print(f"  Custom initialization failed: {e}")
        return None

def test_longer_observation_arc(t0, t1, obs, t_obs, station_eci):
    """Test with longer observation arc."""
    print("=== TESTING LONGER OBSERVATION ARC ===")
    print()
    
    # Extend to 4 hours
    t0_extended = t0
    t1_extended = t0 + 4 * 3600.0  # 4 hours
    
    print(f"Extended arc: {t0_extended/3600:.1f} to {t1_extended/3600:.1f} hours")
    
    # Create extended observations
    t_obs_extended = np.linspace(t0_extended, t1_extended, 25)  # More observations
    
    jd_obs_extended = 2451545.0 + t_obs_extended / 86400.0
    station_eci_extended = np.array([ecef_to_eci(np.array([6378136.3, 0.0, 0.0]), jd) for jd in jd_obs_extended]).T
    
    # Extended observation pattern
    ra_obs_extended = np.linspace(0.0, 0.04, len(t_obs_extended))  # Larger angular range
    dec_obs_extended = np.linspace(0.0, 0.02, len(t_obs_extended))
    
    # Add noise
    ra_obs_extended += np.random.normal(0, 0.0001, len(t_obs_extended))
    dec_obs_extended += np.random.normal(0, 0.0001, len(t_obs_extended))
    
    obs_extended = radec_to_trig(ra_obs_extended, dec_obs_extended)
    
    try:
        beta, model, result = fit_elm_elements_enhanced(t0_extended, t1_extended, L=8, N_colloc=30,
                                                       obs=obs_extended, t_obs=t_obs_extended, station_eci=station_eci_extended,
                                                       lam_f=1.0, lam_r=1000.0, lam_th=1000.0)
        
        # Evaluate solution
        t_eval = np.linspace(t0_extended, t1_extended, 50)
        r, v, a, physics_rms, measurement_rms, position_magnitude_rms = evaluate_solution_elements_enhanced(
            beta, model, t_eval, obs_extended, t_obs_extended, station_eci_extended)
        
        print(f"Extended arc results:")
        print(f"  Position RMS: {position_magnitude_rms:.1f} km")
        print(f"  Measurement RMS: {measurement_rms:.2f} arcsec")
        print(f"  Physics RMS: {physics_rms:.6f}")
        print(f"  Success: {result.success}")
        print(f"  Function evals: {result.nfev}")
        
        return position_magnitude_rms, measurement_rms, physics_rms, result.nfev
        
    except Exception as e:
        print(f"Extended arc test failed: {e}")
        return None

def create_improvement_analysis():
    """Create comprehensive improvement analysis."""
    print("=== CREATING IMPROVEMENT ANALYSIS ===")
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
    t_obs, obs, station_eci = create_observations(t0, t1, noise_level=0.0001, n_obs=15)
    print(f"✓ Created {len(t_obs)} observations with {0.0001*180/np.pi*3600:.2f} arcsec noise")
    
    # Test position weight tuning
    print("3. Testing position weight tuning...")
    weight_results = test_position_weight_tuning(t0, t1, obs, t_obs, station_eci)
    
    # Test better initialization
    print("4. Testing better initialization...")
    init_results = test_better_initialization(t0, t1, obs, t_obs, station_eci)
    
    # Test longer observation arc
    print("5. Testing longer observation arc...")
    extended_results = test_longer_observation_arc(t0, t1, obs, t_obs, station_eci)
    
    # Create analysis plot
    print("6. Creating improvement analysis plot...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Position weight vs performance
    ax = axes[0, 0]
    weights = [r['lam_r'] for r in weight_results if r['success']]
    position_rms = [r['position_rms'] for r in weight_results if r['success']]
    measurement_rms = [r['measurement_rms'] for r in weight_results if r['success']]
    
    ax.loglog(weights, position_rms, 'bo-', label='Position RMS', linewidth=2)
    ax2 = ax.twinx()
    ax2.loglog(weights, measurement_rms, 'ro-', label='Measurement RMS', linewidth=2)
    
    ax.set_xlabel('Position Weight (λ_r)')
    ax.set_ylabel('Position RMS (km)')
    ax2.set_ylabel('Measurement RMS (arcsec)')
    ax.set_title('Position Weight vs Performance')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # 2. Improvement summary
    ax = axes[0, 1]
    ax.axis('off')
    
    # Find best weight
    best_weight_result = min([r for r in weight_results if r['success']], key=lambda x: x['position_rms'])
    
    summary_text = f"""
IMPROVEMENT ANALYSIS RESULTS

Position Weight Tuning:
• Best weight: λ_r = {best_weight_result['lam_r']:.0f}
• Best position RMS: {best_weight_result['position_rms']:.1f} km
• Best measurement RMS: {best_weight_result['measurement_rms']:.2f} arcsec
• Function evals: {best_weight_result['nfev']}

Better Initialization:
• Custom init: {'✓ SUCCESS' if init_results else '✗ FAILED'}
• Position RMS: {init_results[0]:.1f} km if init_results else 'N/A'
• Measurement RMS: {init_results[1]:.2f} arcsec if init_results else 'N/A'

Longer Observation Arc:
• Extended arc: {'✓ SUCCESS' if extended_results else '✗ FAILED'}
• Position RMS: {extended_results[0]:.1f} km if extended_results else 'N/A'
• Measurement RMS: {extended_results[1]:.2f} arcsec if extended_results else 'N/A'

BEST OVERALL RESULT:
• Position RMS: {min([r['position_rms'] for r in weight_results if r['success']] + ([init_results[0]] if init_results else []) + ([extended_results[0]] if extended_results else [])):.1f} km
• Measurement RMS: {'N/A'}
• Status: {'✓ TARGET ACHIEVED' if min([r['position_rms'] for r in weight_results if r['success']] + ([init_results[0]] if init_results else []) + ([extended_results[0]] if extended_results else [])) < 10.0 else '✗ TARGET NOT ACHIEVED'}
"""
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # 3. Performance comparison
    ax = axes[1, 0]
    ax.axis('off')
    
    performance_text = f"""
PERFORMANCE COMPARISON

Original Enhanced Approach:
• Position RMS: 7,612.7 km
• Measurement RMS: 0.71 arcsec
• Status: ✗ Target not achieved

Improved Approach:
• Position RMS: {best_weight_result['position_rms']:.1f} km
• Measurement RMS: {best_weight_result['measurement_rms']:.2f} arcsec
• Status: {'✓ Target achieved' if best_weight_result['position_rms'] < 10.0 else '✗ Target not achieved'}

Improvement:
• Position: {7612.7/best_weight_result['position_rms']:.1f}x better
• Measurement: {'✓ Maintained' if best_weight_result['measurement_rms'] < 5.0 else '✗ Degraded'}

Key Insights:
• Position weight tuning: {'✓ Effective' if best_weight_result['position_rms'] < 7612.7 else '✗ Limited'}
• Better initialization: {'✓ Helpful' if init_results and init_results[0] < 7612.7 else '✗ Not helpful'}
• Longer arcs: {'✓ Beneficial' if extended_results and extended_results[0] < 7612.7 else '✗ Not beneficial'}
"""
    
    ax.text(0.05, 0.95, performance_text, transform=ax.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # 4. Next steps
    ax = axes[1, 1]
    ax.axis('off')
    
    next_steps_text = f"""
NEXT STEPS FOR FURTHER IMPROVEMENT

Current Status:
• Best position RMS: {best_weight_result['position_rms']:.1f} km
• Target: <10 km
• Gap: {best_weight_result['position_rms'] - 10:.1f} km

Immediate Actions:
1. {'✓ COMPLETE' if best_weight_result['position_rms'] < 10.0 else '✗ CONTINUE'} Position weight tuning
2. {'✓ COMPLETE' if init_results and init_results[0] < 10.0 else '✗ IMPLEMENT'} Better initialization
3. {'✓ COMPLETE' if extended_results and extended_results[0] < 10.0 else '✗ TEST'} Longer observation arcs

Advanced Strategies:
1. Adaptive position weighting
2. Multiple restart optimization
3. Ensemble methods
4. Advanced element bounds
5. Physics-informed initialization

Target Achievement:
• <10 km position RMS: {'✓ ACHIEVED' if best_weight_result['position_rms'] < 10.0 else '✗ NOT ACHIEVED'}
• <5 arcsec measurement RMS: {'✓ ACHIEVED' if best_weight_result['measurement_rms'] < 5.0 else '✗ NOT ACHIEVED'}

Recommendation:
{'✓ PROCEED' if best_weight_result['position_rms'] < 10.0 else '✗ CONTINUE IMPROVEMENT'} with current approach
"""
    
    ax.text(0.05, 0.95, next_steps_text, transform=ax.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('data/position_error_improvement.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Improvement analysis saved to: data/position_error_improvement.png")
    
    return weight_results, init_results, extended_results

if __name__ == "__main__":
    create_improvement_analysis()
