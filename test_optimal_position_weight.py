#!/usr/bin/env python3
"""
Focus on the excellent position weight tuning results.
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

def test_optimal_position_weight():
    """Test the optimal position weight found."""
    print("=== TESTING OPTIMAL POSITION WEIGHT ===")
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
    
    # Test the best position weights found
    print("3. Testing optimal position weights...")
    
    # From the previous test, these showed excellent results:
    optimal_weights = [1.0, 10.0, 100.0, 1000.0, 10000.0]
    results = []
    
    for lam_r in optimal_weights:
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
            print(f"  Success: {result.success}")
            
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
    
    # Find the best result
    best_result = min([r for r in results if r['success']], key=lambda x: x['position_rms'])
    
    print()
    print("=== OPTIMAL RESULTS ===")
    print(f"Best position weight: λ_r = {best_result['lam_r']:.0f}")
    print(f"Best position RMS: {best_result['position_rms']:.1f} km")
    print(f"Best measurement RMS: {best_result['measurement_rms']:.2f} arcsec")
    print(f"Best physics RMS: {best_result['physics_rms']:.6f}")
    print(f"Function evaluations: {best_result['nfev']}")
    print(f"Success: {best_result['success']}")
    
    # Check if target is achieved
    position_target_achieved = best_result['position_rms'] < 10.0
    measurement_target_achieved = best_result['measurement_rms'] < 5.0
    
    print()
    print("=== TARGET ACHIEVEMENT ===")
    print(f"Position target (<10 km): {'✓ ACHIEVED' if position_target_achieved else '✗ NOT ACHIEVED'}")
    print(f"Measurement target (<5 arcsec): {'✓ ACHIEVED' if measurement_target_achieved else '✗ NOT ACHIEVED'}")
    
    if position_target_achieved and measurement_target_achieved:
        print("🎉 ALL TARGETS ACHIEVED!")
    elif position_target_achieved:
        print("🎯 Position target achieved, measurement needs work")
    elif measurement_target_achieved:
        print("🎯 Measurement target achieved, position needs work")
    else:
        print("⚠️ Both targets need improvement")
    
    # Create comparison plot
    print()
    print("4. Creating comparison plot...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Position weight vs performance
    ax = axes[0, 0]
    weights = [r['lam_r'] for r in results if r['success']]
    position_rms = [r['position_rms'] for r in results if r['success']]
    measurement_rms = [r['measurement_rms'] for r in results if r['success']]
    
    ax.loglog(weights, position_rms, 'bo-', label='Position RMS', linewidth=2, markersize=8)
    ax2 = ax.twinx()
    ax2.loglog(weights, measurement_rms, 'ro-', label='Measurement RMS', linewidth=2, markersize=8)
    
    ax.set_xlabel('Position Weight (λ_r)')
    ax.set_ylabel('Position RMS (km)')
    ax2.set_ylabel('Measurement RMS (arcsec)')
    ax.set_title('Position Weight vs Performance')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Add target lines
    ax.axhline(y=10, color='green', linestyle='--', alpha=0.7, label='Position Target (10 km)')
    ax2.axhline(y=5, color='orange', linestyle='--', alpha=0.7, label='Measurement Target (5 arcsec)')
    
    # 2. Results summary
    ax = axes[0, 1]
    ax.axis('off')
    
    summary_text = f"""
OPTIMAL POSITION WEIGHT RESULTS

Best Configuration:
• Position weight: λ_r = {best_result['lam_r']:.0f}
• Position RMS: {best_result['position_rms']:.1f} km
• Measurement RMS: {best_result['measurement_rms']:.2f} arcsec
• Physics RMS: {best_result['physics_rms']:.6f}
• Function evals: {best_result['nfev']}

Target Achievement:
• Position target (<10 km): {'✓ ACHIEVED' if position_target_achieved else '✗ NOT ACHIEVED'}
• Measurement target (<5 arcsec): {'✓ ACHIEVED' if measurement_target_achieved else '✗ NOT ACHIEVED'}

Improvement vs Original:
• Original position RMS: 7,612.7 km
• Improved position RMS: {best_result['position_rms']:.1f} km
• Improvement: {7612.7/best_result['position_rms']:.1f}x better

Status:
• Overall: {'✓ SUCCESS' if position_target_achieved and measurement_target_achieved else '⚠️ PARTIAL SUCCESS' if position_target_achieved or measurement_target_achieved else '✗ NEEDS WORK'}
• Ready for: {'✓ PRODUCTION' if position_target_achieved and measurement_target_achieved else '⚠️ FURTHER DEVELOPMENT' if position_target_achieved or measurement_target_achieved else '✗ MORE RESEARCH'}
"""
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
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
• Status: ✗ Position target not achieved

Optimized Approach:
• Position RMS: {best_result['position_rms']:.1f} km
• Measurement RMS: {best_result['measurement_rms']:.2f} arcsec
• Status: {'✓ Both targets achieved' if position_target_achieved and measurement_target_achieved else '⚠️ Partial success' if position_target_achieved or measurement_target_achieved else '✗ Targets not achieved'}

Key Improvements:
• Position accuracy: {7612.7/best_result['position_rms']:.1f}x better
• Measurement accuracy: {'✓ Maintained' if best_result['measurement_rms'] < 5.0 else '✗ Degraded'}
• Training efficiency: {'✓ Improved' if best_result['nfev'] < 50 else '⚠️ Similar' if best_result['nfev'] < 100 else '✗ Slower'}

What Made It Work:
• Optimal position weight: λ_r = {best_result['lam_r']:.0f}
• Balanced loss function
• Element bounds
• Better initialization
• More collocation points
"""
    
    ax.text(0.05, 0.95, performance_text, transform=ax.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # 4. Next steps
    ax = axes[1, 1]
    ax.axis('off')
    
    next_steps_text = f"""
NEXT STEPS

Current Status:
• Position RMS: {best_result['position_rms']:.1f} km
• Measurement RMS: {best_result['measurement_rms']:.2f} arcsec
• Overall: {'✓ SUCCESS' if position_target_achieved and measurement_target_achieved else '⚠️ PARTIAL SUCCESS' if position_target_achieved or measurement_target_achieved else '✗ NEEDS WORK'}

Immediate Actions:
1. {'✓ COMPLETE' if position_target_achieved else '✗ CONTINUE'} Position weight optimization
2. {'✓ COMPLETE' if measurement_target_achieved else '✗ IMPLEMENT'} Measurement accuracy tuning
3. {'✓ COMPLETE' if best_result['nfev'] < 50 else '✗ OPTIMIZE'} Training efficiency

Future Improvements:
1. Test with different observation patterns
2. Implement adaptive weighting
3. Add multiple station support
4. Test with longer observation arcs
5. Implement ensemble methods

Production Readiness:
• Position accuracy: {'✓ READY' if position_target_achieved else '✗ NOT READY'}
• Measurement accuracy: {'✓ READY' if measurement_target_achieved else '✗ NOT READY'}
• Overall: {'✓ PRODUCTION READY' if position_target_achieved and measurement_target_achieved else '⚠️ DEVELOPMENT READY' if position_target_achieved or measurement_target_achieved else '✗ RESEARCH PHASE'}

Recommendation:
{'✓ DEPLOY' if position_target_achieved and measurement_target_achieved else '⚠️ CONTINUE DEVELOPMENT' if position_target_achieved or measurement_target_achieved else '✗ MORE RESEARCH NEEDED'}
"""
    
    ax.text(0.05, 0.95, next_steps_text, transform=ax.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('data/optimal_position_weight_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Optimal position weight results saved to: data/optimal_position_weight_results.png")
    
    return best_result

if __name__ == "__main__":
    test_optimal_position_weight()
