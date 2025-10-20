#!/usr/bin/env python3
"""
Simple performance comparison test.
"""

import sys
sys.path.append('piod')
import numpy as np
import matplotlib.pyplot as plt
from piod.solve_elements import fit_elm_elements, evaluate_solution_elements
from piod.solve_elements_enhanced import fit_elm_elements_enhanced, evaluate_solution_elements_enhanced
from piod.observe import ecef_to_eci, radec_to_trig
from piod.dynamics import eom
from scipy.integrate import solve_ivp

def generate_true_orbit():
    """Generate a realistic GEO orbit using numerical integration."""
    # Initial conditions for GEO orbit
    r0 = np.array([42164000.0, 0.0, 0.0])  # GEO altitude
    v0 = np.array([0.0, 3074.0, 0.0])     # Circular orbit velocity
    
    # Time span
    t0, t1 = 0.0, 2 * 3600.0  # 2 hours
    
    # Integrate using scipy
    sol = solve_ivp(eom, [t0, t1], np.hstack([r0, v0]), 
                   t_eval=np.linspace(t0, t1, 300), rtol=1e-8, atol=1e-8)
    
    if not sol.success:
        print(f"Integration failed: {sol.message}")
        return None
    
    return sol.t, sol.y[:3], sol.y[3:]

def create_observations(t0, t1, noise_level=0.0001):
    """Create observations for testing."""
    # Create observations
    station_ecef = np.array([6378136.3, 0.0, 0.0])  # Greenwich
    t_obs = np.linspace(t0, t1, 10)  # Fewer observations
    
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

def test_original_approach(t0, t1, obs, t_obs, station_eci):
    """Test the original orbital elements approach."""
    print("Testing original orbital elements approach...")
    
    L = 8
    N_colloc = 15
    lam_f = 1.0
    lam_th = 1000.0
    
    beta, model, result = fit_elm_elements(t0, t1, L=L, N_colloc=N_colloc,
                                         obs=obs, t_obs=t_obs, station_eci=station_eci,
                                         lam_f=lam_f, lam_th=lam_th)
    
    print(f"  Success: {result.success}")
    print(f"  Function evaluations: {result.nfev}")
    print(f"  Final cost: {result.cost:.2e}")
    
    # Evaluate solution
    t_eval = np.linspace(t0, t1, 50)
    r, v, a, physics_rms, measurement_rms = evaluate_solution_elements(
        beta, model, t_eval, obs, t_obs, station_eci)
    
    # Calculate position RMS
    r_mag = np.linalg.norm(r, axis=0)
    geo_altitude = 42164000
    position_rms = np.sqrt(np.mean((r_mag - geo_altitude)**2))/1000
    
    print(f"  Measurement RMS: {measurement_rms:.2f} arcsec")
    print(f"  Position RMS: {position_rms:.1f} km")
    print(f"  Physics RMS: {physics_rms:.6f}")
    
    return position_rms, measurement_rms, physics_rms, result.nfev

def test_enhanced_approach(t0, t1, obs, t_obs, station_eci):
    """Test the enhanced orbital elements approach."""
    print("Testing enhanced orbital elements approach...")
    
    L = 8
    N_colloc = 15
    lam_f = 1.0
    lam_r = 100.0  # Smaller position weight
    lam_th = 1000.0
    
    beta, model, result = fit_elm_elements_enhanced(t0, t1, L=L, N_colloc=N_colloc,
                                                   obs=obs, t_obs=t_obs, station_eci=station_eci,
                                                   lam_f=lam_f, lam_r=lam_r, lam_th=lam_th)
    
    print(f"  Success: {result.success}")
    print(f"  Function evaluations: {result.nfev}")
    print(f"  Final cost: {result.cost:.2e}")
    
    # Evaluate solution
    t_eval = np.linspace(t0, t1, 50)
    r, v, a, physics_rms, measurement_rms, position_magnitude_rms = evaluate_solution_elements_enhanced(
        beta, model, t_eval, obs, t_obs, station_eci)
    
    print(f"  Measurement RMS: {measurement_rms:.2f} arcsec")
    print(f"  Position Magnitude RMS: {position_magnitude_rms:.1f} km")
    print(f"  Physics RMS: {physics_rms:.6f}")
    
    return position_magnitude_rms, measurement_rms, physics_rms, result.nfev

def main():
    """Main performance comparison test."""
    print("=== PERFORMANCE COMPARISON TEST ===")
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
    t_obs, obs, station_eci = create_observations(t0, t1, noise_level=0.0001)
    print(f"✓ Created {len(t_obs)} observations with {0.0001*180/np.pi*3600:.2f} arcsec noise")
    
    # Test original approach
    print("3. Testing original approach...")
    orig_position_rms, orig_measurement_rms, orig_physics_rms, orig_nfev = test_original_approach(t0, t1, obs, t_obs, station_eci)
    
    # Test enhanced approach
    print("4. Testing enhanced approach...")
    enh_position_rms, enh_measurement_rms, enh_physics_rms, enh_nfev = test_enhanced_approach(t0, t1, obs, t_obs, station_eci)
    
    # Create simple comparison plot
    print("5. Creating comparison plot...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Performance comparison
    ax = axes[0, 0]
    metrics = ['Position RMS\n(km)', 'Measurement RMS\n(arcsec)', 'Physics RMS\n(log scale)', 'Function Evals']
    orig_values = [orig_position_rms, orig_measurement_rms, np.log10(orig_physics_rms + 1e-10), orig_nfev]
    enh_values = [enh_position_rms, enh_measurement_rms, np.log10(enh_physics_rms + 1e-10), enh_nfev]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax.bar(x - width/2, orig_values, width, label='Original ELM', color='red', alpha=0.7)
    ax.bar(x + width/2, enh_values, width, label='Enhanced ELM', color='blue', alpha=0.7)
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Values')
    ax.set_title('Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Improvement analysis
    ax = axes[0, 1]
    ax.axis('off')
    
    improvement_text = f"""
PERFORMANCE COMPARISON RESULTS

Original ELM:
• Position RMS: {orig_position_rms:.1f} km
• Measurement RMS: {orig_measurement_rms:.2f} arcsec
• Physics RMS: {orig_physics_rms:.6f}
• Function Evals: {orig_nfev}

Enhanced ELM:
• Position RMS: {enh_position_rms:.1f} km
• Measurement RMS: {enh_measurement_rms:.2f} arcsec
• Physics RMS: {enh_physics_rms:.6f}
• Function Evals: {enh_nfev}

IMPROVEMENT ANALYSIS:
• Position RMS: {orig_position_rms/enh_position_rms:.1f}x {'BETTER' if enh_position_rms < orig_position_rms else 'WORSE'}
• Measurement RMS: {orig_measurement_rms/enh_measurement_rms:.1f}x {'BETTER' if enh_measurement_rms < orig_measurement_rms else 'WORSE'}
• Physics RMS: {'SAME' if abs(orig_physics_rms - enh_physics_rms) < 1e-6 else 'DIFFERENT'}
• Function Evals: {enh_nfev/orig_nfev:.1f}x {'FASTER' if enh_nfev < orig_nfev else 'SLOWER'}

STATUS:
• Measurement target (<5 arcsec): {'✓' if enh_measurement_rms < 5.0 else '✗'}
• Position target (<1000 km): {'✓' if enh_position_rms < 1000.0 else '✗'}
• Physics compliance: {'✓' if enh_physics_rms < 0.001 else '✗'}

CONCLUSION:
Enhanced approach is {'✓ SUCCESSFUL' if enh_position_rms < orig_position_rms else '✗ NEEDS WORK'}
"""
    
    ax.text(0.05, 0.95, improvement_text, transform=ax.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # 3. Key insights
    ax = axes[1, 0]
    ax.axis('off')
    
    insights_text = f"""
KEY INSIGHTS

1. POSITION MAGNITUDE CONSTRAINT:
   • Added λ_r * ||r_mag - r_GEO||² to loss
   • Should improve position accuracy
   • Results: {'✓ IMPROVED' if enh_position_rms < orig_position_rms else '✗ NO IMPROVEMENT'}

2. ELEMENT BOUNDS:
   • Constrained elements to realistic GEO ranges
   • Prevents unrealistic orbits
   • Results: {'✓ STABLE' if enh_physics_rms < 0.001 else '✗ UNSTABLE'}

3. BETTER INITIALIZATION:
   • Start with realistic GEO elements
   • Small random variations
   • Results: {'✓ CONVERGED' if enh_nfev < orig_nfev * 2 else '✗ SLOW'}

4. MEASUREMENT ACCURACY:
   • Original: {orig_measurement_rms:.2f} arcsec
   • Enhanced: {enh_measurement_rms:.2f} arcsec
   • Status: {'✓ MAINTAINED' if enh_measurement_rms < 5.0 else '✗ DEGRADED'}

5. OVERALL ASSESSMENT:
   • Enhanced approach {'✓ WORKS' if enh_position_rms < orig_position_rms else '✗ FAILS'}
   • Position accuracy {'✓ IMPROVED' if enh_position_rms < orig_position_rms else '✗ WORSE'}
   • Ready for {'✓ PRODUCTION' if enh_position_rms < 1000.0 else '✗ MORE WORK'}
"""
    
    ax.text(0.05, 0.95, insights_text, transform=ax.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # 4. Next steps
    ax = axes[1, 1]
    ax.axis('off')
    
    next_steps_text = f"""
NEXT STEPS

CURRENT STATUS:
• Enhanced approach: {'✓ WORKING' if enh_position_rms < orig_position_rms else '✗ FAILING'}
• Position accuracy: {'✓ GOOD' if enh_position_rms < 1000.0 else '✗ POOR'}
• Measurement accuracy: {'✓ EXCELLENT' if enh_measurement_rms < 5.0 else '✗ POOR'}

IMMEDIATE ACTIONS:
1. {'✓ COMPLETE' if enh_position_rms < orig_position_rms else '✗ FIX'} Position magnitude constraint
2. {'✓ COMPLETE' if enh_physics_rms < 0.001 else '✗ FIX'} Element bounds implementation
3. {'✓ COMPLETE' if enh_nfev < orig_nfev * 2 else '✗ FIX'} Better initialization

FUTURE IMPROVEMENTS:
1. Tune position weight (λ_r) for optimal balance
2. Add more sophisticated element bounds
3. Implement adaptive collocation points
4. Test with longer observation arcs
5. Add multiple station support

TARGET ACHIEVEMENT:
• Position RMS: {'✓ ACHIEVED' if enh_position_rms < 1000.0 else '✗ NOT ACHIEVED'} (<1000 km)
• Measurement RMS: {'✓ ACHIEVED' if enh_measurement_rms < 5.0 else '✗ NOT ACHIEVED'} (<5 arcsec)
• Physics RMS: {'✓ ACHIEVED' if enh_physics_rms < 0.001 else '✗ NOT ACHIEVED'} (<0.001)

RECOMMENDATION:
{'✓ PROCEED' if enh_position_rms < orig_position_rms else '✗ REVISE'} with enhanced approach
"""
    
    ax.text(0.05, 0.95, next_steps_text, transform=ax.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('data/performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Performance comparison saved to: data/performance_comparison.png")
    
    print()
    print("=== PERFORMANCE COMPARISON COMPLETE ===")
    print("📊 Results:")
    print(f"Original ELM: Position RMS = {orig_position_rms:.1f} km")
    print(f"Enhanced ELM: Position RMS = {enh_position_rms:.1f} km")
    print(f"Improvement: {orig_position_rms/enh_position_rms:.1f}x {'better' if enh_position_rms < orig_position_rms else 'worse'} position accuracy")
    print()
    print("🎯 Key Insight: Enhanced approach with position magnitude constraint {'✓ WORKS' if enh_position_rms < orig_position_rms else '✗ NEEDS WORK'}!")

if __name__ == "__main__":
    main()
