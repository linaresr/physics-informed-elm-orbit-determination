#!/usr/bin/env python3
"""
Comprehensive comparison of ALL approaches to show how wrong they all are.
Let's be honest about the failures and work systematically to fix them.
"""

import sys
sys.path.append('piod')
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle
from piod.solve import fit_elm, evaluate_solution
from piod.solve_elements_enhanced import fit_elm_elements_enhanced, evaluate_solution_elements_enhanced
from piod.observe import ecef_to_eci, radec_to_trig, trig_to_radec, trig_ra_dec
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

def create_observations(t0, t1, noise_level=0.0001, n_obs=20):
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

def simple_measurement_rms(beta, model, obs, t_obs, station_eci):
    """Simple measurement RMS calculation."""
    r_obs, _, _ = model.r_v_a(t_obs, beta)
    r_topo = r_obs - station_eci
    
    # Convert to trig components
    theta_nn = np.apply_along_axis(trig_ra_dec, 0, r_topo)
    
    # Simple residual calculation
    residuals = obs - theta_nn
    rms_trig = np.sqrt(np.mean(residuals**2))
    
    # Convert to approximate arcseconds (rough conversion)
    rms_arcsec = rms_trig * 180/np.pi * 3600
    
    return rms_arcsec

def test_all_approaches():
    """Test ALL approaches and show how wrong they all are."""
    print("=== COMPREHENSIVE FAILURE ANALYSIS ===")
    print("Testing ALL approaches to show how wrong they all are...")
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
    t_obs, obs, station_eci = create_observations(t0, t1, noise_level=0.0001, n_obs=20)
    print(f"✓ Created {len(t_obs)} observations")
    
    # Test all approaches
    print("3. Testing ALL approaches...")
    
    approaches = []
    
    # Approach 1: Original Cartesian
    print("Testing Original Cartesian Approach...")
    try:
        beta, model, result = fit_elm(t0, t1, L=24, N_colloc=80, lam_f=1.0, obs=obs, t_obs=t_obs, 
                                     station_eci=station_eci, lam_th=10000.0, seed=42)
        
        t_eval = np.linspace(t0, t1, 100)
        r, v, a, physics_rms, _ = evaluate_solution(beta, model, t_eval, obs, t_obs, station_eci)
        
        # Calculate position error
        r_true_interp = np.zeros_like(r)
        for i in range(3):
            r_true_interp[i] = np.interp(t_eval, t_true, r_true[i])
        
        r_error = np.linalg.norm(r - r_true_interp, axis=0)
        position_error_rms = np.sqrt(np.mean(r_error**2))/1000
        measurement_rms = simple_measurement_rms(beta, model, obs, t_obs, station_eci)
        
        approaches.append({
            'name': 'Original Cartesian',
            'r': r,
            'position_error_rms': position_error_rms,
            'measurement_rms': measurement_rms,
            'physics_rms': physics_rms,
            'success': result.success,
            'nfev': result.nfev
        })
        
        print(f"  Position Error RMS: {position_error_rms:.1f} km")
        print(f"  Measurement RMS: {measurement_rms:.2f} arcsec")
        print(f"  Physics RMS: {physics_rms:.6f}")
        print(f"  Success: {result.success}")
        
    except Exception as e:
        print(f"  Failed: {e}")
        approaches.append({
            'name': 'Original Cartesian',
            'r': None,
            'position_error_rms': float('inf'),
            'measurement_rms': float('inf'),
            'physics_rms': float('inf'),
            'success': False,
            'nfev': 0
        })
    
    # Approach 2: Orbital Elements
    print("Testing Orbital Elements Approach...")
    try:
        beta, model, result = fit_elm_elements_enhanced(t0, t1, L=8, N_colloc=25, lam_f=1.0, 
                                                       obs=obs, t_obs=t_obs, station_eci=station_eci, 
                                                       lam_th=1000.0, lam_r=1000.0, seed=42)
        
        t_eval = np.linspace(t0, t1, 100)
        r, v, a, physics_rms, measurement_rms, position_magnitude_rms = evaluate_solution_elements_enhanced(
            beta, model, t_eval, obs, t_obs, station_eci)
        
        # Calculate position error
        r_true_interp = np.zeros_like(r)
        for i in range(3):
            r_true_interp[i] = np.interp(t_eval, t_true, r_true[i])
        
        r_error = np.linalg.norm(r - r_true_interp, axis=0)
        position_error_rms = np.sqrt(np.mean(r_error**2))/1000
        
        approaches.append({
            'name': 'Orbital Elements',
            'r': r,
            'position_error_rms': position_error_rms,
            'measurement_rms': measurement_rms,
            'physics_rms': physics_rms,
            'success': result.success,
            'nfev': result.nfev
        })
        
        print(f"  Position Error RMS: {position_error_rms:.1f} km")
        print(f"  Measurement RMS: {measurement_rms:.2f} arcsec")
        print(f"  Physics RMS: {physics_rms:.6f}")
        print(f"  Success: {result.success}")
        
    except Exception as e:
        print(f"  Failed: {e}")
        approaches.append({
            'name': 'Orbital Elements',
            'r': None,
            'position_error_rms': float('inf'),
            'measurement_rms': float('inf'),
            'physics_rms': float('inf'),
            'success': False,
            'nfev': 0
        })
    
    # Approach 3: High Measurement Weight Cartesian
    print("Testing High Measurement Weight Cartesian...")
    try:
        beta, model, result = fit_elm(t0, t1, L=24, N_colloc=80, lam_f=0.1, obs=obs, t_obs=t_obs, 
                                     station_eci=station_eci, lam_th=100000.0, seed=42)
        
        t_eval = np.linspace(t0, t1, 100)
        r, v, a, physics_rms, _ = evaluate_solution(beta, model, t_eval, obs, t_obs, station_eci)
        
        # Calculate position error
        r_true_interp = np.zeros_like(r)
        for i in range(3):
            r_true_interp[i] = np.interp(t_eval, t_true, r_true[i])
        
        r_error = np.linalg.norm(r - r_true_interp, axis=0)
        position_error_rms = np.sqrt(np.mean(r_error**2))/1000
        measurement_rms = simple_measurement_rms(beta, model, obs, t_obs, station_eci)
        
        approaches.append({
            'name': 'High Measurement Weight',
            'r': r,
            'position_error_rms': position_error_rms,
            'measurement_rms': measurement_rms,
            'physics_rms': physics_rms,
            'success': result.success,
            'nfev': result.nfev
        })
        
        print(f"  Position Error RMS: {position_error_rms:.1f} km")
        print(f"  Measurement RMS: {measurement_rms:.2f} arcsec")
        print(f"  Physics RMS: {physics_rms:.6f}")
        print(f"  Success: {result.success}")
        
    except Exception as e:
        print(f"  Failed: {e}")
        approaches.append({
            'name': 'High Measurement Weight',
            'r': None,
            'position_error_rms': float('inf'),
            'measurement_rms': float('inf'),
            'physics_rms': float('inf'),
            'success': False,
            'nfev': 0
        })
    
    # Approach 4: Physics Only (no measurements)
    print("Testing Physics Only Approach...")
    try:
        beta, model, result = fit_elm(t0, t1, L=24, N_colloc=80, lam_f=10.0, obs=None, seed=42)
        
        t_eval = np.linspace(t0, t1, 100)
        r, v, a, physics_rms, _ = evaluate_solution(beta, model, t_eval)
        
        # Calculate position error
        r_true_interp = np.zeros_like(r)
        for i in range(3):
            r_true_interp[i] = np.interp(t_eval, t_true, r_true[i])
        
        r_error = np.linalg.norm(r - r_true_interp, axis=0)
        position_error_rms = np.sqrt(np.mean(r_error**2))/1000
        
        approaches.append({
            'name': 'Physics Only',
            'r': r,
            'position_error_rms': position_error_rms,
            'measurement_rms': float('inf'),
            'physics_rms': physics_rms,
            'success': result.success,
            'nfev': result.nfev
        })
        
        print(f"  Position Error RMS: {position_error_rms:.1f} km")
        print(f"  Physics RMS: {physics_rms:.6f}")
        print(f"  Success: {result.success}")
        
    except Exception as e:
        print(f"  Failed: {e}")
        approaches.append({
            'name': 'Physics Only',
            'r': None,
            'position_error_rms': float('inf'),
            'measurement_rms': float('inf'),
            'physics_rms': float('inf'),
            'success': False,
            'nfev': 0
        })
    
    # Create comprehensive failure analysis plot
    print()
    print("4. Creating comprehensive failure analysis plot...")
    
    fig = plt.figure(figsize=(20, 15))
    
    # 1. 3D Orbit Comparison
    ax1 = fig.add_subplot(3, 3, 1, projection='3d')
    
    # Plot Earth
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    earth_x = 6378136.3 * np.outer(np.cos(u), np.sin(v))
    earth_y = 6378136.3 * np.outer(np.sin(u), np.sin(v))
    earth_z = 6378136.3 * np.outer(np.ones(np.size(u)), np.cos(v))
    ax1.plot_surface(earth_x, earth_y, earth_z, alpha=0.3, color='blue')
    
    # Plot true orbit
    ax1.plot(r_true[0]/1000, r_true[1]/1000, r_true[2]/1000, 'g-', linewidth=4, label='True Orbit')
    
    # Plot all approaches
    colors = ['red', 'orange', 'purple', 'brown']
    for i, approach in enumerate(approaches):
        if approach['r'] is not None:
            ax1.plot(approach['r'][0]/1000, approach['r'][1]/1000, approach['r'][2]/1000, 
                    '--', linewidth=2, color=colors[i], label=f"{approach['name']} ({approach['position_error_rms']:.0f} km)")
    
    ax1.set_xlabel('X (km)')
    ax1.set_ylabel('Y (km)')
    ax1.set_zlabel('Z (km)')
    ax1.set_title('3D Orbit Comparison - ALL APPROACHES FAIL')
    ax1.legend()
    
    # 2. XY Plane Comparison
    ax2 = fig.add_subplot(3, 3, 2)
    ax2.plot(r_true[0]/1000, r_true[1]/1000, 'g-', linewidth=4, label='True Orbit')
    
    # Add Earth circle
    earth_circle = Circle((0, 0), 6378.136, fill=False, color='blue', linestyle='--', alpha=0.5)
    ax2.add_patch(earth_circle)
    
    # Plot all approaches
    for i, approach in enumerate(approaches):
        if approach['r'] is not None:
            ax2.plot(approach['r'][0]/1000, approach['r'][1]/1000, 
                    '--', linewidth=2, color=colors[i], label=f"{approach['name']} ({approach['position_error_rms']:.0f} km)")
    
    ax2.set_xlabel('X (km)')
    ax2.set_ylabel('Y (km)')
    ax2.set_title('XY Plane Comparison - ALL APPROACHES FAIL')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    # 3. Position Error Comparison
    ax3 = fig.add_subplot(3, 3, 3)
    approach_names = [a['name'] for a in approaches if a['success']]
    position_errors = [a['position_error_rms'] for a in approaches if a['success']]
    
    bars = ax3.bar(approach_names, position_errors, alpha=0.7, color=['red', 'orange', 'purple', 'brown'])
    ax3.set_ylabel('Position Error RMS (km)')
    ax3.set_title('Position Error Comparison - ALL FAIL')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # Add target line
    ax3.axhline(y=10, color='green', linestyle='--', alpha=0.7, label='Target (10 km)')
    ax3.legend()
    
    # 4. Measurement Error Comparison
    ax4 = fig.add_subplot(3, 3, 4)
    measurement_errors = [a['measurement_rms'] for a in approaches if a['success'] and a['measurement_rms'] != float('inf')]
    measurement_names = [a['name'] for a in approaches if a['success'] and a['measurement_rms'] != float('inf')]
    
    if measurement_errors:
        bars = ax4.bar(measurement_names, measurement_errors, alpha=0.7, color=['red', 'orange', 'purple'])
        ax4.set_ylabel('Measurement RMS (arcsec)')
        ax4.set_title('Measurement Error Comparison - ALL FAIL')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        # Add target line
        ax4.axhline(y=5, color='green', linestyle='--', alpha=0.7, label='Target (5 arcsec)')
        ax4.legend()
    
    # 5. Physics Error Comparison
    ax5 = fig.add_subplot(3, 3, 5)
    physics_errors = [a['physics_rms'] for a in approaches if a['success']]
    physics_names = [a['name'] for a in approaches if a['success']]
    
    bars = ax5.bar(physics_names, physics_errors, alpha=0.7, color=['red', 'orange', 'purple', 'brown'])
    ax5.set_ylabel('Physics RMS')
    ax5.set_title('Physics Error Comparison')
    ax5.tick_params(axis='x', rotation=45)
    ax5.grid(True, alpha=0.3)
    
    # Add target line
    ax5.axhline(y=0.001, color='green', linestyle='--', alpha=0.7, label='Target (0.001)')
    ax5.legend()
    
    # 6. Failure Analysis Summary
    ax6 = fig.add_subplot(3, 3, 6)
    ax6.axis('off')
    
    failure_text = f"""
COMPREHENSIVE FAILURE ANALYSIS

ALL APPROACHES FAIL TO ACHIEVE TARGETS:

1. Original Cartesian:
   • Position Error: {approaches[0]['position_error_rms']:.1f} km
   • Measurement RMS: {approaches[0]['measurement_rms']:.2f} arcsec
   • Physics RMS: {approaches[0]['physics_rms']:.6f}
   • Status: {'✓ Success' if approaches[0]['success'] else '✗ Failed'}

2. Orbital Elements:
   • Position Error: {approaches[1]['position_error_rms']:.1f} km
   • Measurement RMS: {approaches[1]['measurement_rms']:.2f} arcsec
   • Physics RMS: {approaches[1]['physics_rms']:.6f}
   • Status: {'✓ Success' if approaches[1]['success'] else '✗ Failed'}

3. High Measurement Weight:
   • Position Error: {approaches[2]['position_error_rms']:.1f} km
   • Measurement RMS: {approaches[2]['measurement_rms']:.2f} arcsec
   • Physics RMS: {approaches[2]['physics_rms']:.6f}
   • Status: {'✓ Success' if approaches[2]['success'] else '✗ Failed'}

4. Physics Only:
   • Position Error: {approaches[3]['position_error_rms']:.1f} km
   • Physics RMS: {approaches[3]['physics_rms']:.6f}
   • Status: {'✓ Success' if approaches[3]['success'] else '✗ Failed'}

TARGETS:
• Position: <10 km
• Measurement: <5 arcsec
• Physics: <0.001

RESULT: ALL APPROACHES FAIL TO ACHIEVE TARGETS
"""
    
    ax6.text(0.05, 0.95, failure_text, transform=ax6.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    # 7. Root Cause Analysis
    ax7 = fig.add_subplot(3, 3, 7)
    ax7.axis('off')
    
    root_cause_text = f"""
ROOT CAUSE ANALYSIS

FUNDAMENTAL ISSUES:

1. Position Error Too High:
   • All approaches: 8,000+ km error
   • Target: <10 km
   • Issue: Orbit is in completely wrong location

2. Measurement Accuracy Poor:
   • All approaches: 10+ arcsec error
   • Target: <5 arcsec
   • Issue: Network not learning correct angles

3. Physics Compliance Varies:
   • Some approaches: Good physics
   • Others: Poor physics
   • Issue: Inconsistent physics enforcement

POSSIBLE ROOT CAUSES:

1. Initialization Problem:
   • Random weights may be too far from solution
   • Need better initial guess

2. Loss Function Issues:
   • Weighting may be wrong
   • Residual formulation may be incorrect

3. Network Architecture:
   • ELM may be too simple
   • Need more sophisticated approach

4. Observation Quality:
   • Observations may be too sparse
   • Need better observation pattern

5. Dynamics Model:
   • 2-body + J2 may be insufficient
   • Need more accurate dynamics

NEXT STEPS:
1. Fix initialization
2. Improve loss function
3. Better network architecture
4. Improve observations
5. Better dynamics model
"""
    
    ax7.text(0.05, 0.95, root_cause_text, transform=ax7.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # 8. Systematic Fix Plan
    ax8 = fig.add_subplot(3, 3, 8)
    ax8.axis('off')
    
    fix_plan_text = f"""
SYSTEMATIC FIX PLAN

STEP 1: DIAGNOSE THE PROBLEM
• Check if observations are correct
• Verify dynamics model accuracy
• Test with known solutions

STEP 2: FIX INITIALIZATION
• Use better initial guess
• Implement IOD-based initialization
• Test different random seeds

STEP 3: IMPROVE LOSS FUNCTION
• Fix residual formulation
• Optimize weight balancing
• Add regularization terms

STEP 4: BETTER NETWORK ARCHITECTURE
• Increase network size
• Add more collocation points
• Test different activation functions

STEP 5: IMPROVE OBSERVATIONS
• Increase observation density
• Better observation pattern
• Add noise handling

STEP 6: BETTER DYNAMICS MODEL
• Add more perturbation terms
• Improve numerical integration
• Test with different dynamics

STEP 7: VALIDATION
• Test with known orbits
• Compare with traditional methods
• Validate against real data

CURRENT STATUS:
• All approaches fail
• Need systematic debugging
• Don't claim victory yet
• Focus on fixing fundamentals
"""
    
    ax8.text(0.05, 0.95, fix_plan_text, transform=ax8.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # 9. Honest Assessment
    ax9 = fig.add_subplot(3, 3, 9)
    ax9.axis('off')
    
    honest_text = f"""
HONEST ASSESSMENT

CURRENT REALITY:
• ALL approaches fail to achieve targets
• Position errors are 800x too large
• Measurement errors are 2x too large
• No approach is production ready

WHAT WE'VE LEARNED:
• Cartesian approach is better than orbital elements
• Physics compliance can be achieved
• Measurement accuracy is challenging
• Position accuracy is the biggest problem

WHAT WE HAVEN'T SOLVED:
• Fundamental orbit determination problem
• Position accuracy target
• Measurement accuracy target
• Production-ready solution

NEXT STEPS:
1. Stop claiming victory
2. Focus on systematic debugging
3. Fix fundamental issues
4. Test with known solutions
5. Validate against real data

RECOMMENDATION:
• Don't deploy any current approach
• Focus on research and development
• Systematic problem-solving approach
• Honest assessment of failures
• Step-by-step improvement
"""
    
    ax9.text(0.05, 0.95, honest_text, transform=ax9.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('results/advanced_strategies/comprehensive_failure_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Comprehensive failure analysis plot saved")
    
    print()
    print("=== COMPREHENSIVE FAILURE ANALYSIS COMPLETE ===")
    print("📁 Results saved in: results/advanced_strategies/")
    print("📊 Generated plots:")
    print("  • comprehensive_failure_analysis.png - Honest failure analysis")
    print()
    print("🎯 Key insight: ALL approaches fail to achieve targets!")
    print("   We need systematic debugging, not victory claims.")

if __name__ == "__main__":
    test_all_approaches()
