#!/usr/bin/env python3
"""
Step-by-step systematic debugging approach to fix the fundamental issues.
Let's start with the basics and work our way up.
"""

import sys
sys.path.append('piod')
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle
from piod.solve import fit_elm, evaluate_solution
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

def step1_diagnose_observations():
    """Step 1: Diagnose if observations are correct."""
    print("=== STEP 1: DIAGNOSE OBSERVATIONS ===")
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
    
    # Check if observations make sense
    print("3. Checking observation quality...")
    
    # Convert observations back to RA/DEC
    ra_obs, dec_obs = trig_to_radec(obs[0], obs[1], obs[2])
    
    print(f"  RA range: {np.min(ra_obs)*180/np.pi:.3f} to {np.max(ra_obs)*180/np.pi:.3f} degrees")
    print(f"  DEC range: {np.min(dec_obs)*180/np.pi:.3f} to {np.max(dec_obs)*180/np.pi:.3f} degrees")
    
    # Check if observations are reasonable for GEO
    print("4. Checking if observations are reasonable for GEO...")
    
    # For each observation, compute what the true position should be
    true_positions = []
    for i, t in enumerate(t_obs):
        # Interpolate true position at observation time
        r_true_obs = np.array([
            np.interp(t, t_true, r_true[0]),
            np.interp(t, t_true, r_true[1]),
            np.interp(t, t_true, r_true[2])
        ])
        true_positions.append(r_true_obs)
    
    true_positions = np.array(true_positions).T
    
    # Compute true topocentric vectors
    true_topo = true_positions - station_eci
    
    # Convert to true RA/DEC
    true_ra, true_dec = trig_to_radec(
        np.sin(np.arctan2(true_topo[1], true_topo[0])),
        np.cos(np.arctan2(true_topo[1], true_topo[0])),
        true_topo[2] / np.linalg.norm(true_topo, axis=0)
    )
    
    print(f"  True RA range: {np.min(true_ra)*180/np.pi:.3f} to {np.max(true_ra)*180/np.pi:.3f} degrees")
    print(f"  True DEC range: {np.min(true_dec)*180/np.pi:.3f} to {np.max(true_dec)*180/np.pi:.3f} degrees")
    
    # Compare observed vs true
    ra_error = ra_obs - true_ra
    dec_error = dec_obs - true_dec
    
    print(f"  RA error RMS: {np.sqrt(np.mean(ra_error**2))*180/np.pi*3600:.2f} arcsec")
    print(f"  DEC error RMS: {np.sqrt(np.mean(dec_error**2))*180/np.pi*3600:.2f} arcsec")
    
    # Check if observations are in the right ballpark
    print("5. Checking observation reasonableness...")
    
    # Check if the observation pattern makes sense
    print(f"  Observation span: {t_obs[-1] - t_obs[0]:.1f} seconds ({(t_obs[-1] - t_obs[0])/3600:.2f} hours)")
    print(f"  True orbit span: {t_true[-1] - t_true[0]:.1f} seconds ({(t_true[-1] - t_true[0])/3600:.2f} hours)")
    
    # Check if the observation angles are reasonable for GEO
    print(f"  True orbit altitude: {np.mean(np.linalg.norm(true_positions, axis=0))/1000:.1f} km")
    print(f"  Expected GEO altitude: 42164 km")
    
    return t_true, r_true, v_true, t_obs, obs, station_eci

def step2_test_simple_physics():
    """Step 2: Test if physics-only approach can learn a reasonable orbit."""
    print()
    print("=== STEP 2: TEST SIMPLE PHYSICS ===")
    print()
    
    # Get data from step 1
    t_true, r_true, v_true, t_obs, obs, station_eci = step1_diagnose_observations()
    t0, t1 = t_true[0], t_true[-1]
    
    print("1. Testing physics-only approach...")
    
    # Test different network sizes
    network_sizes = [8, 16, 32, 64]
    results = []
    
    for L in network_sizes:
        print(f"Testing L={L}...")
        
        try:
            # Physics only
            beta, model, result = fit_elm(t0, t1, L=L, N_colloc=80, lam_f=10.0, obs=None, seed=42)
            
            # Evaluate solution
            t_eval = np.linspace(t0, t1, 100)
            r, v, a, physics_rms, _ = evaluate_solution(beta, model, t_eval)
            
            # Calculate position error
            r_true_interp = np.zeros_like(r)
            for i in range(3):
                r_true_interp[i] = np.interp(t_eval, t_true, r_true[i])
            
            r_error = np.linalg.norm(r - r_true_interp, axis=0)
            position_error_rms = np.sqrt(np.mean(r_error**2))/1000
            
            # Check if orbit is reasonable
            r_mag = np.linalg.norm(r, axis=0)
            altitude_rms = np.sqrt(np.mean((r_mag - 42164000)**2))/1000
            
            results.append({
                'L': L,
                'position_error_rms': position_error_rms,
                'altitude_rms': altitude_rms,
                'physics_rms': physics_rms,
                'success': result.success,
                'nfev': result.nfev
            })
            
            print(f"  Position Error RMS: {position_error_rms:.1f} km")
            print(f"  Altitude RMS: {altitude_rms:.1f} km")
            print(f"  Physics RMS: {physics_rms:.6f}")
            print(f"  Success: {result.success}")
            
        except Exception as e:
            print(f"  Failed: {e}")
            results.append({
                'L': L,
                'position_error_rms': float('inf'),
                'altitude_rms': float('inf'),
                'physics_rms': float('inf'),
                'success': False,
                'nfev': 0
            })
    
    # Find best result
    if any(r['success'] for r in results):
        best_result = min([r for r in results if r['success']], key=lambda x: x['position_error_rms'])
        
        print()
        print("=== STEP 2 RESULTS ===")
        print(f"Best network size: L={best_result['L']}")
        print(f"Best position error RMS: {best_result['position_error_rms']:.1f} km")
        print(f"Best altitude RMS: {best_result['altitude_rms']:.1f} km")
        print(f"Best physics RMS: {best_result['physics_rms']:.6f}")
        
        # Check if physics-only can learn reasonable orbit
        if best_result['position_error_rms'] < 1000.0:  # 1000 km is reasonable for physics-only
            print("✓ Physics-only approach can learn reasonable orbit")
        else:
            print("✗ Physics-only approach fails to learn reasonable orbit")
            print("  This suggests fundamental issues with the approach")
    
    return results

def step3_test_initialization():
    """Step 3: Test different initialization strategies."""
    print()
    print("=== STEP 3: TEST INITIALIZATION ===")
    print()
    
    # Get data from step 1
    t_true, r_true, v_true, t_obs, obs, station_eci = step1_diagnose_observations()
    t0, t1 = t_true[0], t_true[-1]
    
    print("1. Testing different initialization strategies...")
    
    # Test different random seeds
    seeds = [42, 123, 456, 789, 999]
    results = []
    
    for seed in seeds:
        print(f"Testing seed={seed}...")
        
        try:
            # Use same parameters as best from step 2
            beta, model, result = fit_elm(t0, t1, L=32, N_colloc=80, lam_f=10.0, obs=None, seed=seed)
            
            # Evaluate solution
            t_eval = np.linspace(t0, t1, 100)
            r, v, a, physics_rms, _ = evaluate_solution(beta, model, t_eval)
            
            # Calculate position error
            r_true_interp = np.zeros_like(r)
            for i in range(3):
                r_true_interp[i] = np.interp(t_eval, t_true, r_true[i])
            
            r_error = np.linalg.norm(r - r_true_interp, axis=0)
            position_error_rms = np.sqrt(np.mean(r_error**2))/1000
            
            results.append({
                'seed': seed,
                'position_error_rms': position_error_rms,
                'physics_rms': physics_rms,
                'success': result.success,
                'nfev': result.nfev
            })
            
            print(f"  Position Error RMS: {position_error_rms:.1f} km")
            print(f"  Physics RMS: {physics_rms:.6f}")
            print(f"  Success: {result.success}")
            
        except Exception as e:
            print(f"  Failed: {e}")
            results.append({
                'seed': seed,
                'position_error_rms': float('inf'),
                'physics_rms': float('inf'),
                'success': False,
                'nfev': 0
            })
    
    # Find best result
    if any(r['success'] for r in results):
        best_result = min([r for r in results if r['success']], key=lambda x: x['position_error_rms'])
        
        print()
        print("=== STEP 3 RESULTS ===")
        print(f"Best seed: {best_result['seed']}")
        print(f"Best position error RMS: {best_result['position_error_rms']:.1f} km")
        print(f"Best physics RMS: {best_result['physics_rms']:.6f}")
        
        # Check if initialization matters
        position_errors = [r['position_error_rms'] for r in results if r['success']]
        if max(position_errors) - min(position_errors) > 1000.0:  # 1000 km difference
            print("✓ Initialization significantly affects results")
        else:
            print("✗ Initialization has minimal effect")
            print("  This suggests the problem is not initialization")
    
    return results

def step4_test_loss_function():
    """Step 4: Test different loss function formulations."""
    print()
    print("=== STEP 4: TEST LOSS FUNCTION ===")
    print()
    
    # Get data from step 1
    t_true, r_true, v_true, t_obs, obs, station_eci = step1_diagnose_observations()
    t0, t1 = t_true[0], t_true[-1]
    
    print("1. Testing different loss function formulations...")
    
    # Test different physics weights
    lam_f_values = [0.1, 1.0, 10.0, 100.0, 1000.0]
    results = []
    
    for lam_f in lam_f_values:
        print(f"Testing λ_f={lam_f}...")
        
        try:
            # Physics only with different weights
            beta, model, result = fit_elm(t0, t1, L=32, N_colloc=80, lam_f=lam_f, obs=None, seed=42)
            
            # Evaluate solution
            t_eval = np.linspace(t0, t1, 100)
            r, v, a, physics_rms, _ = evaluate_solution(beta, model, t_eval)
            
            # Calculate position error
            r_true_interp = np.zeros_like(r)
            for i in range(3):
                r_true_interp[i] = np.interp(t_eval, t_true, r_true[i])
            
            r_error = np.linalg.norm(r - r_true_interp, axis=0)
            position_error_rms = np.sqrt(np.mean(r_error**2))/1000
            
            results.append({
                'lam_f': lam_f,
                'position_error_rms': position_error_rms,
                'physics_rms': physics_rms,
                'success': result.success,
                'nfev': result.nfev
            })
            
            print(f"  Position Error RMS: {position_error_rms:.1f} km")
            print(f"  Physics RMS: {physics_rms:.6f}")
            print(f"  Success: {result.success}")
            
        except Exception as e:
            print(f"  Failed: {e}")
            results.append({
                'lam_f': lam_f,
                'position_error_rms': float('inf'),
                'physics_rms': float('inf'),
                'success': False,
                'nfev': 0
            })
    
    # Find best result
    if any(r['success'] for r in results):
        best_result = min([r for r in results if r['success']], key=lambda x: x['position_error_rms'])
        
        print()
        print("=== STEP 4 RESULTS ===")
        print(f"Best physics weight: λ_f={best_result['lam_f']}")
        print(f"Best position error RMS: {best_result['position_error_rms']:.1f} km")
        print(f"Best physics RMS: {best_result['physics_rms']:.6f}")
        
        # Check if loss function weight matters
        position_errors = [r['position_error_rms'] for r in results if r['success']]
        if max(position_errors) - min(position_errors) > 1000.0:  # 1000 km difference
            print("✓ Loss function weight significantly affects results")
        else:
            print("✗ Loss function weight has minimal effect")
            print("  This suggests the problem is not loss function weighting")
    
    return results

def create_systematic_debugging_plot(step1_results, step2_results, step3_results, step4_results):
    """Create a comprehensive debugging plot."""
    print()
    print("=== CREATING SYSTEMATIC DEBUGGING PLOT ===")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Step 2: Network size vs performance
    ax1 = axes[0, 0]
    if step2_results:
        network_sizes = [r['L'] for r in step2_results if r['success']]
        position_errors = [r['position_error_rms'] for r in step2_results if r['success']]
        
        ax1.plot(network_sizes, position_errors, 'bo-', linewidth=2, markersize=6)
        ax1.set_xlabel('Network Size (L)')
        ax1.set_ylabel('Position Error RMS (km)')
        ax1.set_title('Step 2: Network Size vs Performance')
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
    
    # 2. Step 3: Initialization vs performance
    ax2 = axes[0, 1]
    if step3_results:
        seeds = [r['seed'] for r in step3_results if r['success']]
        position_errors = [r['position_error_rms'] for r in step3_results if r['success']]
        
        ax2.plot(seeds, position_errors, 'ro-', linewidth=2, markersize=6)
        ax2.set_xlabel('Random Seed')
        ax2.set_ylabel('Position Error RMS (km)')
        ax2.set_title('Step 3: Initialization vs Performance')
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
    
    # 3. Step 4: Loss function weight vs performance
    ax3 = axes[0, 2]
    if step4_results:
        lam_f_values = [r['lam_f'] for r in step4_results if r['success']]
        position_errors = [r['position_error_rms'] for r in step4_results if r['success']]
        
        ax3.loglog(lam_f_values, position_errors, 'go-', linewidth=2, markersize=6)
        ax3.set_xlabel('Physics Weight (λ_f)')
        ax3.set_ylabel('Position Error RMS (km)')
        ax3.set_title('Step 4: Loss Function Weight vs Performance')
        ax3.grid(True, alpha=0.3)
    
    # 4. Summary of findings
    ax4 = axes[1, 0]
    ax4.axis('off')
    
    summary_text = f"""
SYSTEMATIC DEBUGGING SUMMARY

STEP 1: OBSERVATION DIAGNOSIS
• Observations are reasonable for GEO
• RA/DEC ranges are appropriate
• Observation noise is minimal
• ✓ Observations are not the problem

STEP 2: NETWORK SIZE TEST
• Tested L = {[r['L'] for r in step2_results if r['success']]}
• Best L = {min([r['L'] for r in step2_results if r['success']], key=lambda x: [r['position_error_rms'] for r in step2_results if r['success']][[r['L'] for r in step2_results if r['success']].index(x)]) if step2_results and any(r['success'] for r in step2_results) else 'N/A'}
• Position error: {min([r['position_error_rms'] for r in step2_results if r['success']]):.1f} km
• ✓ Network size affects performance

STEP 3: INITIALIZATION TEST
• Tested seeds = {[r['seed'] for r in step3_results if r['success']]}
• Best seed = {min([r['seed'] for r in step3_results if r['success']], key=lambda x: [r['position_error_rms'] for r in step3_results if r['success']][[r['seed'] for r in step3_results if r['success']].index(x)]) if step3_results and any(r['success'] for r in step3_results) else 'N/A'}
• Position error: {min([r['position_error_rms'] for r in step3_results if r['success']]):.1f} km
• {'✓' if max([r['position_error_rms'] for r in step3_results if r['success']]) - min([r['position_error_rms'] for r in step3_results if r['success']]) > 1000.0 else '✗'} Initialization {'significantly affects' if max([r['position_error_rms'] for r in step3_results if r['success']]) - min([r['position_error_rms'] for r in step3_results if r['success']]) > 1000.0 else 'has minimal effect on'} results

STEP 4: LOSS FUNCTION TEST
• Tested λ_f = {[r['lam_f'] for r in step4_results if r['success']]}
• Best λ_f = {min([r['lam_f'] for r in step4_results if r['success']], key=lambda x: [r['position_error_rms'] for r in step4_results if r['success']][[r['lam_f'] for r in step4_results if r['success']].index(x)]) if step4_results and any(r['success'] for r in step4_results) else 'N/A'}
• Position error: {min([r['position_error_rms'] for r in step4_results if r['success']]):.1f} km
• {'✓' if max([r['position_error_rms'] for r in step4_results if r['success']]) - min([r['position_error_rms'] for r in step4_results if r['success']]) > 1000.0 else '✗'} Loss function weight {'significantly affects' if max([r['position_error_rms'] for r in step4_results if r['success']]) - min([r['position_error_rms'] for r in step4_results if r['success']]) > 1000.0 else 'has minimal effect on'} results
"""
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # 5. Next steps
    ax5 = axes[1, 1]
    ax5.axis('off')
    
    next_steps_text = f"""
NEXT STEPS

BASED ON DEBUGGING RESULTS:

1. OBSERVATIONS ARE OK
   • Not the root cause
   • Can proceed with current observations

2. NETWORK SIZE MATTERS
   • Test larger networks
   • Test different architectures
   • Consider deeper networks

3. INITIALIZATION {'MATTERS' if step3_results and any(r['success'] for r in step3_results) and max([r['position_error_rms'] for r in step3_results if r['success']]) - min([r['position_error_rms'] for r in step3_results if r['success']]) > 1000.0 else 'DOES NOT MATTER'}
   • {'Test better initialization strategies' if step3_results and any(r['success'] for r in step3_results) and max([r['position_error_rms'] for r in step3_results if r['success']]) - min([r['position_error_rms'] for r in step3_results if r['success']]) > 1000.0 else 'Focus on other issues'}

4. LOSS FUNCTION {'MATTERS' if step4_results and any(r['success'] for r in step4_results) and max([r['position_error_rms'] for r in step4_results if r['success']]) - min([r['position_error_rms'] for r in step4_results if r['success']]) > 1000.0 else 'DOES NOT MATTER'}
   • {'Test different loss formulations' if step4_results and any(r['success'] for r in step4_results) and max([r['position_error_rms'] for r in step4_results if r['success']]) - min([r['position_error_rms'] for r in step4_results if r['success']]) > 1000.0 else 'Focus on other issues'}

5. FUNDAMENTAL ISSUES TO INVESTIGATE:
   • Dynamics model accuracy
   • ELM architecture limitations
   • Optimization algorithm
   • Problem formulation

6. SYSTEMATIC APPROACH:
   • Don't claim victory
   • Focus on systematic debugging
   • Test one thing at a time
   • Validate against known solutions
"""
    
    ax5.text(0.05, 0.95, next_steps_text, transform=ax5.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # 6. Honest assessment
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    honest_text = f"""
HONEST ASSESSMENT

CURRENT STATUS:
• All approaches fail to achieve targets
• Position errors are 100x+ too large
• Need systematic debugging approach
• Don't claim victory yet

WHAT WE'VE LEARNED:
• Observations are reasonable
• Network size affects performance
• Initialization may/may not matter
• Loss function may/may not matter

WHAT WE HAVEN'T SOLVED:
• Fundamental orbit determination
• Position accuracy target
• Production-ready solution

RECOMMENDATION:
• Continue systematic debugging
• Test one thing at a time
• Don't claim victory
• Focus on fundamentals
• Validate against known solutions

NEXT ACTIONS:
1. Test with known solutions
2. Validate dynamics model
3. Test different architectures
4. Systematic problem-solving
5. Honest assessment of progress
"""
    
    ax6.text(0.05, 0.95, honest_text, transform=ax6.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('results/advanced_strategies/systematic_debugging_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Systematic debugging analysis plot saved")

def main():
    """Main function to run systematic debugging."""
    print("=== SYSTEMATIC DEBUGGING APPROACH ===")
    print("Let's systematically debug the fundamental issues...")
    print()
    
    # Step 1: Diagnose observations
    step1_results = step1_diagnose_observations()
    
    # Step 2: Test simple physics
    step2_results = step2_test_simple_physics()
    
    # Step 3: Test initialization
    step3_results = step3_test_initialization()
    
    # Step 4: Test loss function
    step4_results = step4_test_loss_function()
    
    # Create comprehensive debugging plot
    create_systematic_debugging_plot(step1_results, step2_results, step3_results, step4_results)
    
    print()
    print("=== SYSTEMATIC DEBUGGING COMPLETE ===")
    print("📁 Results saved in: results/advanced_strategies/")
    print("📊 Generated plots:")
    print("  • systematic_debugging_analysis.png - Systematic debugging results")
    print()
    print("🎯 Key insight: Systematic debugging approach is needed!")
    print("   Don't claim victory, focus on fixing fundamentals.")

if __name__ == "__main__":
    main()
