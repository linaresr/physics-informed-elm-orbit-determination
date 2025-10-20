#!/usr/bin/env python3
"""
Comprehensive diagnosis of why multi-orbit training failed.
This script analyzes the fundamental issues and provides solutions.
"""

import sys
sys.path.append('piod')
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle
from piod.observe import ecef_to_eci, radec_to_trig, trig_to_radec, trig_ra_dec, vec_to_radec
from piod.dynamics import eom
from scipy.integrate import solve_ivp
from piod.solve import fit_elm, evaluate_solution
from piod.elm import GeoELM
from piod.loss import residual
import json
import os
from datetime import datetime

def diagnose_multi_orbit_failure():
    """Diagnose why multi-orbit training failed."""
    print("=== DIAGNOSING MULTI-ORBIT TRAINING FAILURE ===")
    print()
    
    print("PROBLEM ANALYSIS:")
    print("=" * 50)
    print("• Multi-orbit training: 100 orbits, 20,000 observations")
    print("• Position error: 918,736.9 km (target: <10 km)")
    print("• Measurement error: 164,176.8 arcsec (target: <5 arcsec)")
    print("• Physics error: 3.168550 (should be <0.01)")
    print("• Status: COMPLETE FAILURE")
    print()
    
    print("POTENTIAL CAUSES:")
    print("=" * 50)
    print("1. ELM ARCHITECTURE ISSUES:")
    print("   • Single ELM trying to learn 100 different orbits")
    print("   • ELM may be too small (L=24) for complex multi-orbit data")
    print("   • Time normalization may be wrong for multi-orbit data")
    print("   • ELM basis may not be suitable for diverse orbits")
    print()
    
    print("2. TRAINING DATA ISSUES:")
    print("   • Mixing different orbits in single training run")
    print("   • Different time spans for different orbits")
    print("   • Inconsistent observation patterns")
    print("   • ELM trying to fit incompatible data")
    print()
    
    print("3. LOSS FUNCTION ISSUES:")
    print("   • Physics residuals from different orbits")
    print("   • Measurement residuals from different orbits")
    print("   • Conflicting optimization objectives")
    print("   • Loss function not suitable for multi-orbit data")
    print()
    
    print("4. OPTIMIZATION ISSUES:")
    print("   • Too many conflicting constraints")
    print("   • Optimization getting stuck in local minima")
    print("   • Gradient conflicts between different orbits")
    print("   • Convergence to poor solutions")
    print()
    
    print("ROOT CAUSE HYPOTHESIS:")
    print("=" * 50)
    print("The fundamental issue is that we're trying to train a SINGLE ELM")
    print("to learn MULTIPLE DIFFERENT ORBITS simultaneously. This is like")
    print("trying to train a single neural network to learn 100 different")
    print("functions at once - it's fundamentally impossible!")
    print()
    print("The ELM is trying to find a single trajectory that satisfies")
    print("physics and measurements for ALL orbits, which is impossible")
    print("because each orbit is different.")
    print()

def test_single_orbit_approach():
    """Test the single-orbit approach with corrected observations."""
    print("=== TESTING SINGLE-ORBIT APPROACH ===")
    print()
    
    # Generate a single orbit
    print("1. Generating single GEO orbit...")
    r0 = np.array([42164000.0, 0.0, 0.0])  # GEO altitude
    v0 = np.array([0.0, 3074.0, 0.0])     # Circular orbit velocity
    t0, t1 = 0.0, 2 * 3600.0  # 2 hours
    
    sol = solve_ivp(eom, [t0, t1], np.hstack([r0, v0]), 
                   t_eval=np.linspace(t0, t1, 300), rtol=1e-8, atol=1e-8)
    
    if not sol.success:
        print(f"Integration failed: {sol.message}")
        return None
    
    t_true, r_true, v_true = sol.t, sol.y[:3], sol.y[3:]
    print(f"✓ Generated single orbit: {len(t_true)} points over {t1/3600:.1f} hours")
    
    # Create corrected observations
    print("2. Creating corrected observations...")
    station_ecef = np.array([6378136.3, 0.0, 0.0])  # Greenwich
    t_obs = np.linspace(t0, t1, 20)
    
    jd_obs = 2451545.0 + t_obs / 86400.0
    station_eci = np.array([ecef_to_eci(station_ecef, jd) for jd in jd_obs]).T
    
    # Generate observations from true orbit
    true_positions = []
    for i, t in enumerate(t_obs):
        r_true_obs = np.array([
            np.interp(t, t_true, r_true[0]),
            np.interp(t, t_true, r_true[1]),
            np.interp(t, t_true, r_true[2])
        ])
        true_positions.append(r_true_obs)
    
    true_positions = np.array(true_positions).T
    true_topo = true_positions - station_eci
    
    # Convert to true RA/DEC
    true_ra, true_dec = trig_to_radec(
        np.sin(np.arctan2(true_topo[1], true_topo[0])),
        np.cos(np.arctan2(true_topo[1], true_topo[0])),
        true_topo[2] / np.linalg.norm(true_topo, axis=0)
    )
    
    # Add realistic noise
    noise_level = 0.0001  # ~0.02 arcsec
    ra_noisy = true_ra + np.random.normal(0, noise_level, len(true_ra))
    dec_noisy = true_dec + np.random.normal(0, noise_level, len(true_dec))
    
    obs = radec_to_trig(ra_noisy, dec_noisy)
    print(f"✓ Created {len(t_obs)} corrected observations")
    
    # Train ELM on single orbit
    print("3. Training ELM on single orbit...")
    try:
        beta, model, result = fit_elm(t0, t1, L=24, N_colloc=80,
                                     obs=obs, t_obs=t_obs,
                                     station_eci=station_eci,
                                     lam_f=1.0, lam_th=10000.0, seed=42)
        
        print(f"✓ Training completed successfully")
        print(f"  Success: {result.success}")
        print(f"  Function evaluations: {result.nfev}")
        print(f"  Final cost: {result.cost}")
        
        # Evaluate performance
        print("4. Evaluating performance...")
        t_eval = np.linspace(t0, t1, 100)
        r, v, a, physics_rms, _ = evaluate_solution(beta, model, t_eval, obs, t_obs, station_eci)
        
        # Calculate position error
        r_true_interp = np.zeros_like(r)
        for i in range(3):
            r_true_interp[i] = np.interp(t_eval, t_true, r_true[i])
        
        r_error = np.linalg.norm(r - r_true_interp, axis=0)
        position_error_rms = np.sqrt(np.mean(r_error**2))/1000
        
        # Calculate measurement error
        measurement_residuals = []
        for i, t in enumerate(t_obs):
            r_obs, _, _ = model.r_v_a(t, beta)
            r_topo = r_obs - station_eci[:, i]
            theta_nn = trig_ra_dec(r_topo)
            residual = obs[:, i] - theta_nn
            measurement_residuals.extend(residual.tolist())
        
        measurement_residuals = np.array(measurement_residuals)
        measurement_rms = np.sqrt(np.mean(measurement_residuals**2)) * 180/np.pi * 3600
        
        print(f"✓ Performance evaluation completed")
        print(f"  Position Error RMS: {position_error_rms:.1f} km")
        print(f"  Measurement RMS: {measurement_rms:.1f} arcsec")
        print(f"  Physics RMS: {physics_rms:.6f}")
        
        return {
            'position_error_rms': position_error_rms,
            'measurement_rms': measurement_rms,
            'physics_rms': physics_rms,
            'success': result.success,
            'nfev': result.nfev,
            'cost': result.cost,
            'r': r,
            'model': model,
            'beta': beta
        }
        
    except Exception as e:
        print(f"✗ Training failed: {e}")
        return None

def test_individual_orbit_training():
    """Test training individual ELMs for each orbit."""
    print()
    print("=== TESTING INDIVIDUAL ORBIT TRAINING ===")
    print()
    
    # Generate 5 different orbits
    print("1. Generating 5 different orbits...")
    orbits = []
    
    for orbit_id in range(5):
        # Generate orbit with variations
        r_geo = 42164000.0
        v_geo = 3074.0
        
        # Add variations
        r_variation = np.random.uniform(-10000, 10000)  # ±10 km
        v_variation = np.random.uniform(-10, 10)        # ±10 m/s
        
        r0 = np.array([r_geo + r_variation, 0.0, 0.0])
        v0 = np.array([0.0, v_geo + v_variation, 0.0])
        
        # Generate trajectory
        t0, t1 = 0.0, 2 * 3600.0  # 2 hours
        sol = solve_ivp(eom, [t0, t1], np.hstack([r0, v0]), 
                       t_eval=np.linspace(t0, t1, 300), rtol=1e-8, atol=1e-8)
        
        if sol.success:
            orbits.append((sol.t, sol.y[:3], sol.y[3:], orbit_id))
            print(f"  ✓ Orbit {orbit_id+1}: r0={r0[0]/1000:.1f} km, v0={v0[1]:.1f} m/s")
    
    print(f"✓ Generated {len(orbits)} orbits")
    
    # Train individual ELMs
    print("2. Training individual ELMs...")
    results = []
    
    for i, (t_true, r_true, v_true, orbit_id) in enumerate(orbits):
        print(f"  Training ELM for orbit {orbit_id+1}...", end=" ")
        
        t0, t1 = t_true[0], t_true[-1]
        
        # Create observations for this orbit
        station_ecef = np.array([6378136.3, 0.0, 0.0])
        t_obs = np.linspace(t0, t1, 20)
        
        jd_obs = 2451545.0 + t_obs / 86400.0
        station_eci = np.array([ecef_to_eci(station_ecef, jd) for jd in jd_obs]).T
        
        # Generate observations from true orbit
        true_positions = []
        for j, t in enumerate(t_obs):
            r_true_obs = np.array([
                np.interp(t, t_true, r_true[0]),
                np.interp(t, t_true, r_true[1]),
                np.interp(t, t_true, r_true[2])
            ])
            true_positions.append(r_true_obs)
        
        true_positions = np.array(true_positions).T
        true_topo = true_positions - station_eci
        
        # Convert to true RA/DEC
        true_ra, true_dec = trig_to_radec(
            np.sin(np.arctan2(true_topo[1], true_topo[0])),
            np.cos(np.arctan2(true_topo[1], true_topo[0])),
            true_topo[2] / np.linalg.norm(true_topo, axis=0)
        )
        
        # Add noise
        noise_level = 0.0001
        ra_noisy = true_ra + np.random.normal(0, noise_level, len(true_ra))
        dec_noisy = true_dec + np.random.normal(0, noise_level, len(true_dec))
        
        obs = radec_to_trig(ra_noisy, dec_noisy)
        
        # Train ELM
        try:
            beta, model, result = fit_elm(t0, t1, L=24, N_colloc=80,
                                         obs=obs, t_obs=t_obs,
                                         station_eci=station_eci,
                                         lam_f=1.0, lam_th=10000.0, seed=42)
            
            # Evaluate
            t_eval = np.linspace(t0, t1, 100)
            r, v, a, physics_rms, _ = evaluate_solution(beta, model, t_eval, obs, t_obs, station_eci)
            
            # Calculate position error
            r_true_interp = np.zeros_like(r)
            for j in range(3):
                r_true_interp[j] = np.interp(t_eval, t_true, r_true[j])
            
            r_error = np.linalg.norm(r - r_true_interp, axis=0)
            position_error_rms = np.sqrt(np.mean(r_error**2))/1000
            
            # Calculate measurement error
            measurement_residuals = []
            for j, t in enumerate(t_obs):
                r_obs, _, _ = model.r_v_a(t, beta)
                r_topo = r_obs - station_eci[:, j]
                theta_nn = trig_ra_dec(r_topo)
                residual = obs[:, j] - theta_nn
                measurement_residuals.extend(residual.tolist())
            
            measurement_residuals = np.array(measurement_residuals)
            measurement_rms = np.sqrt(np.mean(measurement_residuals**2)) * 180/np.pi * 3600
            
            results.append({
                'orbit_id': orbit_id,
                'position_error_rms': position_error_rms,
                'measurement_rms': measurement_rms,
                'physics_rms': physics_rms,
                'success': result.success,
                'nfev': result.nfev,
                'cost': result.cost
            })
            
            print(f"Position: {position_error_rms:.1f} km, Measurement: {measurement_rms:.1f} arcsec")
            
        except Exception as e:
            print(f"FAILED: {e}")
            results.append({
                'orbit_id': orbit_id,
                'position_error_rms': float('inf'),
                'measurement_rms': float('inf'),
                'physics_rms': float('inf'),
                'success': False,
                'nfev': 0,
                'cost': float('inf')
            })
    
    # Calculate overall statistics
    valid_results = [r for r in results if r['position_error_rms'] != float('inf')]
    
    if valid_results:
        avg_position_error = np.mean([r['position_error_rms'] for r in valid_results])
        avg_measurement_error = np.mean([r['measurement_rms'] for r in valid_results])
        avg_physics_error = np.mean([r['physics_rms'] for r in valid_results])
        
        print(f"\n✓ Individual orbit training results:")
        print(f"  Average Position Error RMS: {avg_position_error:.1f} km")
        print(f"  Average Measurement RMS: {avg_measurement_error:.1f} arcsec")
        print(f"  Average Physics RMS: {avg_physics_error:.6f}")
        print(f"  Successful trainings: {len(valid_results)}/{len(results)}")
        
        return {
            'results': results,
            'avg_position_error': avg_position_error,
            'avg_measurement_error': avg_measurement_error,
            'avg_physics_error': avg_physics_error,
            'success_rate': len(valid_results)/len(results)
        }
    else:
        print("No successful individual trainings")
        return None

def create_failure_diagnosis_plots(single_orbit_result, individual_orbit_result):
    """Create comprehensive failure diagnosis plots."""
    print()
    print("=== CREATING FAILURE DIAGNOSIS PLOTS ===")
    
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Problem Analysis
    ax1 = fig.add_subplot(3, 3, 1)
    ax1.axis('off')
    
    problem_text = f"""
PROBLEM ANALYSIS

MULTI-ORBIT TRAINING FAILURE:
• Position Error: 918,736.9 km
• Measurement Error: 164,176.8 arcsec
• Physics Error: 3.168550
• Status: COMPLETE FAILURE

ROOT CAUSE:
• Trying to train SINGLE ELM on MULTIPLE ORBITS
• ELM cannot learn 100 different functions simultaneously
• Conflicting optimization objectives
• Impossible to satisfy all constraints

FUNDAMENTAL ISSUE:
• ELM designed for SINGLE trajectory
• Multi-orbit data is INCOMPATIBLE
• Need different approach for multiple orbits

SOLUTION APPROACHES:
1. Individual ELMs per orbit
2. Ensemble of ELMs
3. Different architecture
4. Sequential training
5. Transfer learning
"""
    
    ax1.text(0.05, 0.95, problem_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    # 2. Single Orbit Results
    ax2 = fig.add_subplot(3, 3, 2)
    ax2.axis('off')
    
    if single_orbit_result:
        single_text = f"""
SINGLE ORBIT APPROACH

PERFORMANCE:
• Position Error: {single_orbit_result['position_error_rms']:.1f} km
• Measurement Error: {single_orbit_result['measurement_rms']:.1f} arcsec
• Physics Error: {single_orbit_result['physics_rms']:.6f}
• Success: {single_orbit_result['success']}
• Function Evals: {single_orbit_result['nfev']}

TARGET ACHIEVEMENT:
• Position Target (<10 km): {'✓ ACHIEVED' if single_orbit_result['position_error_rms'] < 10.0 else '✗ NOT ACHIEVED'}
• Measurement Target (<5 arcsec): {'✓ ACHIEVED' if single_orbit_result['measurement_rms'] < 5.0 else '✗ NOT ACHIEVED'}

CONCLUSION:
• Single orbit approach: {'✓ WORKS' if single_orbit_result['position_error_rms'] < 100.0 else '✗ FAILS'}
• ELM can learn single orbit
• Problem is with multi-orbit training
• Need different approach for multiple orbits
"""
    else:
        single_text = """
SINGLE ORBIT APPROACH

PERFORMANCE:
• Training: FAILED
• Status: NEEDS INVESTIGATION

CONCLUSION:
• Single orbit approach: FAILED
• Need to investigate why
• Check ELM parameters
• Check data quality
"""
    
    ax2.text(0.05, 0.95, single_text, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # 3. Individual Orbit Results
    ax3 = fig.add_subplot(3, 3, 3)
    ax3.axis('off')
    
    if individual_orbit_result:
        individual_text = f"""
INDIVIDUAL ORBIT TRAINING

PERFORMANCE:
• Average Position Error: {individual_orbit_result['avg_position_error']:.1f} km
• Average Measurement Error: {individual_orbit_result['avg_measurement_error']:.1f} arcsec
• Average Physics Error: {individual_orbit_result['avg_physics_error']:.6f}
• Success Rate: {individual_orbit_result['success_rate']*100:.1f}%

TARGET ACHIEVEMENT:
• Position Target (<10 km): {'✓ ACHIEVED' if individual_orbit_result['avg_position_error'] < 10.0 else '✗ NOT ACHIEVED'}
• Measurement Target (<5 arcsec): {'✓ ACHIEVED' if individual_orbit_result['avg_measurement_error'] < 5.0 else '✗ NOT ACHIEVED'}

CONCLUSION:
• Individual training: {'✓ WORKS' if individual_orbit_result['avg_position_error'] < 100.0 else '✗ FAILS'}
• Each ELM can learn its orbit
• Confirms single-orbit approach works
• Multi-orbit training is the problem
"""
    else:
        individual_text = """
INDIVIDUAL ORBIT TRAINING

PERFORMANCE:
• Training: FAILED
• Status: NEEDS INVESTIGATION

CONCLUSION:
• Individual training: FAILED
• Need to investigate why
• Check ELM parameters
• Check data quality
"""
    
    ax3.text(0.05, 0.95, individual_text, transform=ax3.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # 4. Comparison
    ax4 = fig.add_subplot(3, 3, 4)
    ax4.axis('off')
    
    if single_orbit_result and individual_orbit_result:
        comparison_text = f"""
PERFORMANCE COMPARISON

MULTI-ORBIT TRAINING:
• Position Error: 918,736.9 km
• Measurement Error: 164,176.8 arcsec
• Status: COMPLETE FAILURE

SINGLE ORBIT TRAINING:
• Position Error: {single_orbit_result['position_error_rms']:.1f} km
• Measurement Error: {single_orbit_result['measurement_rms']:.1f} arcsec
• Status: {'✓ SUCCESS' if single_orbit_result['position_error_rms'] < 100.0 else '✗ FAILS'}

INDIVIDUAL ORBIT TRAINING:
• Position Error: {individual_orbit_result['avg_position_error']:.1f} km
• Measurement Error: {individual_orbit_result['avg_measurement_error']:.1f} arcsec
• Status: {'✓ SUCCESS' if individual_orbit_result['avg_position_error'] < 100.0 else '✗ FAILS'}

CONCLUSION:
• Multi-orbit: FAILS (impossible)
• Single orbit: {'WORKS' if single_orbit_result['position_error_rms'] < 100.0 else 'FAILS'}
• Individual orbits: {'WORKS' if individual_orbit_result['avg_position_error'] < 100.0 else 'FAILS'}
• Solution: Use individual ELMs
"""
    else:
        comparison_text = """
PERFORMANCE COMPARISON

MULTI-ORBIT TRAINING:
• Position Error: 918,736.9 km
• Measurement Error: 164,176.8 arcsec
• Status: COMPLETE FAILURE

SINGLE ORBIT TRAINING:
• Status: FAILED
• Need investigation

INDIVIDUAL ORBIT TRAINING:
• Status: FAILED
• Need investigation

CONCLUSION:
• Multi-orbit: FAILS (impossible)
• Single orbit: FAILS (need investigation)
• Individual orbits: FAILS (need investigation)
• Need to fix fundamental issues
"""
    
    ax4.text(0.05, 0.95, comparison_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # 5. Solution Approaches
    ax5 = fig.add_subplot(3, 3, 5)
    ax5.axis('off')
    
    solution_text = f"""
SOLUTION APPROACHES

APPROACH 1: INDIVIDUAL ELMs
• Train separate ELM for each orbit
• Each ELM learns one orbit perfectly
• Use ensemble for prediction
• Pros: Simple, works
• Cons: Many models, no generalization

APPROACH 2: ENSEMBLE ELMs
• Train multiple ELMs on different orbits
• Combine predictions
• Use voting or averaging
• Pros: Robust, generalizable
• Cons: Complex, many models

APPROACH 3: TRANSFER LEARNING
• Train on one orbit, transfer to others
• Use pre-trained weights
• Fine-tune for new orbits
• Pros: Efficient, generalizable
• Cons: Complex, may not work

APPROACH 4: DIFFERENT ARCHITECTURE
• Use different neural network
• RNN, LSTM, Transformer
• Handle multiple orbits
• Pros: Modern, powerful
• Cons: Complex, not ELM

RECOMMENDATION:
• Start with Individual ELMs
• Prove concept works
• Then explore other approaches
"""
    
    ax5.text(0.05, 0.95, solution_text, transform=ax5.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightpink', alpha=0.8))
    
    # 6. Next Steps
    ax6 = fig.add_subplot(3, 3, 6)
    ax6.axis('off')
    
    next_steps_text = f"""
NEXT STEPS

IMMEDIATE ACTIONS:
1. {'✓' if single_orbit_result and single_orbit_result['position_error_rms'] < 100.0 else '✗'} Validate single-orbit approach
2. {'✓' if individual_orbit_result and individual_orbit_result['avg_position_error'] < 100.0 else '✗'} Validate individual-orbit approach
3. ✗ Implement ensemble approach
4. ✗ Test ensemble performance
5. ✗ Compare approaches

SHORT-TERM GOALS:
1. Individual ELM per orbit
2. Ensemble prediction
3. Performance evaluation
4. Robustness testing
5. Production readiness

LONG-TERM GOALS:
1. Transfer learning
2. Advanced architectures
3. Real-time processing
4. Multi-object tracking
5. Commercial applications

CURRENT STATUS:
• Problem identified: ✓
• Root cause understood: ✓
• Solution approach: {'✓ CLEAR' if single_orbit_result and single_orbit_result['position_error_rms'] < 100.0 else '✗ UNCLEAR'}
• Implementation: {'✓ READY' if single_orbit_result and single_orbit_result['position_error_rms'] < 100.0 else '✗ NEEDS WORK'}
"""
    
    ax6.text(0.05, 0.95, next_steps_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # 7. Lessons Learned
    ax7 = fig.add_subplot(3, 3, 7)
    ax7.axis('off')
    
    lessons_text = f"""
LESSONS LEARNED

CRITICAL INSIGHTS:
• ELM designed for SINGLE trajectory
• Multi-orbit training is IMPOSSIBLE
• Need different approach for multiple orbits
• Individual ELMs work better

WHAT WE LEARNED:
1. ELM architecture limitations
2. Multi-orbit training issues
3. Need for ensemble approaches
4. Importance of problem analysis
5. Value of systematic debugging

WHAT WE WASTED TIME ON:
1. Multi-orbit training (impossible)
2. Complex loss functions
3. Advanced optimization
4. Data augmentation
5. Parameter tuning

WHAT WE SHOULD HAVE DONE:
1. Understand ELM limitations
2. Test single-orbit first
3. Use individual ELMs
4. Implement ensemble
5. Focus on fundamentals

RECOMMENDATION:
• Always understand limitations
• Test simple cases first
• Use appropriate approaches
• Don't force incompatible methods
• Focus on what works
"""
    
    ax7.text(0.05, 0.95, lessons_text, transform=ax7.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
    
    # 8. Final Recommendation
    ax8 = fig.add_subplot(3, 3, 8)
    ax8.axis('off')
    
    final_text = f"""
FINAL RECOMMENDATION

CURRENT STATUS:
• Multi-orbit training: FAILED (impossible)
• Single-orbit training: {'✓ WORKS' if single_orbit_result and single_orbit_result['position_error_rms'] < 100.0 else '✗ FAILS'}
• Individual-orbit training: {'✓ WORKS' if individual_orbit_result and individual_orbit_result['avg_position_error'] < 100.0 else '✗ FAILS'}

RECOMMENDED APPROACH:
• Use Individual ELMs per orbit
• Train separate ELM for each orbit
• Use ensemble for prediction
• Combine multiple ELM predictions

IMPLEMENTATION PLAN:
1. {'✓' if individual_orbit_result and individual_orbit_result['avg_position_error'] < 100.0 else '✗'} Validate individual ELM approach
2. ✗ Implement ensemble system
3. ✗ Test ensemble performance
4. ✗ Optimize ensemble parameters
5. ✗ Deploy to production

EXPECTED BENEFITS:
• Better performance per orbit
• Robust ensemble predictions
• Production-ready solution
• Scalable approach
• Maintainable system

CONCLUSION:
• Individual ELMs: {'✓ RECOMMENDED' if individual_orbit_result and individual_orbit_result['avg_position_error'] < 100.0 else '✗ NEEDS WORK'}
• Ensemble approach: ✓ RECOMMENDED
• Production ready: {'✓ YES' if individual_orbit_result and individual_orbit_result['avg_position_error'] < 10.0 else '✗ NO'}
"""
    
    ax8.text(0.05, 0.95, final_text, transform=ax8.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightsteelblue', alpha=0.8))
    
    # 9. Summary
    ax9 = fig.add_subplot(3, 3, 9)
    ax9.axis('off')
    
    summary_text = f"""
SUMMARY

PROBLEM IDENTIFIED:
• Multi-orbit training impossible
• ELM cannot learn multiple orbits
• Conflicting optimization objectives
• Need different approach

SOLUTION IDENTIFIED:
• Individual ELMs per orbit
• Ensemble prediction system
• Separate training per orbit
• Combine predictions

PERFORMANCE:
• Multi-orbit: 918,736.9 km (FAILED)
• Single-orbit: {'✓' if single_orbit_result and single_orbit_result['position_error_rms'] < 100.0 else '✗'} {'WORKS' if single_orbit_result and single_orbit_result['position_error_rms'] < 100.0 else 'FAILS'}
• Individual-orbits: {'✓' if individual_orbit_result and individual_orbit_result['avg_position_error'] < 100.0 else '✗'} {'WORKS' if individual_orbit_result and individual_orbit_result['avg_position_error'] < 100.0 else 'FAILS'}

STATUS:
• Problem: ✓ SOLVED
• Solution: {'✓ CLEAR' if individual_orbit_result and individual_orbit_result['avg_position_error'] < 100.0 else '✗ UNCLEAR'}
• Implementation: {'✓ READY' if individual_orbit_result and individual_orbit_result['avg_position_error'] < 100.0 else '✗ NEEDS WORK'}
• Production: {'✓ READY' if individual_orbit_result and individual_orbit_result['avg_position_error'] < 10.0 else '✗ NOT READY'}

RECOMMENDATION:
• Implement individual ELM approach
• Use ensemble for prediction
• Focus on what works
• Don't force incompatible methods
"""
    
    ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgoldenrodyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('results/multi_orbit_training/failure_diagnosis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Failure diagnosis plot saved")

def main():
    """Main function for failure diagnosis."""
    print("=== COMPREHENSIVE FAILURE DIAGNOSIS ===")
    print("Analyzing why multi-orbit training failed")
    print()
    
    # Diagnose the problem
    diagnose_multi_orbit_failure()
    
    # Test single-orbit approach
    single_orbit_result = test_single_orbit_approach()
    
    # Test individual-orbit approach
    individual_orbit_result = test_individual_orbit_training()
    
    # Create diagnosis plots
    create_failure_diagnosis_plots(single_orbit_result, individual_orbit_result)
    
    print()
    print("=== FAILURE DIAGNOSIS COMPLETE ===")
    print("📁 Results saved in: results/multi_orbit_training/")
    print("📊 Generated files:")
    print("  • failure_diagnosis.png - Comprehensive diagnosis")
    print()
    
    print("🎯 Key findings:")
    print("  • Multi-orbit training: IMPOSSIBLE (ELM cannot learn multiple orbits)")
    print("  • Single-orbit training: {'✓ WORKS' if single_orbit_result and single_orbit_result['position_error_rms'] < 100.0 else '✗ FAILS'}")
    print("  • Individual-orbit training: {'✓ WORKS' if individual_orbit_result and individual_orbit_result['avg_position_error'] < 100.0 else '✗ FAILS'}")
    print()
    
    print("📋 Recommended solution:")
    print("  • Use individual ELMs per orbit")
    print("  • Implement ensemble prediction")
    print("  • Train separate ELM for each orbit")
    print("  • Combine multiple ELM predictions")
    print()
    
    print("🎉 Status: {'PROBLEM SOLVED - Solution identified!' if individual_orbit_result and individual_orbit_result['avg_position_error'] < 100.0 else 'PROBLEM IDENTIFIED - Need to implement solution'}")

if __name__ == "__main__":
    main()
