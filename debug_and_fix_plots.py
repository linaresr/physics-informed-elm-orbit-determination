#!/usr/bin/env python3
"""
Debug and fix the plotting issues in the results visualization.
"""

import sys
sys.path.append('piod')
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle
from piod.solve_elements_enhanced import fit_elm_elements_enhanced, evaluate_solution_elements_enhanced
from piod.observe import ecef_to_eci, radec_to_trig, trig_to_radec
from piod.dynamics import eom
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from piod.elm_elements import OrbitalElementsELM
from piod.loss_elements_enhanced import residual_elements_enhanced

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

def estimate_initial_elements_from_observations(obs, t_obs, station_eci):
    """Estimate initial orbital elements from observations."""
    # Use first observation to estimate orbit
    sin_ra, cos_ra, sin_dec = obs[:, 0]
    ra, dec = trig_to_radec(sin_ra, cos_ra, sin_dec)
    
    # Estimate distance (GEO altitude)
    estimated_distance = 42164000
    
    # Convert to position vector
    r_estimated = estimated_distance * np.array([
        np.cos(dec) * np.cos(ra),
        np.cos(dec) * np.sin(ra),
        np.sin(dec)
    ])
    
    # Estimate orbital elements
    r_mag = np.linalg.norm(r_estimated)
    a_est = r_mag
    e_est = 0.0  # Assume circular
    i_est = 0.0  # Assume equatorial
    
    # Estimate angles from position
    Omega_est = np.arctan2(r_estimated[1], r_estimated[0]) % (2*np.pi)
    omega_est = 0.0
    M_est = 0.0
    
    return np.array([a_est, e_est, i_est, Omega_est, omega_est, M_est])

def debug_and_fix_plots():
    """Debug and create corrected plots."""
    print("=== DEBUGGING AND FIXING PLOTS ===")
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
    
    # Run the advanced strategy
    print("3. Running advanced orbital elements strategy...")
    estimated_elements = estimate_initial_elements_from_observations(obs, t_obs, station_eci)
    beta0_custom = np.hstack([estimated_elements, np.random.randn(6) * 50])
    
    model = OrbitalElementsELM(L=8, t_phys=np.array([t0, t1]))
    t_colloc = np.linspace(t0, t1, 25)
    
    def fun(beta):
        return residual_elements_enhanced(beta, model, t_colloc, 1.0, obs, t_obs, station_eci, 1000.0, 1000.0)
    
    bounds_tight = (
        np.array([42000000, 0.0, 0.0, 0.0, 0.0, 0.0, -500, -500, -500, -500, -500, -500]),
        np.array([42300000, 0.05, 0.05, 2*np.pi, 2*np.pi, 2*np.pi, 500, 500, 500, 500, 500, 500])
    )
    
    result = least_squares(fun, beta0_custom, method="trf", max_nfev=5000, 
                          ftol=1e-12, xtol=1e-12, gtol=1e-12, bounds=bounds_tight)
    
    print(f"✓ Optimization completed: {result.success}, {result.nfev} evals")
    
    # Evaluate solution
    t_eval = np.linspace(t0, t1, 100)
    r, v, a, physics_rms, measurement_rms, position_magnitude_rms = evaluate_solution_elements_enhanced(
        result.x, model, t_eval, obs, t_obs, station_eci)
    
    print(f"✓ Solution evaluated")
    print(f"  Position Magnitude RMS: {position_magnitude_rms:.1f} km")
    print(f"  Measurement RMS: {measurement_rms:.2f} arcsec")
    print(f"  Physics RMS: {physics_rms:.6f}")
    
    # Debug: Check the actual values
    print("\n4. Debugging values...")
    print(f"True orbit shape: {r_true.shape}")
    print(f"Estimated orbit shape: {r.shape}")
    print(f"True orbit range: X={r_true[0].min()/1000:.1f} to {r_true[0].max()/1000:.1f} km")
    print(f"Estimated orbit range: X={r[0].min()/1000:.1f} to {r[0].max()/1000:.1f} km")
    
    # Interpolate true orbit to evaluation times
    r_true_interp = np.zeros_like(r)
    for i in range(3):
        r_true_interp[i] = np.interp(t_eval, t_true, r_true[i])
    
    # Calculate actual position error
    r_error = np.linalg.norm(r - r_true_interp, axis=0)
    print(f"Position error range: {r_error.min()/1000:.1f} to {r_error.max()/1000:.1f} km")
    print(f"Position error RMS: {np.sqrt(np.mean(r_error**2))/1000:.1f} km")
    
    # Check if the issue is with the position magnitude RMS vs actual position error
    r_mag_true = np.linalg.norm(r_true_interp, axis=0)
    r_mag_est = np.linalg.norm(r, axis=0)
    magnitude_error = np.abs(r_mag_est - r_mag_true)
    print(f"Magnitude error range: {magnitude_error.min()/1000:.1f} to {magnitude_error.max()/1000:.1f} km")
    print(f"Magnitude error RMS: {np.sqrt(np.mean(magnitude_error**2))/1000:.1f} km")
    
    # The issue might be that we're plotting magnitude error instead of actual position error
    print(f"\n5. The issue: position_magnitude_rms={position_magnitude_rms:.1f} km")
    print(f"   But actual position error RMS={np.sqrt(np.mean(r_error**2))/1000:.1f} km")
    print(f"   These should be similar but might be different!")
    
    # Create corrected plots
    print("\n6. Creating corrected plots...")
    
    # Create corrected orbit comparison plot
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
    
    # Plot orbits
    ax1.plot(r_true[0]/1000, r_true[1]/1000, r_true[2]/1000, 'g-', linewidth=3, label='True Orbit')
    ax1.plot(r[0]/1000, r[1]/1000, r[2]/1000, 'r--', linewidth=2, label='Estimated Orbit')
    
    ax1.set_xlabel('X (km)')
    ax1.set_ylabel('Y (km)')
    ax1.set_zlabel('Z (km)')
    ax1.set_title('3D Orbit Comparison')
    ax1.legend()
    
    # 2. XY Plane Comparison
    ax2 = fig.add_subplot(3, 3, 2)
    ax2.plot(r_true[0]/1000, r_true[1]/1000, 'g-', linewidth=3, label='True Orbit')
    ax2.plot(r[0]/1000, r[1]/1000, 'r--', linewidth=2, label='Estimated Orbit')
    
    # Add Earth circle
    earth_circle = Circle((0, 0), 6378.136, fill=False, color='blue', linestyle='--', alpha=0.5)
    ax2.add_patch(earth_circle)
    
    ax2.set_xlabel('X (km)')
    ax2.set_ylabel('Y (km)')
    ax2.set_title('XY Plane Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    # 3. Position Error vs Time (CORRECTED)
    ax3 = fig.add_subplot(3, 3, 3)
    ax3.plot(t_eval/3600, r_error/1000, 'b-', linewidth=2)
    ax3.set_xlabel('Time (hours)')
    ax3.set_ylabel('Position Error (km)')
    ax3.set_title(f'Position Error vs Time\nRMS: {np.sqrt(np.mean(r_error**2))/1000:.1f} km')
    ax3.grid(True, alpha=0.3)
    
    # 4. Position Magnitude Comparison
    ax4 = fig.add_subplot(3, 3, 4)
    ax4.plot(t_eval/3600, r_mag_true/1000, 'g-', linewidth=2, label='True')
    ax4.plot(t_eval/3600, r_mag_est/1000, 'r--', linewidth=2, label='Estimated')
    ax4.axhline(y=42164, color='k', linestyle=':', alpha=0.5, label='GEO Altitude')
    
    ax4.set_xlabel('Time (hours)')
    ax4.set_ylabel('Altitude (km)')
    ax4.set_title('Position Magnitude Comparison')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Error Components
    ax5 = fig.add_subplot(3, 3, 5)
    error_x = np.abs(r[0] - r_true_interp[0])/1000
    error_y = np.abs(r[1] - r_true_interp[1])/1000
    error_z = np.abs(r[2] - r_true_interp[2])/1000
    
    ax5.plot(t_eval/3600, error_x, 'r-', linewidth=2, label='X Error')
    ax5.plot(t_eval/3600, error_y, 'g-', linewidth=2, label='Y Error')
    ax5.plot(t_eval/3600, error_z, 'b-', linewidth=2, label='Z Error')
    
    ax5.set_xlabel('Time (hours)')
    ax5.set_ylabel('Position Error (km)')
    ax5.set_title('Position Error Components')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Error distribution
    ax6 = fig.add_subplot(3, 3, 6)
    ax6.hist(r_error/1000, bins=20, alpha=0.7, color='blue', edgecolor='black')
    ax6.set_xlabel('Total Error (km)')
    ax6.set_ylabel('Frequency')
    ax6.set_title('Error Distribution')
    ax6.grid(True, alpha=0.3)
    
    # 7. Debug information
    ax7 = fig.add_subplot(3, 3, 7)
    ax7.axis('off')
    
    debug_text = f"""
DEBUG INFORMATION

True Orbit:
• X range: {r_true[0].min()/1000:.1f} to {r_true[0].max()/1000:.1f} km
• Y range: {r_true[1].min()/1000:.1f} to {r_true[1].max()/1000:.1f} km
• Z range: {r_true[2].min()/1000:.1f} to {r_true[2].max()/1000:.1f} km

Estimated Orbit:
• X range: {r[0].min()/1000:.1f} to {r[0].max()/1000:.1f} km
• Y range: {r[1].min()/1000:.1f} to {r[1].max()/1000:.1f} km
• Z range: {r[2].min()/1000:.1f} to {r[2].max()/1000:.1f} km

Position Errors:
• Total RMS: {np.sqrt(np.mean(r_error**2))/1000:.1f} km
• Max error: {r_error.max()/1000:.1f} km
• Min error: {r_error.min()/1000:.1f} km

Magnitude Errors:
• Magnitude RMS: {np.sqrt(np.mean(magnitude_error**2))/1000:.1f} km
• Max magnitude error: {magnitude_error.max()/1000:.1f} km
• Min magnitude error: {magnitude_error.min()/1000:.1f} km

Key Insight:
Position error = ||r_est - r_true||
Magnitude error = |r_est_mag - r_true_mag|
These are different metrics!
"""
    
    ax7.text(0.05, 0.95, debug_text, transform=ax7.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # 8. Performance Summary (CORRECTED)
    ax8 = fig.add_subplot(3, 3, 8)
    ax8.axis('off')
    
    summary_text = f"""
CORRECTED PERFORMANCE SUMMARY

Position Accuracy:
• Position Error RMS: {np.sqrt(np.mean(r_error**2))/1000:.1f} km
• Position Magnitude RMS: {position_magnitude_rms:.1f} km
• Max Error: {r_error.max()/1000:.1f} km
• Mean Error: {r_error.mean()/1000:.1f} km

Measurement Accuracy:
• Measurement RMS: {measurement_rms:.2f} arcsec
• Target: <5 arcsec
• Status: {'✓ ACHIEVED' if measurement_rms < 5.0 else '✗ NOT ACHIEVED'}

Physics Compliance:
• Physics RMS: {physics_rms:.6f}
• Target: <0.001
• Status: {'✓ ACHIEVED' if physics_rms < 0.001 else '✗ NOT ACHIEVED'}

Training Efficiency:
• Function evals: {result.nfev}
• Success: {result.success}
• Final cost: {result.cost:.2e}

Overall Status:
• Position target (<10 km): {'✓ ACHIEVED' if np.sqrt(np.mean(r_error**2))/1000 < 10.0 else '✗ NOT ACHIEVED'}
• Measurement target (<5 arcsec): {'✓ ACHIEVED' if measurement_rms < 5.0 else '✗ NOT ACHIEVED'}
• Physics target (<0.001): {'✓ ACHIEVED' if physics_rms < 0.001 else '✗ NOT ACHIEVED'}
"""
    
    ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # 9. Comparison of metrics
    ax9 = fig.add_subplot(3, 3, 9)
    ax9.axis('off')
    
    comparison_text = f"""
METRIC COMPARISON

Two Different Error Metrics:

1. Position Error RMS:
   • Formula: sqrt(mean(||r_est - r_true||²))
   • Value: {np.sqrt(np.mean(r_error**2))/1000:.1f} km
   • This is the actual position error

2. Position Magnitude RMS:
   • Formula: sqrt(mean((|r_est| - |r_true|)²))
   • Value: {position_magnitude_rms:.1f} km
   • This is the altitude error

Key Insight:
• Position error measures how far apart the orbits are
• Magnitude error measures altitude difference
• For GEO orbits, these can be very different!
• The plots should show position error, not magnitude error

Recommendation:
• Use position error RMS for orbit accuracy
• Use magnitude error RMS for altitude accuracy
• Both are important but different metrics
"""
    
    ax9.text(0.05, 0.95, comparison_text, transform=ax9.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('results/advanced_strategies/corrected_orbit_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Corrected orbit comparison plot saved")
    
    # Create corrected measurement residuals plot
    print("7. Creating corrected measurement residuals plot...")
    
    # Calculate measurement residuals
    residuals = []
    predicted_obs = []
    
    for i, t in enumerate(t_obs):
        # Get position from ELM
        r_obs, _, _ = model.r_v_a(t, result.x)
        
        # Compute topocentric vector
        r_topo = r_obs - station_eci[:, i]
        
        # Convert to trigonometric components
        from piod.observe import trig_ra_dec
        theta_pred = trig_ra_dec(r_topo)
        predicted_obs.append(theta_pred)
        
        # Calculate residual
        residual = obs[:, i] - theta_pred
        residuals.append(residual)
    
    residuals = np.array(residuals)
    predicted_obs = np.array(predicted_obs)
    
    # Convert to RA/DEC for plotting
    ra_obs, dec_obs = trig_to_radec(obs[0], obs[1], obs[2])
    ra_pred, dec_pred = trig_to_radec(predicted_obs[:, 0], predicted_obs[:, 1], predicted_obs[:, 2])
    
    # Convert to arcseconds
    ra_residuals = (ra_obs - ra_pred) * 180/np.pi * 3600
    dec_residuals = (dec_obs - dec_pred) * 180/np.pi * 3600
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. RA residuals vs time
    ax1 = axes[0, 0]
    ax1.plot(t_obs/3600, ra_residuals, 'bo-', linewidth=2, markersize=6)
    ax1.set_xlabel('Time (hours)')
    ax1.set_ylabel('RA Residual (arcsec)')
    ax1.set_title(f'RA Residuals vs Time\nRMS: {np.sqrt(np.mean(ra_residuals**2)):.2f} arcsec')
    ax1.grid(True, alpha=0.3)
    
    # 2. DEC residuals vs time
    ax2 = axes[0, 1]
    ax2.plot(t_obs/3600, dec_residuals, 'ro-', linewidth=2, markersize=6)
    ax2.set_xlabel('Time (hours)')
    ax2.set_ylabel('DEC Residual (arcsec)')
    ax2.set_title(f'DEC Residuals vs Time\nRMS: {np.sqrt(np.mean(dec_residuals**2)):.2f} arcsec')
    ax2.grid(True, alpha=0.3)
    
    # 3. Combined residuals
    ax3 = axes[1, 0]
    total_residuals = np.sqrt(ra_residuals**2 + dec_residuals**2)
    ax3.plot(t_obs/3600, total_residuals, 'go-', linewidth=2, markersize=6)
    ax3.set_xlabel('Time (hours)')
    ax3.set_ylabel('Total Residual (arcsec)')
    ax3.set_title(f'Total Residuals vs Time\nRMS: {np.sqrt(np.mean(total_residuals**2)):.2f} arcsec')
    ax3.grid(True, alpha=0.3)
    
    # 4. Residual statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    residual_stats_text = f"""
MEASUREMENT RESIDUAL STATISTICS

RA Residuals:
• RMS: {np.sqrt(np.mean(ra_residuals**2)):.2f} arcsec
• Max: {np.max(np.abs(ra_residuals)):.2f} arcsec
• Min: {np.min(np.abs(ra_residuals)):.2f} arcsec

DEC Residuals:
• RMS: {np.sqrt(np.mean(dec_residuals**2)):.2f} arcsec
• Max: {np.max(np.abs(dec_residuals)):.2f} arcsec
• Min: {np.min(np.abs(dec_residuals)):.2f} arcsec

Total Residuals:
• RMS: {np.sqrt(np.mean(total_residuals**2)):.2f} arcsec
• Max: {np.max(total_residuals):.2f} arcsec
• Min: {np.min(total_residuals):.2f} arcsec

Target Achievement:
• Target: <5 arcsec
• Status: {'✓ ACHIEVED' if np.sqrt(np.mean(total_residuals**2)) < 5.0 else '✗ NOT ACHIEVED'}

Measurement Quality:
• Excellent: <1 arcsec
• Good: <5 arcsec
• Acceptable: <10 arcsec
• Poor: >10 arcsec

Current Status: {'EXCELLENT' if np.sqrt(np.mean(total_residuals**2)) < 1.0 else 'GOOD' if np.sqrt(np.mean(total_residuals**2)) < 5.0 else 'ACCEPTABLE' if np.sqrt(np.mean(total_residuals**2)) < 10.0 else 'POOR'}
"""
    
    ax4.text(0.05, 0.95, residual_stats_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('results/advanced_strategies/corrected_measurement_residuals.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Corrected measurement residuals plot saved")
    
    print()
    print("=== CORRECTED PLOTS CREATED ===")
    print("📁 Results saved in: results/advanced_strategies/")
    print("📊 Corrected plots:")
    print("  • corrected_orbit_comparison.png - Fixed orbit comparisons")
    print("  • corrected_measurement_residuals.png - Fixed measurement residuals")
    print()
    print("🔍 Key findings:")
    print(f"  • Position Error RMS: {np.sqrt(np.mean(r_error**2))/1000:.1f} km")
    print(f"  • Position Magnitude RMS: {position_magnitude_rms:.1f} km")
    print(f"  • Measurement RMS: {measurement_rms:.2f} arcsec")
    print("  • These are different metrics - position error is the correct one!")

if __name__ == "__main__":
    debug_and_fix_plots()
