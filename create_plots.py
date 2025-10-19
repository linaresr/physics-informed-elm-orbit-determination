#!/usr/bin/env python3
"""
Plotting script for ELM training results.
Reads saved training data and creates comprehensive visualizations.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import os

def load_training_data():
    """Load training results from JSON file."""
    if not os.path.exists('data/training_results.json'):
        print("Error: training_results.json not found!")
        print("Please run collect_training_data.py first.")
        return None
    
    with open('data/training_results.json', 'r') as f:
        return json.load(f)

def create_training_analysis_plots(data):
    """Create detailed training analysis plots."""
    print("Creating detailed training analysis plots...")
    
    detailed = data['detailed_analysis']
    if not detailed.get('success', False):
        print("Skipping detailed analysis plots - training failed")
        return
    
    # Convert back to numpy arrays
    r = np.array(detailed['r'])
    v = np.array(detailed['v'])
    t_eval = np.array(detailed['t_eval'])
    t_obs = np.array(detailed['t_obs'])
    ra_obs = np.array(detailed['ra_obs'])
    dec_obs = np.array(detailed['dec_obs'])
    station_eci = np.array(detailed['station_eci'])
    t_colloc = np.array(detailed['t_colloc'])
    physics_residuals = np.array(detailed['physics_residuals'])
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Physics-Informed ELM Training Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Observation Geometry
    ax1 = axes[0, 0]
    ax1.plot(ra_obs * 180/np.pi, dec_obs * 180/np.pi, 'ro-', markersize=8, linewidth=2, label='Observations')
    ax1.set_xlabel('Right Ascension (degrees)')
    ax1.set_ylabel('Declination (degrees)')
    ax1.set_title('Observation Geometry')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xlim(0, 5)
    ax1.set_ylim(0, 3)
    
    # Plot 2: Time Series of Observations
    ax2 = axes[0, 1]
    ax2.plot(t_obs/3600, ra_obs * 180/np.pi, 'ro-', markersize=6, label='RA')
    ax2.plot(t_obs/3600, dec_obs * 180/np.pi, 'bo-', markersize=6, label='DEC')
    ax2.set_xlabel('Time (hours)')
    ax2.set_ylabel('Angle (degrees)')
    ax2.set_title('Observation Time Series')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Position Magnitude
    ax3 = axes[0, 2]
    r_mag = np.linalg.norm(r, axis=0)
    ax3.plot(t_eval/3600, r_mag/1000, 'g-', linewidth=2, label='ELM Trajectory')
    ax3.axhline(y=42164, color='r', linestyle='--', alpha=0.7, label='GEO Altitude')
    ax3.set_xlabel('Time (hours)')
    ax3.set_ylabel('Position Magnitude (km)')
    ax3.set_title('Satellite Position Magnitude')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plot 4: 2D Trajectory Projection (XY plane)
    ax4 = axes[1, 0]
    ax4.plot(r[0]/1000, r[1]/1000, 'b-', linewidth=2, label='ELM Trajectory')
    ax4.scatter(station_eci[0]/1000, station_eci[1]/1000, 
               c='red', s=100, label='Station Positions', alpha=0.7)
    ax4.set_xlabel('X (km)')
    ax4.set_ylabel('Y (km)')
    ax4.set_title('Trajectory Projection (XY plane)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Velocity Magnitude
    ax5 = axes[1, 1]
    v_mag = np.linalg.norm(v, axis=0)
    ax5.plot(t_eval/3600, v_mag, 'purple', linewidth=2, label='ELM Velocity')
    ax5.axhline(y=3074, color='r', linestyle='--', alpha=0.7, label='GEO Velocity')
    ax5.set_xlabel('Time (hours)')
    ax5.set_ylabel('Velocity Magnitude (m/s)')
    ax5.set_title('Satellite Velocity Magnitude')
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    
    # Plot 6: Residual Analysis
    ax6 = axes[1, 2]
    ax6.plot(t_colloc/3600, physics_residuals, 'g-', linewidth=2, label='Physics Residuals')
    ax6.set_xlabel('Time (hours)')
    ax6.set_ylabel('Physics Residual Magnitude')
    ax6.set_title('Physics Residuals Over Time')
    ax6.grid(True, alpha=0.3)
    ax6.legend()
    
    plt.tight_layout()
    plt.savefig('data/training_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: data/training_analysis.png")

def create_learning_curve_plots(data):
    """Create learning curve analysis plots."""
    print("Creating learning curve plots...")
    
    results = data['learning_curve']
    
    # Extract data
    L_vals = [r['L'] for r in results]
    success = [r['success'] for r in results]
    nfev = [r['nfev'] if r['success'] else 0 for r in results]
    cost = [r['cost'] if r['success'] else np.nan for r in results]
    physics_rms = [r['physics_rms'] if r['success'] else np.nan for r in results]
    measurement_rms = [r['measurement_rms'] if r['success'] else np.nan for r in results]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('ELM Learning Curve Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Function Evaluations vs Network Size
    ax1 = axes[0, 0]
    ax1.plot(L_vals, nfev, 'bo-', markersize=8, linewidth=2)
    ax1.set_xlabel('Hidden Neurons (L)')
    ax1.set_ylabel('Function Evaluations')
    ax1.set_title('Convergence Speed vs Network Size')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Final Cost vs Network Size
    ax2 = axes[0, 1]
    ax2.semilogy(L_vals, cost, 'ro-', markersize=8, linewidth=2)
    ax2.set_xlabel('Hidden Neurons (L)')
    ax2.set_ylabel('Final Cost (log scale)')
    ax2.set_title('Final Cost vs Network Size')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Residual RMS vs Network Size
    ax3 = axes[1, 0]
    ax3.semilogy(L_vals, physics_rms, 'go-', markersize=8, linewidth=2, label='Physics RMS')
    ax3.semilogy(L_vals, measurement_rms, 'mo-', markersize=8, linewidth=2, label='Measurement RMS')
    ax3.set_xlabel('Hidden Neurons (L)')
    ax3.set_ylabel('Residual RMS (log scale)')
    ax3.set_title('Residual RMS vs Network Size')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plot 4: Position Range vs Network Size
    ax4 = axes[1, 1]
    pos_min = [r['position_range'][0] if r['success'] else np.nan for r in results]
    pos_max = [r['position_range'][1] if r['success'] else np.nan for r in results]
    ax4.plot(L_vals, pos_min, 'co-', markersize=8, linewidth=2, label='Min Position')
    ax4.plot(L_vals, pos_max, 'yo-', markersize=8, linewidth=2, label='Max Position')
    ax4.axhline(y=42164, color='r', linestyle='--', alpha=0.7, label='GEO Altitude')
    ax4.set_xlabel('Hidden Neurons (L)')
    ax4.set_ylabel('Position Magnitude (km)')
    ax4.set_title('Position Range vs Network Size')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('data/learning_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: data/learning_curves.png")

def create_residual_comparison_plots(data):
    """Create measurement residual comparison plots."""
    print("Creating residual comparison plots...")
    
    results = data['weight_comparison']
    
    # Filter successful results
    successful_results = [r for r in results if r['success']]
    
    if not successful_results:
        print("Skipping residual comparison plots - no successful results")
        return
    
    labels = [r['label'] for r in successful_results]
    measurement_rms = [r['measurement_rms'] for r in successful_results]
    physics_rms = [r['physics_rms'] for r in successful_results]
    pos_min = [r['position_range'][0] for r in successful_results]
    pos_max = [r['position_range'][1] for r in successful_results]
    
    colors = ['red', 'orange', 'green', 'blue'][:len(labels)]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Measurement Residual Improvement Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Measurement Residual Comparison
    ax1 = axes[0, 0]
    bars = ax1.bar(labels, measurement_rms, color=colors, alpha=0.7)
    ax1.set_ylabel('Measurement Residual RMS (arcsec)')
    ax1.set_title('Measurement Residual Comparison')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars, measurement_rms):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                 f'{val:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Position Range Comparison
    ax2 = axes[0, 1]
    x_pos = np.arange(len(labels))
    width = 0.35
    
    ax2.bar(x_pos - width/2, pos_min, width, label='Min Position', alpha=0.7, color='lightblue')
    ax2.bar(x_pos + width/2, pos_max, width, label='Max Position', alpha=0.7, color='darkblue')
    ax2.axhline(y=42164, color='r', linestyle='--', alpha=0.7, label='GEO Altitude')
    ax2.set_xlabel('Weight Configuration')
    ax2.set_ylabel('Position Magnitude (km)')
    ax2.set_title('Position Range Comparison')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Physics vs Measurement Residuals
    ax3 = axes[1, 0]
    ax3.scatter(physics_rms, measurement_rms, c=colors, s=100, alpha=0.7)
    for i, label in enumerate(labels):
        ax3.annotate(label, (physics_rms[i], measurement_rms[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    ax3.set_xlabel('Physics Residual RMS')
    ax3.set_ylabel('Measurement Residual RMS (arcsec)')
    ax3.set_title('Physics vs Measurement Residuals')
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Trajectory Comparison (position magnitude over time)
    ax4 = axes[1, 1]
    for i, result in enumerate(successful_results):
        if result['r'] is not None:
            r = np.array(result['r'])
            t_eval = np.array(result['t_eval'])
            r_mag = np.linalg.norm(r, axis=0)
            ax4.plot(t_eval/3600, r_mag/1000, 
                    color=colors[i], linewidth=2, label=result['label'], alpha=0.8)
    
    ax4.axhline(y=42164, color='r', linestyle='--', alpha=0.7, label='GEO Altitude')
    ax4.set_xlabel('Time (hours)')
    ax4.set_ylabel('Position Magnitude (km)')
    ax4.set_title('Trajectory Comparison')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('data/residual_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: data/residual_comparison.png")

def create_summary_plot(data):
    """Create a summary plot showing key metrics."""
    print("Creating summary plot...")
    
    # Extract key metrics
    learning_curve = data['learning_curve']
    weight_comparison = data['weight_comparison']
    
    # Find best results
    best_learning = min([r for r in learning_curve if r['success']], 
                       key=lambda x: x['measurement_rms'])
    best_weight = min([r for r in weight_comparison if r['success']], 
                     key=lambda x: x['measurement_rms'])
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    fig.suptitle('ELM Training Results Summary', fontsize=16, fontweight='bold')
    
    # Create a summary table
    metrics = [
        ['Metric', 'Learning Curve Best', 'Weight Comparison Best'],
        ['Network Size (L)', f"{best_learning['L']}", f"{best_weight.get('L', 'N/A')}"],
        ['Function Evaluations', f"{best_learning['nfev']}", f"{best_weight['nfev']}"],
        ['Physics RMS', f"{best_learning['physics_rms']:.6f}", f"{best_weight['physics_rms']:.6f}"],
        ['Measurement RMS (arcsec)', f"{best_learning['measurement_rms']:.1f}", f"{best_weight['measurement_rms']:.1f}"],
        ['Position Range (km)', f"{best_learning['position_range'][0]:.1f} - {best_learning['position_range'][1]:.1f}", 
         f"{best_weight['position_range'][0]:.1f} - {best_weight['position_range'][1]:.1f}"],
    ]
    
    # Create table
    table = ax.table(cellText=metrics[1:], colLabels=metrics[0], 
                    cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2)
    
    # Style the table
    for i in range(len(metrics[0])):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('data/summary_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: data/summary_results.png")

def main():
    """Main plotting function."""
    print("=== ELM TRAINING RESULTS VISUALIZATION ===")
    print("Loading training data...")
    
    data = load_training_data()
    if data is None:
        return
    
    print(f"Loaded data from {data['metadata']['timestamp']}")
    print(f"Total configurations: {data['metadata']['total_configurations']}")
    print(f"Successful configurations: {data['metadata']['successful_configurations']}")
    print()
    
    # Create all plots
    create_training_analysis_plots(data)
    create_learning_curve_plots(data)
    create_residual_comparison_plots(data)
    create_summary_plot(data)
    
    print()
    print("=== PLOTTING COMPLETE ===")
    print("Created plots:")
    print("• data/training_analysis.png - Detailed training analysis")
    print("• data/learning_curves.png - Learning curve analysis")
    print("• data/residual_comparison.png - Measurement residual comparison")
    print("• data/summary_results.png - Summary results table")
    print()
    print("All visualizations ready for analysis!")

if __name__ == "__main__":
    main()
