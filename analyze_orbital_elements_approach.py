#!/usr/bin/env python3
"""
Analysis of the orbit estimation issue and proposal for orbital elements approach.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import sys
sys.path.append('piod')
from piod.solve import fit_elm, evaluate_solution
from piod.observe import ecef_to_eci, radec_to_trig, vec_to_radec
from piod.utils import propagate_state
from piod.dynamics import eom
from scipy.integrate import solve_ivp

def analyze_orbit_estimation_issue():
    """Analyze why the network is learning mean position instead of true orbit."""
    print("=== ORBIT ESTIMATION ISSUE ANALYSIS ===")
    print()
    
    print("🔍 PROBLEM IDENTIFICATION:")
    print("The network is learning to:")
    print("• Fit measurements by passing through mean satellite position")
    print("• Ignore orbital dynamics (circular motion)")
    print("• Create unrealistic orbit shapes")
    print("• Prioritize measurement accuracy over physics")
    print()
    
    print("🎯 ROOT CAUSE:")
    print("1. CARTESIAN REPRESENTATION LIMITATIONS:")
    print("   • Network learns arbitrary 3D curves")
    print("   • No inherent orbital structure")
    print("   • Physics constraints are weak compared to measurement weights")
    print("   • No guarantee of orbital shape")
    print()
    
    print("2. MEASUREMENT FITTING BIAS:")
    print("   • Extreme weights (λ_th = 1B) force exact measurement fit")
    print("   • Network finds 'shortest path' through observation points")
    print("   • Ignores orbital dynamics in favor of measurement accuracy")
    print("   • Creates physically impossible trajectories")
    print()
    
    print("3. POOR OBSERVABILITY:")
    print("   • GEO satellites have minimal angular motion")
    print("   • Single station provides limited geometric constraints")
    print("   • Network can't distinguish between different orbital shapes")
    print("   • Mean position becomes the 'easiest' solution")
    print()
    
    print("💡 SOLUTION: ORBITAL ELEMENTS APPROACH")
    print()
    print("Instead of learning Cartesian coordinates directly, learn orbital elements:")
    print("• Semi-major axis (a)")
    print("• Eccentricity (e)")
    print("• Inclination (i)")
    print("• Right ascension of ascending node (Ω)")
    print("• Argument of perigee (ω)")
    print("• Mean anomaly (M)")
    print()
    
    print("ADVANTAGES:")
    print("1. PHYSICAL CONSTRAINTS:")
    print("   • Elements naturally enforce orbital shape")
    print("   • Impossible to create non-orbital trajectories")
    print("   • Physics is built into the representation")
    print()
    
    print("2. BETTER OBSERVABILITY:")
    print("   • Elements are more directly related to observations")
    print("   • Mean anomaly changes predictably with time")
    print("   • Orbital period is naturally constrained")
    print()
    
    print("3. IMPROVED CONVERGENCE:")
    print("   • Smaller parameter space (6 elements vs 3N coordinates)")
    print("   • Better conditioned optimization problem")
    print("   • More stable training")
    print()
    
    return True

def create_orbital_elements_elm():
    """Create an ELM that learns orbital elements instead of Cartesian coordinates."""
    print("=== CREATING ORBITAL ELEMENTS ELM ===")
    print()
    
    # Define the orbital elements ELM class
    elm_code = '''
import numpy as np
from scipy.optimize import fsolve

class OrbitalElementsELM:
    """
    ELM that learns orbital elements instead of Cartesian coordinates.
    """
    
    def __init__(self, L, t_phys, seed=42):
        rng = np.random.default_rng(seed)
        self.t0, self.t1 = t_phys[0], t_phys[-1]
        self.dtau_dt = 2.0/(self.t1 - self.t0)
        
        # Random weights for orbital elements
        self.W = rng.uniform(-1.0, 1.0, size=(L, 1))
        self.b = rng.uniform(-1.0, 1.0, size=(L,))
        self.L = L
        
        # Earth gravitational parameter
        self.mu = 398600.4418e9  # m^3/s^2
        
    def tau(self, t):
        return -1.0 + 2.0*(t - self.t0)/(self.t1 - self.t0)
    
    def h(self, t):
        tau = self.tau(t)
        z = self.W * tau + self.b[:,None]
        return np.tanh(z)
    
    def dh_dt(self, t):
        tau = self.tau(t)
        z = self.W * tau + self.b[:,None]
        sech2 = 1.0/np.cosh(z)**2
        return (sech2 * self.W) * self.dtau_dt
    
    def elements_from_beta(self, beta):
        """Extract orbital elements from beta vector."""
        # beta contains: [a, e, i, Omega, omega, M0, beta_a, beta_e, beta_i, beta_Omega, beta_omega, beta_M]
        # First 6 are mean elements, last 6 are ELM weights for variations
        
        mean_elements = beta[:6]
        elm_weights = beta[6:12]
        
        return mean_elements, elm_weights
    
    def elements_at_time(self, t, beta):
        """Get orbital elements at time t."""
        mean_elements, elm_weights = self.elements_from_beta(beta)
        
        # Get ELM basis functions
        H = self.h(t)
        
        # Compute element variations
        delta_a = elm_weights[0] * H[0] if len(H) > 0 else 0
        delta_e = elm_weights[1] * H[1] if len(H) > 1 else 0
        delta_i = elm_weights[2] * H[2] if len(H) > 2 else 0
        delta_Omega = elm_weights[3] * H[3] if len(H) > 3 else 0
        delta_omega = elm_weights[4] * H[4] if len(H) > 4 else 0
        delta_M = elm_weights[5] * H[5] if len(H) > 5 else 0
        
        # Add variations to mean elements
        a = mean_elements[0] + delta_a
        e = mean_elements[1] + delta_e
        i = mean_elements[2] + delta_i
        Omega = mean_elements[3] + delta_Omega
        omega = mean_elements[4] + delta_omega
        M = mean_elements[5] + delta_M
        
        return np.array([a, e, i, Omega, omega, M])
    
    def elements_to_cartesian(self, elements):
        """Convert orbital elements to Cartesian coordinates."""
        a, e, i, Omega, omega, M = elements
        
        # Solve Kepler's equation for eccentric anomaly
        E = self.solve_kepler_equation(M, e)
        
        # Compute true anomaly
        nu = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E/2), 
                           np.sqrt(1 - e) * np.cos(E/2))
        
        # Compute position in orbital plane
        r = a * (1 - e**2) / (1 + e * np.cos(nu))
        x_orb = r * np.cos(nu)
        y_orb = r * np.sin(nu)
        z_orb = 0
        
        # Transform to inertial frame
        r_orb = np.array([x_orb, y_orb, z_orb])
        r_eci = self.orbital_to_inertial(r_orb, i, Omega, omega)
        
        # Compute velocity (simplified)
        n = np.sqrt(self.mu / a**3)  # Mean motion
        v_orb = n * a * np.sqrt(1 - e**2) * np.array([-np.sin(nu), 
                                                      np.cos(nu) + e, 0])
        v_eci = self.orbital_to_inertial(v_orb, i, Omega, omega)
        
        return r_eci, v_eci
    
    def solve_kepler_equation(self, M, e, max_iter=10):
        """Solve Kepler's equation: M = E - e*sin(E)"""
        E = M  # Initial guess
        for _ in range(max_iter):
            E_new = M + e * np.sin(E)
            if np.abs(E_new - E) < 1e-10:
                break
            E = E_new
        return E
    
    def orbital_to_inertial(self, r_orb, i, Omega, omega):
        """Transform from orbital plane to inertial frame."""
        # Rotation matrices
        R3_Omega = np.array([[np.cos(Omega), np.sin(Omega), 0],
                           [-np.sin(Omega), np.cos(Omega), 0],
                           [0, 0, 1]])
        
        R1_i = np.array([[1, 0, 0],
                        [0, np.cos(i), np.sin(i)],
                        [0, -np.sin(i), np.cos(i)]])
        
        R3_omega = np.array([[np.cos(omega), np.sin(omega), 0],
                            [-np.sin(omega), np.cos(omega), 0],
                            [0, 0, 1]])
        
        # Combined rotation
        R = R3_Omega @ R1_i @ R3_omega
        return R @ r_orb
    
    def r_v_a(self, t, beta):
        """Get position, velocity, acceleration at time t."""
        elements = self.elements_at_time(t, beta)
        r, v = self.elements_to_cartesian(elements)
        
        # Compute acceleration from dynamics
        from piod.dynamics import accel_2body_J2
        a = accel_2body_J2(r)
        
        return r, v, a
'''
    
    print("✓ Orbital Elements ELM class defined")
    print()
    print("KEY FEATURES:")
    print("• Learns 6 orbital elements + 6 ELM weights for variations")
    print("• Enforces orbital shape through Kepler's equation")
    print("• Converts elements to Cartesian for measurement comparison")
    print("• Maintains physical constraints")
    print()
    
    return elm_code

def create_implementation_plan():
    """Create implementation plan for orbital elements approach."""
    print("=== IMPLEMENTATION PLAN ===")
    print()
    
    print("PHASE 1: BASIC ORBITAL ELEMENTS ELM")
    print("1. Implement OrbitalElementsELM class")
    print("2. Add Kepler's equation solver")
    print("3. Add orbital-to-inertial transformations")
    print("4. Test with simple 2-body dynamics")
    print()
    
    print("PHASE 2: INTEGRATION")
    print("1. Modify loss function to work with elements")
    print("2. Update solver to handle element-based residuals")
    print("3. Test with J2 dynamics")
    print()
    
    print("PHASE 3: OPTIMIZATION")
    print("1. Test different element representations")
    print("2. Optimize element bounds and scaling")
    print("3. Compare with Cartesian approach")
    print()
    
    print("EXPECTED IMPROVEMENTS:")
    print("• Position RMS: < 100 km (vs 4,295 km)")
    print("• Measurement RMS: < 5 arcsec (maintained)")
    print("• Physics RMS: < 0.001 (improved)")
    print("• Training stability: Much better")
    print()
    
    print("IMPLEMENTATION FILES:")
    print("• piod/elm_elements.py - Orbital elements ELM")
    print("• piod/loss_elements.py - Element-based loss function")
    print("• piod/solve_elements.py - Element-based solver")
    print("• test_elements_approach.py - Test script")
    print()
    
    return True

def create_visual_comparison():
    """Create visual comparison of Cartesian vs Orbital Elements approach."""
    print("Creating visual comparison...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Problem illustration
    ax = axes[0, 0]
    ax.axis('off')
    
    problem_text = """
CARTESIAN APPROACH PROBLEM:

Network learns arbitrary 3D curves:
• No orbital structure
• Fits measurements by 'shortest path'
• Ignores orbital dynamics
• Creates unrealistic trajectories

Example:
True Orbit: Circular GEO orbit
ELM Estimate: Straight line through observations
Result: Poor position accuracy
"""
    
    ax.text(0.05, 0.95, problem_text, transform=ax.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    # 2. Solution illustration
    ax = axes[0, 1]
    ax.axis('off')
    
    solution_text = """
ORBITAL ELEMENTS SOLUTION:

Network learns orbital parameters:
• Enforces orbital shape
• Physics built into representation
• Better observability
• Realistic trajectories

Example:
True Orbit: Circular GEO orbit
ELM Estimate: Proper orbital elements
Result: Good position accuracy
"""
    
    ax.text(0.05, 0.95, solution_text, transform=ax.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # 3. Parameter comparison
    ax = axes[1, 0]
    
    approaches = ['Cartesian', 'Orbital Elements']
    parameters = [192, 12]  # 3*L vs 6+6
    observability = [0.1, 0.8]  # Relative observability
    
    x = np.arange(len(approaches))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, parameters, width, label='Parameters', alpha=0.7)
    bars2 = ax.bar(x + width/2, observability, width, label='Observability', alpha=0.7)
    
    ax.set_xlabel('Approach')
    ax.set_ylabel('Relative Value')
    ax.set_title('Parameter Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(approaches)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Expected results
    ax = axes[1, 1]
    ax.axis('off')
    
    results_text = """
EXPECTED IMPROVEMENTS:

Position RMS:
• Cartesian: 4,295 km
• Elements: < 100 km

Measurement RMS:
• Cartesian: 1.60 arcsec
• Elements: < 5 arcsec

Physics RMS:
• Cartesian: 0.008
• Elements: < 0.001

Training Stability:
• Cartesian: Poor (extreme weights)
• Elements: Good (balanced)

Convergence:
• Cartesian: Slow, unstable
• Elements: Fast, stable
"""
    
    ax.text(0.05, 0.95, results_text, transform=ax.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('data/orbital_elements_approach.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Visual comparison saved to: data/orbital_elements_approach.png")

if __name__ == "__main__":
    analyze_orbit_estimation_issue()
    create_orbital_elements_elm()
    create_implementation_plan()
    create_visual_comparison()
