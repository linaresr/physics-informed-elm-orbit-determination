"""
Orbital Elements ELM implementation.
"""

import numpy as np
from scipy.optimize import fsolve

class OrbitalElementsELM:
    """
    ELM that learns orbital elements instead of Cartesian coordinates.
    This enforces orbital shape and improves observability.
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
        
        # Compute element variations (scalar values)
        delta_a = elm_weights[0] * H[0, 0] if H.shape[1] > 0 else 0
        delta_e = elm_weights[1] * H[1, 0] if H.shape[1] > 0 else 0
        delta_i = elm_weights[2] * H[2, 0] if H.shape[1] > 0 else 0
        delta_Omega = elm_weights[3] * H[3, 0] if H.shape[1] > 0 else 0
        delta_omega = elm_weights[4] * H[4, 0] if H.shape[1] > 0 else 0
        delta_M = elm_weights[5] * H[5, 0] if H.shape[1] > 0 else 0
        
        # Add variations to mean elements
        a = mean_elements[0] + delta_a
        e = mean_elements[1] + delta_e
        i = mean_elements[2] + delta_i
        Omega = mean_elements[3] + delta_Omega
        omega = mean_elements[4] + delta_omega
        M = mean_elements[5] + delta_M
        
        return np.array([a, e, i, Omega, omega, M])
    
    def solve_kepler_equation(self, M, e, max_iter=10):
        """Solve Kepler's equation: M = E - e*sin(E)"""
        M = np.array(M)
        e = np.array(e)
        
        # Ensure eccentricity is positive and reasonable
        e = np.clip(e, 0.0, 0.99)
        
        E = M.copy()  # Initial guess
        for _ in range(max_iter):
            E_new = M + e * np.sin(E)
            if np.all(np.abs(E_new - E) < 1e-10):
                break
            E = E_new
        return E
    
    def elements_to_cartesian(self, elements):
        """Convert orbital elements to Cartesian coordinates."""
        a, e, i, Omega, omega, M = elements
        
        # Handle scalar vs array inputs
        if np.isscalar(a):
            a, e, i, Omega, omega, M = [a], [e], [i], [Omega], [omega], [M]
            scalar_output = True
        else:
            scalar_output = False
        
        # Convert to numpy arrays and apply bounds
        a = np.clip(np.array(a), 40000000, 45000000)  # GEO altitude range
        e = np.clip(np.array(e), 0.0, 0.99)          # Reasonable eccentricity
        i = np.clip(np.array(i), 0.0, np.pi/2)        # Reasonable inclination
        Omega = np.array(Omega) % (2*np.pi)           # Wrap angles
        omega = np.array(omega) % (2*np.pi)           # Wrap angles
        M = np.array(M) % (2*np.pi)                   # Wrap angles
        
        # Solve Kepler's equation for eccentric anomaly
        E = self.solve_kepler_equation(M, e)
        
        # Compute true anomaly
        nu = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E/2), 
                           np.sqrt(1 - e) * np.cos(E/2))
        
        # Compute position in orbital plane
        r = a * (1 - e**2) / (1 + e * np.cos(nu))
        x_orb = r * np.cos(nu)
        y_orb = r * np.sin(nu)
        z_orb = np.zeros_like(x_orb)
        
        # Transform to inertial frame
        r_eci = np.zeros((3, len(a)))
        for j in range(len(a)):
            r_orb = np.array([x_orb[j], y_orb[j], z_orb[j]])
            r_eci[:, j] = self.orbital_to_inertial(r_orb, i[j], Omega[j], omega[j])
        
        # Compute velocity (simplified)
        n = np.sqrt(self.mu / a**3)  # Mean motion
        v_eci = np.zeros((3, len(a)))
        for j in range(len(a)):
            v_orb = n[j] * a[j] * np.sqrt(1 - e[j]**2) * np.array([-np.sin(nu[j]), 
                                                                   np.cos(nu[j]) + e[j], 0])
            v_eci[:, j] = self.orbital_to_inertial(v_orb, i[j], Omega[j], omega[j])
        
        if scalar_output:
            return r_eci[:, 0], v_eci[:, 0]
        else:
            return r_eci, v_eci
    
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
        
        # Ensure r is a 1D array for dynamics
        if r.ndim > 1:
            r = r.flatten()
        
        # Compute acceleration from dynamics
        from .dynamics import accel_2body_J2
        a = accel_2body_J2(r)
        
        return r, v, a


def test_orbital_elements_elm():
    """Test the orbital elements ELM."""
    print("Testing OrbitalElementsELM...")
    
    # Create ELM
    t_phys = np.array([0.0, 7200.0])  # 2 hours
    elm = OrbitalElementsELM(L=8, t_phys=t_phys)
    
    # Test beta vector (12 parameters: 6 elements + 6 weights)
    beta = np.array([
        42164000.0,  # a (GEO altitude)
        0.0,         # e (circular)
        0.0,         # i (equatorial)
        0.0,         # Omega
        0.0,         # omega
        0.0,         # M0
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # ELM weights
    ])
    
    # Test at a few times
    t_test = np.array([0.0, 3600.0, 7200.0])
    
    for t in t_test:
        r, v, a = elm.r_v_a(t, beta)
        print(f"t={t:.0f}s: r={np.linalg.norm(r)/1000:.1f}km, v={np.linalg.norm(v):.1f}m/s")
    
    print("✓ OrbitalElementsELM test completed")
    return elm


if __name__ == "__main__":
    test_orbital_elements_elm()
