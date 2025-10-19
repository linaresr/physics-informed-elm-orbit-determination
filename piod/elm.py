"""
Physics-Informed Extreme Learning Machine (ELM) for orbit determination.
Single hidden layer with fixed random weights and trainable output weights.
"""

import numpy as np


class GeoELM:
    """
    Extreme Learning Machine for representing satellite trajectory.
    
    The network represents position as: r(t) = β^T tanh(W*τ(t) + b)
    where τ(t) ∈ [-1,1] is normalized time, W and b are fixed random weights,
    and β are the trainable output weights.
    """
    
    def __init__(self, L, t_phys, seed=42):
        """
        Initialize ELM with fixed random hidden layer.
        
        Parameters:
        -----------
        L : int
            Number of hidden neurons
        t_phys : array_like, shape (2,)
            Physical time bounds [t0, t1] in seconds
        seed : int
            Random seed for reproducible weights
        """
        self.L = L
        self.t0, self.t1 = t_phys[0], t_phys[-1]
        self.dtau_dt = 2.0 / (self.t1 - self.t0)  # dτ/dt for time scaling
        
        # Initialize random weights and biases (fixed)
        rng = np.random.default_rng(seed)
        self.W = rng.uniform(-1.0, 1.0, size=(L, 1))  # input weights
        self.b = rng.uniform(-1.0, 1.0, size=(L,))     # biases
        
    def tau(self, t):
        """
        Convert physical time to normalized time τ ∈ [-1,1].
        
        Parameters:
        -----------
        t : array_like
            Physical time(s) in seconds
            
        Returns:
        --------
        tau : ndarray
            Normalized time(s) ∈ [-1,1]
        """
        return -1.0 + 2.0 * (t - self.t0) / (self.t1 - self.t0)
    
    def h(self, t):
        """
        Compute hidden layer activations h(τ) = tanh(W*τ + b).
        
        Parameters:
        -----------
        t : array_like
            Physical time(s) in seconds
            
        Returns:
        --------
        h : ndarray, shape (L, N)
            Hidden layer activations for N time points
        """
        tau = self.tau(t)
        z = self.W * tau + self.b[:, None]  # Broadcasting: (L,1) * (1,N) + (L,1)
        return np.tanh(z)
    
    def dh_dt(self, t):
        """
        Compute first derivative of hidden layer w.r.t. time.
        
        Parameters:
        -----------
        t : array_like
            Physical time(s) in seconds
            
        Returns:
        --------
        dh_dt : ndarray, shape (L, N)
            First derivative of hidden layer activations
        """
        tau = self.tau(t)
        z = self.W * tau + self.b[:, None]
        sech2 = 1.0 / np.cosh(z)**2
        return (sech2 * self.W) * self.dtau_dt
    
    def d2h_dt2(self, t):
        """
        Compute second derivative of hidden layer w.r.t. time.
        
        Parameters:
        -----------
        t : array_like
            Physical time(s) in seconds
            
        Returns:
        --------
        d2h_dt2 : ndarray, shape (L, N)
            Second derivative of hidden layer activations
        """
        tau = self.tau(t)
        z = self.W * tau + self.b[:, None]
        tanh_z = np.tanh(z)
        sech2 = 1.0 / np.cosh(z)**2
        
        # d/dt (sech^2(z)) = -2*tanh(z)*sech^2(z) * dz/dt
        dz_dt = self.W * self.dtau_dt
        return (-2.0 * tanh_z * sech2) * (dz_dt * dz_dt)
    
    def split_beta(self, beta):
        """
        Split output weight vector into x, y, z components.
        
        Parameters:
        -----------
        beta : array_like, shape (3*L,)
            Concatenated output weights [βx; βy; βz]
            
        Returns:
        --------
        bx, by, bz : ndarray, each shape (L,)
            Output weights for x, y, z components
        """
        L = self.L
        return beta[:L], beta[L:2*L], beta[2*L:3*L]
    
    def r_v_a(self, t, beta):
        """
        Compute position, velocity, and acceleration from ELM.
        
        Parameters:
        -----------
        t : array_like
            Physical time(s) in seconds
        beta : array_like, shape (3*L,)
            Output weights [βx; βy; βz]
            
        Returns:
        --------
        r : ndarray, shape (3, N)
            Position vectors for N time points
        v : ndarray, shape (3, N)
            Velocity vectors for N time points
        a : ndarray, shape (3, N)
            Acceleration vectors for N time points
        """
        bx, by, bz = self.split_beta(beta)
        
        H = self.h(t)        # (L, N)
        dH = self.dh_dt(t)   # (L, N)
        d2H = self.d2h_dt2(t)  # (L, N)
        
        # Position: r_k = β_k^T h(τ)
        rx = bx @ H
        ry = by @ H
        rz = bz @ H
        
        # Velocity: v_k = β_k^T dh/dt
        vx = bx @ dH
        vy = by @ dH
        vz = bz @ dH
        
        # Acceleration: a_k = β_k^T d²h/dt²
        ax = bx @ d2H
        ay = by @ d2H
        az = bz @ d2H
        
        r = np.vstack([rx, ry, rz])
        v = np.vstack([vx, vy, vz])
        a = np.vstack([ax, ay, az])
        
        return r, v, a


def test_elm():
    """
    Test function to verify ELM implementation.
    """
    # Test with simple parameters
    L = 32
    t_phys = np.array([0.0, 3600.0])  # 1 hour arc
    elm = GeoELM(L, t_phys, seed=42)
    
    # Test time points
    t_test = np.linspace(0, 3600, 10)
    
    # Random output weights
    beta = np.random.randn(3 * L)
    
    # Compute trajectory
    r, v, a = elm.r_v_a(t_test, beta)
    
    print(f"ELM test:")
    print(f"Hidden neurons: {L}")
    print(f"Time arc: {t_phys[0]} to {t_phys[1]} seconds")
    print(f"Position shape: {r.shape}")
    print(f"Velocity shape: {v.shape}")
    print(f"Acceleration shape: {a.shape}")
    print(f"Sample position: {r[:, 0]}")
    
    return elm, r, v, a


if __name__ == "__main__":
    test_elm()
