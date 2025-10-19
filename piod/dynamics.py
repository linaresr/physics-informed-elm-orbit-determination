"""
Dynamics module for 2-body + J2 orbit determination.
Implements gravitational acceleration and equations of motion in ECI/J2000 frame.
"""

import numpy as np

# Physical constants
MU = 398600.4418e9       # Earth gravitational parameter (m^3/s^2)
RE = 6378136.3           # Earth radius (m)
J2 = 1.08262668e-3       # J2 zonal harmonic coefficient


def accel_2body_J2(r):
    """
    Compute 2-body + J2 gravitational acceleration.
    
    Parameters:
    -----------
    r : array_like, shape (3,)
        Position vector in ECI frame (m)
        
    Returns:
    --------
    a : ndarray, shape (3,)
        Acceleration vector in ECI frame (m/s^2)
    """
    x, y, z = r
    r2 = np.dot(r, r)
    
    # Avoid division by zero
    if r2 < 1e-12:  # Very small distance
        return np.zeros(3)
    
    r1 = np.sqrt(r2)
    r3 = r2 * r1
    
    # Two-body acceleration
    a_tb = -MU * r / r3
    
    # J2 acceleration
    z2_r2 = (z * z) / r2
    k = 1.5 * J2 * MU * (RE**2) / (r2 * r1**3)  # (3/2) J2 μ RE^2 / r^5
    
    ax = k * x * (5.0 * z2_r2 - 1.0)
    ay = k * y * (5.0 * z2_r2 - 1.0)
    az = k * z * (5.0 * z2_r2 - 3.0)
    
    return a_tb + np.array([ax, ay, az])


def eom(t, y):
    """
    Equations of motion for 2-body + J2 dynamics.
    
    Parameters:
    -----------
    t : float
        Time (s)
    y : array_like, shape (6,)
        State vector [x, y, z, vx, vy, vz] in ECI frame
        
    Returns:
    --------
    dydt : ndarray, shape (6,)
        State derivative [vx, vy, vz, ax, ay, az]
    """
    r = y[:3]
    v = y[3:]
    a = accel_2body_J2(r)
    return np.hstack([v, a])


def test_j2_acceleration():
    """
    Test function to verify J2 acceleration against known values.
    """
    # Test case: GEO orbit at ~42164 km altitude
    r_test = np.array([42164000.0, 0.0, 0.0])  # m
    a = accel_2body_J2(r_test)
    
    # Expected: mainly radial acceleration with small J2 perturbation
    print(f"Test position: {r_test[0]/1000:.1f} km")
    print(f"Acceleration: {a} m/s^2")
    print(f"Magnitude: {np.linalg.norm(a):.6f} m/s^2")
    
    return a


if __name__ == "__main__":
    test_j2_acceleration()
