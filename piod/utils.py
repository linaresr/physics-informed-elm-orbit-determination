"""
Utility functions for time scaling, I/O, and general helpers.
"""

import numpy as np
from datetime import datetime, timedelta


def seconds_to_jd(t_seconds, epoch_jd=2451545.0):
    """
    Convert seconds since epoch to Julian date.
    
    Parameters:
    -----------
    t_seconds : float or array_like
        Time in seconds since epoch
    epoch_jd : float
        Epoch Julian date (default: J2000.0)
        
    Returns:
    --------
    jd : float or ndarray
        Julian date
    """
    return epoch_jd + t_seconds / 86400.0


def jd_to_seconds(jd, epoch_jd=2451545.0):
    """
    Convert Julian date to seconds since epoch.
    
    Parameters:
    -----------
    jd : float or array_like
        Julian date
    epoch_jd : float
        Epoch Julian date (default: J2000.0)
        
    Returns:
    --------
    t_seconds : float or ndarray
        Time in seconds since epoch
    """
    return (jd - epoch_jd) * 86400.0


def create_time_grid(t0, t1, N, method='linear'):
    """
    Create time grid for collocation or evaluation.
    
    Parameters:
    -----------
    t0, t1 : float
        Start and end times (seconds)
    N : int
        Number of points
    method : str
        Grid method: 'linear', 'chebyshev', 'log'
        
    Returns:
    --------
    t : ndarray
        Time grid
    """
    if method == 'linear':
        return np.linspace(t0, t1, N)
    elif method == 'chebyshev':
        # Chebyshev points mapped to [t0, t1]
        cheb_points = np.cos(np.pi * np.arange(N) / (N - 1))
        return 0.5 * (t1 - t0) * (1 - cheb_points) + t0
    elif method == 'log':
        # Logarithmic spacing (useful for long arcs)
        return np.logspace(np.log10(t0 + 1), np.log10(t1 + 1), N) - 1
    else:
        raise ValueError(f"Unknown method: {method}")


def propagate_state(r0, v0, t0, t1, dt=60.0):
    """
    Simple state propagation using 4th-order Runge-Kutta.
    
    Parameters:
    -----------
    r0, v0 : array_like, shape (3,)
        Initial position and velocity (m, m/s)
    t0, t1 : float
        Start and end times (seconds)
    dt : float
        Integration step size (seconds)
        
    Returns:
    --------
    t : ndarray
        Time points
    r : ndarray, shape (3, N)
        Position vectors
    v : ndarray, shape (3, N)
        Velocity vectors
    """
    from .dynamics import eom
    
    # Create time grid
    t = np.arange(t0, t1 + dt, dt)
    N = len(t)
    
    # Initialize state
    y = np.zeros((6, N))
    y[:3, 0] = r0
    y[3:, 0] = v0
    
    # RK4 integration
    for i in range(N - 1):
        h = t[i + 1] - t[i]
        k1 = eom(t[i], y[:, i])
        k2 = eom(t[i] + h/2, y[:, i] + h/2 * k1)
        k3 = eom(t[i] + h/2, y[:, i] + h/2 * k2)
        k4 = eom(t[i] + h, y[:, i] + h * k3)
        
        y[:, i + 1] = y[:, i] + h/6 * (k1 + 2*k2 + 2*k3 + k4)
    
    return t, y[:3], y[3:]


def save_solution(filename, beta, model, t_eval, r, v, a, physics_rms, measurement_rms=None):
    """
    Save solution to file.
    
    Parameters:
    -----------
    filename : str
        Output filename
    beta : array_like
        ELM output weights
    model : GeoELM
        ELM model instance
    t_eval : array_like
        Evaluation time points
    r, v, a : ndarray
        Position, velocity, acceleration vectors
    physics_rms : float
        Physics residual RMS
    measurement_rms : float, optional
        Measurement residual RMS
    """
    data = {
        'beta': beta,
        'L': model.L,
        't0': model.t0,
        't1': model.t1,
        't_eval': t_eval,
        'r': r,
        'v': v,
        'a': a,
        'physics_rms': physics_rms,
        'measurement_rms': measurement_rms
    }
    
    np.savez(filename, **data)
    print(f"Solution saved to {filename}")


def load_solution(filename):
    """
    Load solution from file.
    
    Parameters:
    -----------
    filename : str
        Input filename
        
    Returns:
    --------
    data : dict
        Solution data
    """
    data = np.load(filename)
    return {key: data[key] for key in data.files}


def print_solution_summary(beta, model, physics_rms, measurement_rms=None):
    """
    Print summary of solution.
    
    Parameters:
    -----------
    beta : array_like
        ELM output weights
    model : GeoELM
        ELM model instance
    physics_rms : float
        Physics residual RMS
    measurement_rms : float, optional
        Measurement residual RMS
    """
    print("\n" + "="*50)
    print("SOLUTION SUMMARY")
    print("="*50)
    print(f"ELM hidden neurons: {model.L}")
    print(f"Time arc: {model.t0:.1f} to {model.t1:.1f} seconds ({model.t1-model.t0:.1f} s)")
    print(f"Output weights RMS: {np.sqrt(np.mean(beta**2)):.6f}")
    print(f"Physics residual RMS: {physics_rms:.6f}")
    if measurement_rms is not None:
        print(f"Measurement residual RMS: {measurement_rms:.3f} arcsec")
    print("="*50)


def test_utils():
    """
    Test function to verify utility functions.
    """
    # Test time conversions
    t_sec = 3600.0  # 1 hour
    jd = seconds_to_jd(t_sec)
    t_sec_back = jd_to_seconds(jd)
    print(f"Time conversion test: {t_sec} s -> {jd:.6f} JD -> {t_sec_back:.6f} s")
    
    # Test time grids
    t_linear = create_time_grid(0, 3600, 10, 'linear')
    t_cheb = create_time_grid(0, 3600, 10, 'chebyshev')
    print(f"Linear grid: {t_linear[:3]} ... {t_linear[-3:]}")
    print(f"Chebyshev grid: {t_cheb[:3]} ... {t_cheb[-3:]}")
    
    # Test state propagation
    r0 = np.array([42164000.0, 0.0, 0.0])  # GEO-like
    v0 = np.array([0.0, 3074.0, 0.0])     # Circular orbit velocity
    t, r, v = propagate_state(r0, v0, 0, 3600, dt=300)
    print(f"Propagation test: {len(t)} points, final position: {np.linalg.norm(r[:, -1])/1000:.1f} km")
    
    return t_linear, t_cheb, t, r, v


if __name__ == "__main__":
    test_utils()
