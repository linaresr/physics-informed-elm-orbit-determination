"""
Observation module for angle-only measurements.
Implements GMST calculation, ECEF→ECI conversion, and RA/DEC trig functions.
"""

import numpy as np


def gmst_rad(jd):
    """
    Compute Greenwich Mean Sidereal Time in radians.
    
    Parameters:
    -----------
    jd : float or array_like
        Julian date
        
    Returns:
    --------
    theta : float or ndarray
        GMST in radians
    """
    T = (jd - 2451545.0) / 36525.0
    theta = (280.46061837 + 360.98564736629 * (jd - 2451545.0)
             + 0.000387933 * T * T - (T**3) / 38710000.0) * np.pi / 180.0
    return np.mod(theta, 2 * np.pi)


def ecef_to_eci(r_ecef, jd):
    """
    Convert position from ECEF to ECI frame using simple Earth rotation.
    
    Parameters:
    -----------
    r_ecef : array_like, shape (3,) or (3, N)
        Position vector(s) in ECEF frame (m)
    jd : float or array_like
        Julian date
        
    Returns:
    --------
    r_eci : ndarray, shape (3,) or (3, N)
        Position vector(s) in ECI frame (m)
    """
    th = gmst_rad(jd)
    c, s = np.cos(th), np.sin(th)
    
    # Rotation matrix from ECEF to ECI
    R = np.array([[c, s, 0],
                  [-s, c, 0],
                  [0, 0, 1]])
    
    if r_ecef.ndim == 1:
        return R @ r_ecef
    else:
        return R @ r_ecef


def vec_to_radec(r):
    """
    Convert position vector to right ascension and declination.
    
    Parameters:
    -----------
    r : array_like, shape (3,)
        Position vector in ECI frame (m)
        
    Returns:
    --------
    ra : float
        Right ascension in radians
    dec : float
        Declination in radians
    """
    x, y, z = r
    rho = np.linalg.norm(r)
    dec = np.arcsin(z / rho)
    ra = np.arctan2(y, x) % (2 * np.pi)
    return ra, dec


def trig_ra_dec(r):
    """
    Convert position vector to trigonometric components of RA/DEC.
    This avoids wrap-around issues in the loss function.
    
    Parameters:
    -----------
    r : array_like, shape (3,)
        Position vector in ECI frame (m)
        
    Returns:
    --------
    trig : ndarray, shape (3,)
        [sin(RA), cos(RA), sin(DEC)]
    """
    ra, dec = vec_to_radec(r)
    return np.array([np.sin(ra), np.cos(ra), np.sin(dec)])


def radec_to_trig(ra, dec):
    """
    Convert RA/DEC angles to trigonometric components.
    
    Parameters:
    -----------
    ra : float or array_like
        Right ascension in radians
    dec : float or array_like
        Declination in radians
        
    Returns:
    --------
    trig : ndarray, shape (3,) or (3, N)
        [sin(RA), cos(RA), sin(DEC)]
    """
    return np.array([np.sin(ra), np.cos(ra), np.sin(dec)])


def trig_to_radec(sin_ra, cos_ra, sin_dec):
    """
    Convert trigonometric components back to RA/DEC angles.
    
    Parameters:
    -----------
    sin_ra : float or array_like
        sin(Right ascension)
    cos_ra : float or array_like
        cos(Right ascension)
    sin_dec : float or array_like
        sin(Declination)
        
    Returns:
    --------
    ra : float or ndarray
        Right ascension in radians
    dec : float or ndarray
        Declination in radians
    """
    ra = np.arctan2(sin_ra, cos_ra) % (2 * np.pi)
    dec = np.arcsin(sin_dec)
    return ra, dec


def test_observation():
    """
    Test function to verify observation module.
    """
    # Test GMST calculation
    jd_test = 2451545.0  # J2000.0
    gmst = gmst_rad(jd_test)
    print(f"GMST at J2000.0: {gmst:.6f} rad ({gmst*180/np.pi:.6f} deg)")
    
    # Test ECEF to ECI conversion
    r_ecef_test = np.array([6378136.3, 0.0, 0.0])  # On equator at Greenwich
    r_eci = ecef_to_eci(r_ecef_test, jd_test)
    print(f"ECEF position: {r_ecef_test}")
    print(f"ECI position: {r_eci}")
    
    # Test RA/DEC conversion
    r_test = np.array([42164000.0, 0.0, 0.0])  # GEO-like position
    ra, dec = vec_to_radec(r_test)
    trig = trig_ra_dec(r_test)
    print(f"Position: {r_test}")
    print(f"RA: {ra:.6f} rad ({ra*180/np.pi:.6f} deg)")
    print(f"DEC: {dec:.6f} rad ({dec*180/np.pi:.6f} deg)")
    print(f"Trig components: {trig}")
    
    return gmst, r_eci, ra, dec, trig


if __name__ == "__main__":
    test_observation()
