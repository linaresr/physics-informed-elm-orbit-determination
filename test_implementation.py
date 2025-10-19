#!/usr/bin/env python3
"""
Test script to validate the physics-informed ELM implementation.
"""

import numpy as np
import sys
import os

# Add piod module to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'piod'))

from piod.dynamics import accel_2body_J2, eom, test_j2_acceleration
from piod.elm import GeoELM, test_elm
from piod.observe import gmst_rad, ecef_to_eci, vec_to_radec, trig_ra_dec, test_observation
from piod.loss import residual, physics_residual_rms, test_residual
from piod.solve import fit_physics_only, test_solver, evaluate_solution
from piod.utils import create_time_grid, propagate_state, test_utils


def test_all_modules():
    """
    Run all module tests to validate implementation.
    """
    print("="*60)
    print("RUNNING ALL MODULE TESTS")
    print("="*60)
    
    # Test dynamics
    print("\n1. Testing dynamics module...")
    try:
        test_j2_acceleration()
        print("✓ Dynamics module test passed")
    except Exception as e:
        print(f"✗ Dynamics module test failed: {e}")
        return False
    
    # Test ELM
    print("\n2. Testing ELM module...")
    try:
        elm, r, v, a = test_elm()
        print("✓ ELM module test passed")
    except Exception as e:
        print(f"✗ ELM module test failed: {e}")
        return False
    
    # Test observations
    print("\n3. Testing observation module...")
    try:
        gmst, r_eci, ra, dec, trig = test_observation()
        print("✓ Observation module test passed")
    except Exception as e:
        print(f"✗ Observation module test failed: {e}")
        return False
    
    # Test loss function
    print("\n4. Testing loss module...")
    try:
        res_physics, res_full = test_residual()
        print("✓ Loss module test passed")
    except Exception as e:
        print(f"✗ Loss module test failed: {e}")
        return False
    
    # Test solver
    print("\n5. Testing solver module...")
    try:
        beta, model, result = test_solver()
        print("✓ Solver module test passed")
    except Exception as e:
        print(f"✗ Solver module test failed: {e}")
        return False
    
    # Test utilities
    print("\n6. Testing utils module...")
    try:
        t_linear, t_cheb, t, r, v = test_utils()
        print("✓ Utils module test passed")
    except Exception as e:
        print(f"✗ Utils module test failed: {e}")
        return False
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("="*60)
    return True


def test_integration():
    """
    Test integration between modules.
    """
    print("\n" + "="*60)
    print("INTEGRATION TEST")
    print("="*60)
    
    # Create a simple test case
    t0, t1 = 0.0, 1800.0  # 30 minutes
    L = 32
    N_colloc = 40
    
    print(f"Testing integration with {L} hidden neurons, {N_colloc} collocation points")
    
    try:
        # Fit physics-only ELM
        beta, model, result = fit_physics_only(t0, t1, L=L, N_colloc=N_colloc)
        
        if not result.success:
            print(f"✗ Optimization failed: {result.message}")
            return False
        
        # Evaluate solution
        t_eval = create_time_grid(t0, t1, 100)
        r, v, a, physics_rms, _ = evaluate_solution(beta, model, t_eval)
        
        # Check that we get reasonable results
        r_mag = np.linalg.norm(r, axis=0)
        v_mag = np.linalg.norm(v, axis=0)
        
        print(f"Position range: {np.min(r_mag)/1000:.1f} - {np.max(r_mag)/1000:.1f} km")
        print(f"Velocity range: {np.min(v_mag):.1f} - {np.max(v_mag):.1f} m/s")
        print(f"Physics residual RMS: {physics_rms:.6f}")
        
        # Basic sanity checks (relaxed for testing)
        if np.min(r_mag) < 1e6 or np.max(r_mag) > 1e8:  # Reasonable orbit range
            print("✗ Position magnitudes seem unreasonable")
            return False
        
        if np.min(v_mag) < 100 or np.max(v_mag) > 50000:  # Reasonable velocity range
            print("✗ Velocity magnitudes seem unreasonable")
            return False
        
        if physics_rms > 10.0:  # Should be reasonably small
            print("✗ Physics residuals too large")
            return False
        
        print("✓ Integration test passed")
        return True
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """
    Main test function.
    """
    print("Physics-Informed ELM Implementation Tests")
    print("="*60)
    
    # Run module tests
    if not test_all_modules():
        return 1
    
    # Run integration test
    if not test_integration():
        return 1
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("The implementation is ready to use.")
    print("="*60)
    
    return 0


if __name__ == "__main__":
    exit(main())
