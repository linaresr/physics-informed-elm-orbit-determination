from __future__ import annotations

import os
import numpy as np
from typing import Dict, List, Tuple

from scipy.optimize import least_squares

# Local imports
from piod.elm import GeoELM
from piod.observe import trig_ra_dec
from piod.loss import residual as full_residual


def _measurement_only_residual(beta: np.ndarray,
                               model: GeoELM,
                               t_obs: np.ndarray,
                               obs: np.ndarray,
                               station_eci: np.ndarray) -> np.ndarray:
    """
    Compute measurement-only residuals in trig space for a given ELM model.

    Shapes:
    - beta: (3L,)
    - t_obs: (N_obs,)
    - obs: (3, N_obs)
    - station_eci: (3, N_obs)
    """
    r_pred, _, _ = model.r_v_a(t_obs, beta)  # (3, N_obs)
    r_topo = r_pred - station_eci            # (3, N_obs)
    theta_pred = np.apply_along_axis(trig_ra_dec, 0, r_topo)  # (3, N_obs)
    return (obs - theta_pred).ravel()


def quick_fit_measurement_only(t0: float,
                               t1: float,
                               L: int,
                               t_obs: np.ndarray,
                               obs: np.ndarray,
                               station_eci: np.ndarray,
                               seed: int,
                               max_nfev: int = 300) -> Tuple[np.ndarray, GeoELM, least_squares]:
    """
    Fast measurement-only fit used for ensemble gating.
    Returns (beta, model, result).
    """
    model = GeoELM(L=L, t_phys=np.array([t0, t1]), seed=seed)
    # small random init (same rationale as prior fix)
    rng = np.random.default_rng(seed)
    beta0 = rng.standard_normal(3 * L) * 0.01

    def fun(b):
        return _measurement_only_residual(b, model, t_obs, obs, station_eci)

    res = least_squares(fun, beta0, method="trf", max_nfev=max_nfev, ftol=1e-8, xtol=1e-8, gtol=1e-8)
    return res.x, model, res


def refine_fit_physics_and_measurements(t0: float,
                                        t1: float,
                                        beta_init: np.ndarray,
                                        model: GeoELM,
                                        t_colloc: np.ndarray,
                                        t_obs: np.ndarray,
                                        obs: np.ndarray,
                                        station_eci: np.ndarray,
                                        lam_f: float = 1.0,
                                        lam_th: float = 1e4,
                                        max_nfev: int = 8000) -> Tuple[np.ndarray, GeoELM, least_squares]:
    """
    Refinement using full physics+measurement residuals.
    """
    def fun(b):
        return full_residual(b, model, t_colloc, lam_f, obs, t_obs, station_eci, lam_th)

    res = least_squares(fun, beta_init, method="trf", max_nfev=max_nfev, ftol=1e-10, xtol=1e-10, gtol=1e-10)
    return res.x, model, res


def evaluate_measurement_rms(beta: np.ndarray,
                              model: GeoELM,
                              t_obs: np.ndarray,
                              obs: np.ndarray,
                              station_eci: np.ndarray) -> float:
    resid = _measurement_only_residual(beta, model, t_obs, obs, station_eci)
    # convert trig-space RMS to arcsec proxy by treating components in radians
    rms_rad = np.sqrt(np.mean(resid**2))
    return float(rms_rad * 180.0 / np.pi * 3600.0)


def run_ensemble_selection(t0: float,
                           t1: float,
                           L: int,
                           N_colloc: int,
                           t_obs: np.ndarray,
                           obs: np.ndarray,
                           station_eci: np.ndarray,
                           num_candidates: int = 32,
                           shortlist_k: int = 4,
                           quick_max_nfev: int = 300,
                           lam_f_refine: float = 1.0,
                           lam_th_refine: float = 1e4,
                           refine_max_nfev: int = 8000,
                           base_seed: int = 42) -> Dict:
    """
    Try many random ELM bases quickly (measurement-only), keep the best few by measurement RMS,
    then refine each with physics+measurement and return the best.
    """
    # Quick gating
    gating_results: List[Tuple[float, np.ndarray, GeoELM, least_squares, int]] = []
    for i in range(num_candidates):
        seed = base_seed + i
        beta_q, model_q, res_q = quick_fit_measurement_only(t0, t1, L, t_obs, obs, station_eci, seed, max_nfev=quick_max_nfev)
        meas_rms_q = evaluate_measurement_rms(beta_q, model_q, t_obs, obs, station_eci)
        gating_results.append((meas_rms_q, beta_q, model_q, res_q, seed))

    gating_results.sort(key=lambda x: x[0])
    shortlist = gating_results[:max(1, shortlist_k)]

    # Refinement
    t_colloc = np.linspace(t0, t1, N_colloc)
    refined: List[Tuple[float, float, np.ndarray, GeoELM, least_squares, int]] = []
    for meas_rms_q, beta_q, model_q, _, seed in shortlist:
        beta_r, model_r, res_r = refine_fit_physics_and_measurements(t0, t1, beta_q, model_q, t_colloc, t_obs, obs, station_eci,
                                                                     lam_f=lam_f_refine, lam_th=lam_th_refine, max_nfev=refine_max_nfev)
        meas_rms_r = evaluate_measurement_rms(beta_r, model_r, t_obs, obs, station_eci)
        refined.append((meas_rms_r, res_r.cost, beta_r, model_r, res_r, seed))

    refined.sort(key=lambda x: (x[0], x[1]))  # prioritize measurement RMS then cost
    best_meas_rms, best_cost, best_beta, best_model, best_res, best_seed = refined[0]

    return {
        "best_beta": best_beta,
        "best_model": best_model,
        "best_result": best_res,
        "best_measurement_rms": best_meas_rms,
        "best_cost": best_cost,
        "best_seed": best_seed,
        "gating_summary": [
            {"seed": seed, "measurement_rms": float(rms)} for rms, _, _, __, seed in gating_results[:8]
        ]
    }


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_ensemble_fit(output_dir: str,
                      t_true: np.ndarray,
                      r_true: np.ndarray,
                      t_obs: np.ndarray,
                      obs: np.ndarray,
                      station_eci: np.ndarray,
                      beta: np.ndarray,
                      model: GeoELM) -> None:
    import matplotlib.pyplot as plt

    ensure_dir(output_dir)

    # Build eval grid
    t_eval = np.linspace(t_true[0], t_true[-1], 400)
    r_est, v_est, a_est = model.r_v_a(t_eval, beta)

    # Interpolate truth
    r_true_interp = np.zeros_like(r_est)
    for i in range(3):
        r_true_interp[i] = np.interp(t_eval, t_true, r_true[i])

    # Errors
    pos_err = np.linalg.norm(r_est - r_true_interp, axis=0) / 1000.0

    # Compute predicted angles at obs times
    r_pred_obs, _, _ = model.r_v_a(t_obs, beta)
    theta_pred = np.apply_along_axis(trig_ra_dec, 0, r_pred_obs - station_eci)
    meas_resid = (obs - theta_pred)
    meas_resid_arcsec = meas_resid * 180.0 / np.pi * 3600.0

    # Plot 3 panels
    fig = plt.figure(figsize=(14, 10))

    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.plot(r_true[0]/1000.0, r_true[1]/1000.0, r_true[2]/1000.0, 'k-', lw=1.5, label='True')
    ax1.plot(r_est[0]/1000.0, r_est[1]/1000.0, r_est[2]/1000.0, 'r--', lw=1.0, label='ELM')
    ax1.set_title('Orbit (True vs ELM)')
    ax1.legend()

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot((t_eval - t_eval[0])/3600.0, pos_err, 'b-')
    ax2.set_title('Position error vs time (km)')
    ax2.set_xlabel('Hours')
    ax2.set_ylabel('km')

    ax3 = fig.add_subplot(2, 1, 2)
    ax3.plot((t_obs - t_obs[0])/3600.0, meas_resid_arcsec[0], 'r.-', label='sin(RA) residual (arcsec)')
    ax3.plot((t_obs - t_obs[0])/3600.0, meas_resid_arcsec[1], 'g.-', label='cos(RA) residual (arcsec)')
    ax3.plot((t_obs - t_obs[0])/3600.0, meas_resid_arcsec[2], 'b.-', label='sin(DEC) residual (arcsec)')
    ax3.set_title('Measurement residuals (arcsec)')
    ax3.set_xlabel('Hours')
    ax3.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'ensemble_fit.png'), dpi=180, bbox_inches='tight')
    plt.close(fig)
