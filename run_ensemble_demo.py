import os
import numpy as np
from scipy.integrate import solve_ivp

from piod.dynamics import eom
from piod.observe import ecef_to_eci, trig_to_radec, radec_to_trig
from ensemble_selection import run_ensemble_selection, plot_ensemble_fit, ensure_dir


def generate_truth(hours=4.0, seed=123):
    rng = np.random.default_rng(seed)
    r_geo = 42164000.0
    v_geo = 3074.0
    # small near-GEO variation
    r0 = np.array([r_geo + rng.uniform(-30000, 30000), 0.0, 0.0])
    v0 = np.array([0.0, v_geo + rng.uniform(-30, 30), 0.0])

    t0, t1 = 0.0, hours * 3600.0
    sol = solve_ivp(eom, [t0, t1], np.hstack([r0, v0]),
                    t_eval=np.linspace(t0, t1, int(hours*120)), rtol=1e-8, atol=1e-8)
    return t0, t1, sol.t, sol.y[:3]


def make_observations(t_true, r_true, n_obs=24, noise_rad=2e-5):
    t0, t1 = t_true[0], t_true[-1]
    t_obs = np.linspace(t0, t1, n_obs)
    station_ecef = np.array([6378136.3, 0.0, 0.0])
    jd_obs = 2451545.0 + t_obs / 86400.0
    station_eci = np.array([ecef_to_eci(station_ecef, jd) for jd in jd_obs]).T

    r_interp = np.vstack([
        np.interp(t_obs, t_true, r_true[0]),
        np.interp(t_obs, t_true, r_true[1]),
        np.interp(t_obs, t_true, r_true[2])
    ])
    topo = r_interp - station_eci

    # true angles
    ra_true, dec_true = trig_to_radec(
        np.sin(np.arctan2(topo[1], topo[0])),
        np.cos(np.arctan2(topo[1], topo[0])),
        topo[2] / np.linalg.norm(topo, axis=0)
    )

    # add noise
    ra_noisy = ra_true + np.random.normal(0, noise_rad, size=ra_true.shape)
    dec_noisy = dec_true + np.random.normal(0, noise_rad, size=dec_true.shape)

    obs = radec_to_trig(ra_noisy, dec_noisy)
    return t_obs, obs, station_eci


def main():
    out_dir = 'results/ensemble_demo'
    ensure_dir(out_dir)

    # Truth and observations
    t0, t1, t_true, r_true = generate_truth(hours=4.0)
    t_obs, obs, station_eci = make_observations(t_true, r_true, n_obs=36, noise_rad=2e-5)

    # Ensemble selection without prior orbit knowledge
    result = run_ensemble_selection(
        t0=t0, t1=t1, L=32, N_colloc=120,
        t_obs=t_obs, obs=obs, station_eci=station_eci,
        num_candidates=24, shortlist_k=4,
        quick_max_nfev=300, lam_f_refine=1.0, lam_th_refine=1e4,
        refine_max_nfev=4000, base_seed=100
    )

    beta = result['best_beta']
    model = result['best_model']

    # Plot
    plot_ensemble_fit(out_dir, t_true, r_true, t_obs, obs, station_eci, beta, model)

    # Save metrics
    with open(os.path.join(out_dir, 'metrics.txt'), 'w') as f:
        f.write(f"Best measurement RMS (arcsec): {result['best_measurement_rms']:.2f}\n")
        f.write(f"Best cost: {result['best_cost']}\n")
        f.write(f"Best seed: {result['best_seed']}\n")
        f.write("Gating top seeds (first 8):\n")
        for g in result['gating_summary']:
            f.write(f"  seed={g['seed']} rms={g['measurement_rms']:.2f} arcsec\n")

    print(f"Saved plots and metrics to {out_dir}")

if __name__ == '__main__':
    main()
