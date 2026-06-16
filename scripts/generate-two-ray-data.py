import numpy as np


def generate_data():
    ht = 10.0
    hr = 1.5
    f = 2.4e9
    c = 299792458.0
    lam = c / f
    k = 2 * np.pi / lam

    # Typical medium-dry ground parameters
    eps_r = 15.0 * (f / 1e9) ** (-0.1)
    sigma = 0.035 * (f / 1e9) ** (1.63)
    eps_i = 17.98 * sigma / (f / 1e9)
    eps_c = eps_r - 1j * eps_i

    d = np.logspace(0, 5, 2000)

    dp = np.sqrt(d**2 + (ht - hr) ** 2)
    dpp = np.sqrt(d**2 + (ht + hr) ** 2)

    sin_alpha = (ht + hr) / dpp
    cos_alpha = d / dpp

    # R_parallel (TM)
    term1_par = eps_c * sin_alpha
    term2_par = np.sqrt(eps_c - cos_alpha**2)
    R_par = (term1_par - term2_par) / (term1_par + term2_par)

    # R_perpendicular (TE)
    term1_perp = sin_alpha
    term2_perp = np.sqrt(eps_c - cos_alpha**2)
    R_perp = (term1_perp - term2_perp) / (term1_perp + term2_perp)

    def to_db(x):
        return 10 * np.log10(np.abs(x) ** 2)

    factor = lam / (4 * np.pi)  # Base for FSPL/Gain calculation

    E_los = factor * np.exp(-1j * k * dp) / dp

    E_ref_par = factor * R_par * np.exp(-1j * k * dpp) / dpp
    E_tot_par = E_los + E_ref_par

    E_ref_perp = factor * R_perp * np.exp(-1j * k * dpp) / dpp
    E_tot_perp = E_los + E_ref_perp

    P_los = to_db(E_los)
    P_ref_par = to_db(E_ref_par)
    P_tot_par = to_db(E_tot_par)
    P_ref_perp = to_db(E_ref_perp)
    P_tot_perp = to_db(E_tot_perp)

    data = np.stack([d, P_los, P_ref_perp, P_tot_perp, P_ref_par, P_tot_par], axis=1)
    np.savetxt(
        "data/two-ray-power.txt",
        data,
        delimiter=",",
        header="d,p_los,p_ref_perp,p_tot_perp,p_ref_par,p_tot_par",
        comments="",
    )
    print("Generated data/two-ray-power.txt successfully.")


if __name__ == "__main__":
    generate_data()
