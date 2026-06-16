import numpy as np


def calculate_keller_coefficients(freq_GHz, alpha_deg, phi_prime_deg):
    f = freq_GHz * 1e9
    c = 299792458.0
    k = 2 * np.pi * f / c

    alpha = alpha_deg * np.pi / 180.0
    n = (2 * np.pi - alpha) / np.pi

    phi_prime = phi_prime_deg * np.pi / 180.0

    # Avoid exact singularities by slightly offsetting or just dropping exact values
    # but let's use a very fine grid
    phi_deg = np.linspace(0.001, 360 - alpha_deg - 0.001, 1000)
    phi = phi_deg * np.pi / 180.0

    # Equation 4.24
    num = -np.exp(-1j * np.pi / 4) * np.sin(np.pi / n)
    den = n * np.sqrt(2 * np.pi * k)  # sin(beta_0) = 1

    coef_term1 = 1 / (np.cos(np.pi / n) - np.cos((phi - phi_prime) / n))
    coef_term2 = 1 / (np.cos(np.pi / n) - np.cos((phi + phi_prime) / n))

    Ds = (num / den) * (coef_term1 - coef_term2)
    Dh = (num / den) * (coef_term1 + coef_term2)

    Ds_dB = 20 * np.log10(np.abs(Ds))
    Dh_dB = 20 * np.log10(np.abs(Dh))

    # For Fig 4.10, replace inf/nan with large numbers or just ignore

    return phi_deg, Ds_dB, Dh_dB


def calculate_scattered_field(freq_GHz, alpha_deg, phi_prime_deg, s):
    f = freq_GHz * 1e9
    c = 299792458.0
    k = 2 * np.pi * f / c

    alpha = alpha_deg * np.pi / 180.0
    n = (2 * np.pi - alpha) / np.pi

    phi_prime = phi_prime_deg * np.pi / 180.0

    phi_deg = np.linspace(0.001, 360 - alpha_deg - 0.001, 2000)
    phi = phi_deg * np.pi / 180.0

    C_val = 1.0  # 0 dB incident

    Ei = C_val * np.exp(1j * k * s * np.cos(phi - phi_prime))
    Er = -C_val * np.exp(1j * k * s * np.cos(phi + phi_prime))

    # Equation 4.24
    num = -np.exp(-1j * np.pi / 4) * np.sin(np.pi / n)
    den = n * np.sqrt(2 * np.pi * k)

    coef_term1 = 1 / (np.cos(np.pi / n) - np.cos((phi - phi_prime) / n))
    coef_term2 = 1 / (np.cos(np.pi / n) - np.cos((phi + phi_prime) / n))

    Ds = (num / den) * (coef_term1 - coef_term2)

    Ed = C_val * Ds * np.exp(-1j * k * s) / np.sqrt(s)

    Etot = np.zeros_like(Ei, dtype=np.complex128)

    RSB_rad = np.pi - phi_prime
    ISB_rad = np.pi + phi_prime

    reg1 = phi < RSB_rad
    reg2 = (phi >= RSB_rad) & (phi < ISB_rad)
    reg3 = phi >= ISB_rad

    Etot[reg1] = Ei[reg1] + Er[reg1] + Ed[reg1]
    Etot[reg2] = Ei[reg2] + Ed[reg2]
    Etot[reg3] = Ed[reg3]

    # Ei_dB = 20 * np.log10(np.abs(Ei))
    # Er_dB = 20 * np.log10(np.abs(Er))
    Ed_dB = 20 * np.log10(np.abs(Ed))
    Etot_dB = 20 * np.log10(np.abs(Etot))

    return phi_deg, Ed_dB, Etot_dB


def write_keller_data(filename="data/keller-coefficients-fig-4-10.txt"):
    phi_deg, Ds_dB, Dh_dB = calculate_keller_coefficients(10.0, 40.0, 55.0)

    with open(filename, "w") as f:
        f.write("phi,Ds_dB,Dh_dB\n")
        for i in range(len(phi_deg)):
            f.write(f"{phi_deg[i]:.4f},{Ds_dB[i]:.4f},{Dh_dB[i]:.4f}\n")


def write_field_data(filename="data/fig411_data.txt"):
    phi_deg, Ed_dB, Etot_dB = calculate_scattered_field(3.0, 40.0, 55.0, 1.0)

    with open(filename, "w") as f:
        f.write("phi,Ed_dB,Etot_dB\n")
        for i in range(len(phi_deg)):
            f.write(f"{phi_deg[i]:.4f},{Ed_dB[i]:.4f},{Etot_dB[i]:.4f}\n")


if __name__ == "__main__":
    write_keller_data()
    write_field_data()
