import numpy as np
from itertools import product
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator


def M_nogse(t_C, tau_c, G, D0=1e-5, N=2, T=21.5 * 1e-3):
    """Función que calcula la magnetización resultante de aplicar la secuencia de Pulsos Non Oscillating Gradient Spin Echo (NOGSE).

    Args:
        t_C (float): Tiempo de refocalización en secuencia CPMG. (ms) Nuestra variable que vamos a cambiar.
        T (float): Tiempo total de la secuencia. (ms)
        N (int): Número de pulsos/ciclos de la secuencia CPMG. (adimensional)
        tau_c (float): Tiempo de correlación. (ms) -> relacionado con l_c como tau_c = l_c^2 / (2 D0)
        D0 (float): Coeficiente de difusión. (cm^2 / ms)
        G (float): Gradiente de campo magnético. (G/cm)

    Returns:
        float: Magnetización resultante de aplicar la secuencia NOGSE.
    """
    g = 26752.218744  # s**-1 y gauss**-1 #gamma del nucleo del proton

    T = np.array(T)
    t_C = np.array(t_C)
    N = np.array(N)
    G = np.array(G)

    t_H = T - (N - 1) * t_C  # Tiempo de la secuencia Hahn

    bSE = (
        g * G * np.sqrt(D0 * tau_c)
    )  # tau_c es el tiempo de correlacion, D0 coefficiente de difusion del orden de

    return (
        np.exp(
            -(bSE**2)
            * tau_c**2
            * (4 * np.exp(-t_H / tau_c / 2) - np.exp(-t_H / tau_c) - 3 + t_H / tau_c)
        )
        * np.exp(
            -(bSE**2)
            * tau_c**2
            * (
                (N - 1) * t_C / tau_c
                + (-1) ** (N - 1) * np.exp(-(N - 1) * t_C / tau_c)
                + 1
                - 2 * N
                - 4
                * np.exp(-(N - 1) * t_C / tau_c) ** (1 / (N - 1) / 2)
                * (-np.exp(-(N - 1) * t_C / tau_c) ** (1 / (N - 1))) ** (N - 1)
                / (np.exp(-(N - 1) * t_C / tau_c) ** (1 / (N - 1)) + 1)
                + 4
                * np.exp(-(N - 1) * t_C / tau_c) ** (1 / (N - 1) / 2)
                / (np.exp(-(N - 1) * t_C / tau_c) ** (1 / (N - 1)) + 1)
                + 4
                * (-np.exp(-(N - 1) * t_C / tau_c) ** (1 / (N - 1))) ** (N - 1)
                * np.exp(-(N - 1) * t_C / tau_c) ** (1 / (N - 1))
                / (np.exp(-(N - 1) * t_C / tau_c) ** (1 / (N - 1)) + 1) ** 2
                + 4
                * np.exp(-(N - 1) * t_C / tau_c) ** (1 / (N - 1))
                * ((N - 1) * np.exp(-(N - 1) * t_C / tau_c) ** (1 / (N - 1)) + N - 2)
                / (np.exp(-(N - 1) * t_C / tau_c) ** (1 / (N - 1)) + 1) ** 2
            )
        )
        * np.exp(
            2
            * tau_c**2
            * (
                (
                    np.exp((-t_H + 2 * t_C) / tau_c / 2)
                    + np.exp((t_C - 2 * t_H) / tau_c / 2)
                    - np.exp((t_C - t_H) / tau_c) / 2
                    - np.exp(-t_H / tau_c) / 2
                    + np.exp(t_C / tau_c / 2)
                    + np.exp(-t_H / tau_c / 2)
                    - np.exp(t_C / tau_c) / 2
                    - 0.1e1 / 0.2e1
                )
                * (-1) ** (2 * N)
                + 2
                * (-1) ** (1 + N)
                * np.exp(-(2 * N * t_C - 3 * t_C + t_H) / tau_c / 2)
                + (
                    np.exp(((3 - 2 * N) * t_C - 2 * t_H) / tau_c / 2)
                    - np.exp((-N * t_C + 2 * t_C - t_H) / tau_c) / 2
                    + np.exp(-(2 * N * t_C - 4 * t_C + t_H) / tau_c / 2)
                    + np.exp(-(2 * N * t_C - 2 * t_C + t_H) / tau_c / 2)
                    - np.exp((-N * t_C + t_C - t_H) / tau_c) / 2
                    + np.exp(-t_C * (-3 + 2 * N) / tau_c / 2)
                    - np.exp(-t_C * (N - 2) / tau_c) / 2
                    - np.exp(-(N - 1) * t_C / tau_c) / 2
                )
                * (-1) ** N
                + 2 * (-1) ** (1 + 2 * N) * np.exp((t_C - t_H) / tau_c / 2)
            )
            * bSE**2
            / (np.exp(t_C / tau_c) + 1)
        )
    )


def S_nogse(
    lcs,
    lcm: float,
    sigma: float,
    t_C: float,
    G: float,
    D0=1e-5,
    N=2,
    T=21.5 * 1e-3,
    TE=54.3 * 1e-3,
    T2=31.5 * 1e-3,
):
    """Función que calcula la señal resultante de aplicar la secuencia de Pulsos Non Oscillating Gradient Spin Echo (NOGSE) con los efectos de la relajación transversal
    en un valor de tiempo de refocalización de CPMG t_C.

    Args:
        lc (np.array) : Longitud de correlación para la distribución de tamaños. (cm) relacionado con tau_c = l_c^2 / (2 D0)
        lcm (float): Mediana de la longitud de correlación. (cm)
        sigma (float): Desviación estándar de la distribución de tamaños.
        t_C (float): Tiempo de refocalización en secuencia CPMG. (s) Nuestra variable temporal.
        G (float): Gradiente de campo magnético. (G/cm)
        D0 (float): Coeficiente de difusión. (cm^2 / s)
        N (int): Número de pulsos/ciclos de la secuencia CPMG. (adimensional)
        T (float): Tiempo total de la secuencia. (s)
        TE (float): Tiempo total de evolución mayor a T. (s)
        T2 (float): Tiempo de relajación transversal. (s)

    Returns:
        S(t_C) (float): Señal resultante
    """

    D0 = np.array(D0)
    tau_cs = np.power(lcs, 2) / (2 * D0)

    return np.sum(
        M_nogse(t_C, tau_cs, G, D0, N, T) * Plog(lcs, lcm, sigma) * np.exp(-TE / T2)
    )


def Plog(lc, lcm, sigma):
    """Función de densidad de probabilidad de una distribución lognormal.

    Args:
        lc (np.array): longitudes de correlación para las cuales se calcula la densidad de probabilidad.
        lcm (float): Mediana de la longitud de correlación.
        sigma (float): Desviación estándar de la distribución.

    Returns:
        float: Densidad de probabilidad en lc.

    Asegurarse de poner lc y lcm en las mismas unidades. (micrómetros)
    """

    lc = np.array(lc)

    pre_factor = 1 / (lc * sigma * np.sqrt(2 * np.pi))
    numerator = (np.log(lc) - np.log(lcm)) ** 2
    denominator = 2 * sigma**2
    Pl = pre_factor * np.exp(-numerator / denominator)
    return Pl / np.sum(Pl)


if __name__ == "__main__":
    # Parametros de las posibles distribuciones de tamaños de restricción
    lcm_min = 0.5 * 1e-3  # cm
    lcm_step = 0.1 * 1e-3  # cm
    lcm_max = 6 * 1e-3  # cm

    sigma_min = 0.01
    sigma_step = 0.01
    sigma_max = 1.0

    # Longitudes de correlación, simulamos un rango de 0.01 a 50 micrómetros con N_compartimientos = 5000
    lc_min = 0.1 * 1e-3  # cm
    lc_max = 50 * 1e-3  # cm
    N_compartimientos = 5000

    lcs = np.linspace(lc_min, lc_max, N_compartimientos)  # Longitudes de correlación

    # Parametros fijos de la señal de NOGSE
    T = 21.5 * 1e-3  # s
    TE = 54.3 * 1e-3  # s
    T2 = 31.5 * 1e-3  # s
    N = 2  # adimensional
    D0 = 1e-5  # cm^2/s

    # Tiempos de CMPG
    t_Cs = np.linspace(0, T / N, 1100)  # s

    # Gradientes de campo magnético
    Gs = [42.5, 50]  # G/cm
    lcms = [0.5e-3, 6e-3]  # cm
    sigmas = [0.05, 0.1]

    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    for i, (sigma, lcm) in enumerate(product(sigmas, lcms)):
        axs[i % 2, 1].plot(
            lcs * 1e3,
            Plog(lcs, lcm, sigma),
            label=f"lcm = {lcm * 1e3} um, sigma = {sigma}",
        )
        axs[i % 2, 1].legend(fontsize=12)
        axs[i % 2, 1].tick_params(
            axis="both", which="both", direction="in", top=True, right=True
        )
        axs[i % 2, 1].grid(which="both", alpha=0.2)
        axs[i % 2, 1].set_xlabel("lc (um)", fontsize=12)
        axs[i % 2, 1].set_ylabel("Plog", fontsize=12)

    S_nogse_tc = np.zeros(len(t_Cs))
    S_nogse_tc2 = np.zeros(len(t_Cs))
    for _t, t_C in enumerate(t_Cs):
        S_nogse_tc[_t] = S_nogse(lcs, 0.5e-3, 0.05, t_C, Gs[0], D0, N, T, TE, T2)
        S_nogse_tc2[_t] = S_nogse(lcs, 0.5e-3, 0.1, t_C, Gs[1], D0, N, T, TE, T2)

    axs[0, 0].plot(
        t_Cs * 1e3,
        S_nogse_tc,
        label=f"G = {Gs[0]} G/cm",
    )
    axs[0, 0].plot(
        t_Cs * 1e3,
        S_nogse_tc2,
        label=f"G = {Gs[1]} G/cm",
    )
    axs[0, 0].legend(fontsize=12)
    axs[0, 0].tick_params(
        axis="both", which="both", direction="in", top=True, right=True
    )
    axs[0, 0].grid(which="both", alpha=0.2)
    axs[0, 0].set_xlabel("t_C (ms)", fontsize=12)
    axs[0, 0].set_ylabel("S(t_C)", fontsize=12)

    S_nogse_tc = np.zeros(len(t_Cs))
    S_nogse_tc2 = np.zeros(len(t_Cs))
    for _t, t_C in enumerate(t_Cs):
        S_nogse_tc[_t] = S_nogse(lcs, 6e-3, 0.05, t_C, Gs[0], D0, N, T, TE, T2)
        S_nogse_tc2[_t] = S_nogse(lcs, 6e-3, 0.1, t_C, Gs[1], D0, N, T, TE, T2)

    axs[1, 0].plot(
        t_Cs * 1e3,
        S_nogse_tc,
        label=f"G = {Gs[0]} G/cm",
    )
    axs[1, 0].plot(
        t_Cs * 1e3,
        S_nogse_tc2,
        label=f"G = {Gs[1]} G/cm",
    )
    axs[1, 0].legend(fontsize=12)
    axs[1, 0].tick_params(
        axis="both", which="both", direction="in", top=True, right=True
    )
    axs[1, 0].grid(which="both", alpha=0.2)
    axs[1, 0].set_xlabel("t_C (ms)", fontsize=12)
    axs[1, 0].set_ylabel("S(t_C)", fontsize=12)

    plt.show()

    # for

    #     for G in Gs:
    #         S_nogse_tc = np.zeros(len(t_Cs))
    #         for j, t_C in enumerate(t_Cs):
    #             S_nogse_tc[j] = S_nogse(lcs, lcm, sigma, t_C, G, D0, N, T, TE, T2)
    #         axs[i % 2, 0].plot(
    #             t_Cs * 1e3,
    #             S_nogse_tc,
    #             label=f"G = {G} G/cm",
    #         )
    #         axs[i % 2, 0].legend(fontsize=12)
    #         axs[i % 2, 0].tick_params(
    #             axis="both", which="both", direction="in", top=True, right=True
    #         )
    #         axs[i % 2, 0].grid(which="both", alpha=0.2)
    #         axs[i % 2, 0].set_xlabel("t_C (ms)", fontsize=12)
    #         axs[i % 2, 0].set_ylabel("S(t_C)", fontsize=12)
    # plt.show()

    # fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # configs =

    # for G in Gs:
    #     plt.plot(
    #         t_Cs * 1e3,
    #         M_nogse(t_Cs, (lc_max**2) / (2 * D0), G, D0, N, T),
    #         label=f"G = {G} G/cm",
    #     )

    # plt.legend(fontsize=12)
    # plt.tick_params(axis="both", which="both", direction="in", top=True, right=True)
    # plt.grid(which="both", alpha=0.2)
    # plt.xlabel("$t_C$ (ms)", fontsize=12)
    # plt.ylabel("$M_NOGSE$", fontsize=12)

    # ax[i].set_title(f"$k = {k}$")
    # ax[i].legend(fontsize=12)
    # ax[i].xaxis.set_minor_locator(AutoMinorLocator())
    # ax[i].yaxis.set_minor_locator(AutoMinorLocator())
    # ax[i].tick_params(
    #     axis="both", which="both", direction="in", top=True, right=True, labelsize=12
    # )
    # ax[i].grid(which="both", alpha=0.2)
    # ax[i].set_xlabel("Tiempo", fontsize=12)
    # ax[i].set_ylabel("Fase", fontsize=12)
