from Mnogse_Pdist import M_nogse, Plog, M_nogse_T2, SignalNOGSE

# Parametros de las posibles distribuciones de tamaños de restricción
# Mediana de la longitud de correlación
lcm_min = 0.5 # mu m
lcm_step = 0.1 # mu m
lcm_max = 6 # mu m
# Desviación estándar de la distribución
sigma_min = 0.01
sigma_step = 0.01
sigma_max = 1.0

# Longitud de correlación, simulamos un rango de 0.01 a 50 micrómetros con N_compartimientos = 5000
lc_min = 0.01 # mu m
lc_max = 50 # mu m
N_compartimientos = 5000

# Parametros experimentales de la secuencia NOGSE
TE = 54.3 # ms Tiempo total de evolución
T2 = 31.5 # ms Tiempo de relajación transversal
T = 21.5 # ms Tiempo total de la secuencia
Gs = [12.5, 35.5, 42.5, 50, 80] # Valores de gradientes G/cm
D0 = 23e-12 # cm^2/ms Coeficiente de difusión

N_refocusing_periods = 100

if __name__ == "__main__":
    # Testeamos la señal con el paper
    lcm_test = 1 #mu m
    sigma_test = 1.0
    G_test = 40 # G/cm
    T = 30 # ms
    N = 8
    D0 = 0.7e-8 # cm^2/ms

