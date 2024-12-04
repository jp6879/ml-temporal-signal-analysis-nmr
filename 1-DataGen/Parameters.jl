# En este momento decidimos cuales son los parámetros que dejamos fijos al momento de generar los datos
# Se dejan fijos el numero de compartimientos N, el rango de tamaños de compartimientos l0 y lf, el tiempo final de simulación lf
# Además se muestrea el tiempo en dos partes, una corta y una larga para tener una mejor resolución en la parte inicial de la señal
# Siendo esta imporatnte para diferenciar las señales de diferentes distribuciones de tamaños de compartimientos en el caso de que estas sean muy similares (lcm y σ grandes)

# Constantes útiles para el cálculo de las seañes 
# Las unidades utilzadas estan en μm, s y T

# Constantes físicas
γ = 2.675e8  # Factor girómagetico del esín nuclear del proton (s⁻¹T⁻¹) de https://physics.nist.gov/cgi-bin/cuu/Value?gammap
D0 = 1e3 # Coeficiente de difusión (μm²/s) considerado

# Parámetros experimentales
G = 8.73e-7  # Gradiente externo (T/μm) de Validating NOGSE’s size distribution predictions in yeast cells Paper 1
te = 1 # Tiempo final de simulación en s

# Parámetros fijos
N = 5000 # Número de compartimientos para la distribución de tamaños de compartimientos
time_sample_lenght_short = 1000 # Número de puntos en el muestreo corto
time_sample_lenght_long = 100 # Número de puntos en el muestreo largo


# Parametros que se varian, estos se corresponden a la mediana y la desviación estándar de la distribución de tamaños de compartimientos lcms en μm y σs adimensionales
# Esto da la cantidad de combinaciones de parámetros que se pueden tener para generar las señales
lcms_min = 0.5
lcms_step = 0.01
lcms_max = 6

σs_min = 0.01
σ_step = 0.01
σs_max = 1

# Rangos de los parámetros de las distribuciones de tamaños de compartimientos
lcms = lcms_min:lcms_step:lcms_max # Esto nos da un muestreo de 0,01 μm en lcm
σs = σs_min:σ_step:σs_max # Esto nos da un muestreo de 0,01 en σ

# Ahora generamos los datos para eso necesitamos hacer el sampling de los lcs y los t

# Para la generación de P(lc) necesitamos un muestreo de los tamaños de compartimientos lc
# Rango de tamaños de compartimientos en μm
l0 = 0.01 # Tamaño mínimo de compartimiento en μm
lf = 45 # Tamaño máximo de compartimiento en μm
lcs = range(l0, lf, length = N) # Esto nos da un muestreo de 0,008998 μm en lc

# Para la generación de S(t) necesitamos un muestreo del tiempo t
t_short = collect(range(0, 0.1, length = time_sample_lenght_short)) # Muestreo corto de 0.1 ms
t_long = collect(range(0.1, te, length = time_sample_lenght_long)) # Muestreo largo de 10 ms

# Concatenamos los tiempos para tener un muestreo completo 
times = vcat(t_short, t_long)
times = unique(times) # Eliminamos los tiempos repetidos