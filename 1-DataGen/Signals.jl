# Programa de generación de señales de Hahn
# Autor: Juan Pablo Morales

# Incluimos los parametros de la simulación
include("./Parameters.jl")

##########################################################################################

# Distribución log-normal de tamaños de compartimientos lc
# Mediana lcm y desviación estándar σ
function P(lc, lcm, σ)
    """Función que genera un valor de la distribución log-normal
    Parámetros:
        lc (float): tamaño de compartimiento
        lcm (float): tamaño medio de compartimiento
        σ (float): desviación estándar de compartimiento

    Retorna:
        P(lc) (float): distribución log-normal
    """
    return ( exp( -(log(lc) - log(lcm))^2 / (2σ^2) ) ) / (lc*σ*sqrt(2π))
end

# Magnetización de Hahn a tiempo t para un tamaño de compartimiento lc
# Si el tamaño de los compartimientos es único entonces la señal detectada es simplemente la magnetización de Hahn M(t)
function Ml_Hahn(t, lc, G)
    """Función que genera la magnetización de Hahn a tiempo t
    Parámetros:
        t (float): tiempo actual de simulación
        lc (float): tamaño de compartimiento
        G (float): gradiente de campo magnético
    Retorna:
        Ml_Hahn(t, lc) (float): magnetización de Hahn a tiempo t
    """
    τc = lc^2 / (2 * D0)
    term1 = -γ^2 * G^2 * D0 * τc^2
    term2 = t - τc * (3 + exp(-t / τc) - 4 * exp(-t / (2 * τc)))
    return exp(term1 * term2)
end


# Señal de Hahn a tiempo t
# La señal detectada S(t) es la suma de las señales de Hahn ponderadas por la distribución de tamaños de compartimientos P(lc)
function S_Hahn(lcm, σ, t, G)
    """Función que genera la señal detectada a tiempo t: S(t)
    Parámetros:
        lcm (float): tamaño medio de compartimiento
        σ (float): desviación estándar de compartimiento
        t (float): tiempo actual de simulación
        G (float): gradiente de campo magnético
    Retorna:
        S: señal detectada a tiempo t
    """
    # Generamos la distribución y magnetización de Hahn para cada tamaño de compartimiento (parameters.jl)
    P_lc = P.(lcs, lcm, σ) # Consideramos media lcm y ancho σ    
    M_lc = Ml_Hahn.(t, lcs, G) # Calculamos M_lc(t) para cada tamaño de compartimiento
    
    S = sum(M_lc .* P_lc) # La señal detectada es la suma de las señales de Hahn ponderadas por la distribución de tamaños de compartimientos P(lc)
    return S
end


# Generamos los datos de la señal para los tiempos times (parameters.jl)
function genSignalHanh(lcm, σ, times, G)
    """Función que genera las señales de Hahn normalizadas
    Parámetros:
        times (Array{float}): arreglo de tiempos
        lcm (float): tamaño medio de compartimiento
        σ (float): desviación estándar de compartimiento
        G (float): gradiente de campo magnético
    """

    # Generamos la señal de Hahn en tiempo 0
    S0 = S_Hahn(lcm, σ, 0, G)
    S = S_Hahn.(lcm, σ, times, G) ./ S0

    return S
end

# Señal NOGSE

#TODO: Implementar función de señal con secuencia NOGSE