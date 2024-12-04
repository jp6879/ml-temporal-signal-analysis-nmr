
function DerivateSignals(t, signal)
    """Función que calcula las derivadas de las señales	a partir de diferencias finitas centradas
    Args:
        t (Vector{Float}) : Vector de tiempos
        signal (Vector{Float}) : Señal a derivar
    Returns:
        derivadas (Vector{Float}) : Derivadas de la señal
    """
    # Calcula el tamaño de la ventana
    w = 1
    # Calcula el tamaño de la señal
    n = length(signal)
    # Inicializa el arreglo de derivadas
    derivadas = zeros(n)
    for i in 1:n
        # Encuentra los índices de la ventana
        inicio = max(1, i-w)
        final = min(n, i+w)
        # Utiliza diferencias finitas centradas si es posible
        if inicio != i && final != i
            derivadas[i] = (signal[final] - signal[inicio]) / (t[final] - t[inicio])
        elseif inicio == i
            # Diferencia hacia adelante si estamos en el comienzo del arreglo
            derivadas[i] = (signal[i+1] - signal[i]) / (t[i+1] - t[i])
        else
            # Diferencia hacia atrás si estamos al final del arreglo
            derivadas[i] = (signal[i] - signal[i-1]) / (t[i] - t[i-1])
        end
    end
    return derivadas
end

# Función que obtiene las derivadas de las señales y las devuelve normalizadas
function GetSignalsDeriv(t, Signals)
    Signals_derivadas= zeros(size(Signals))

    # Obtenemos las derivadas de las señales de validación
    for i in 1:size(Signals)[1]
        Signals_derivadas[i,:] = DerivateSignals(t,Signals[i,:])
    end

    # La transponemos y la convertimos a Float32
    Signals_derivadas = Float32.(Matrix(Signals_derivadas'))

    # Normalizamos las derivadas
    for i in 1:size(Signals)[1]
        Signals_derivadas[:,i] = Signals_derivadas[:,i] ./ maximum(abs.(Signals_derivadas[:,i]))
    end

    return Signals_derivadas
end

# Función que devuelve las funciones interpoladoras con los datos de las señales entrantes
# Las señales entran como un arreglo de arreglos y los tiempos como un arreglo
function get_interpolated(t, Signals)
    itp = []
    # Interpolamos las derivadas
    for i in 1:size(Signals)[2]
        interpol = BSplineKit.interpolate(t, Signals[:,i], BSplineOrder(2))
        extrapol = BSplineKit.extrapolate(interpol, Smooth())
        push!(itp, extrapol)
    end
    return itp
end

function GetSignalsSampled(path, indexes, muestreo_corto, muestreo_largo)
    
    signalsTotal = Float32.(Matrix(CSV.read(path, DataFrame))[:,indexes])
    signalsTotal = transpose(signalsTotal)
    
    signalsSampledShort = signalsTotal[:,1:muestreo_corto:1000]
    signalsSampledLong = signalsTotal[:,1001:muestreo_largo:end]

    singalsSampled = hcat(signalsSampledShort, signalsSampledLong)

    return Float32.(singalsSampled)
end


function GetSignalsDataSet(path, parameters, sampled_sigmas, sampled_lcms, muestreo_corto, muestreo_largo, t)
    """Función para obtener un conjunto de señales sampleadas a partir de rangos de sigmas y lcms
    Args:
        path (string) : Ruta del archivo CSV con las señales
        parameters (DataFrame) : DataFrame con los parámetros de todas las señales generadas
        sampled_sigmas (Vector{Float32}) : Sigmas sampleados
        sampled_lcms (Vector{Float32}) : LCMs sampleados
        muestreo_corto (Int) : Cantidad de muestras a tomar para los tiempos entre 0 y 0.1 segundos
        muestreo_largo (Int) : Cantidad de muestras a tomar para los tiempos entre 0.1 y 1 segundos
        t (Vector{Float}) : Vector de tiempos
    Returns:
        Signals_rep (Matrix{Float32}) : Señales representativas
        Signals_rep_derivadas (Matrix{Float32}) : Derivadas de las señales representativas
        column_lcm_rep (Vector{Float32}) : LCMs de las señales representativas
        column_sigmas_rep (Vector{Float32}) : Sigmas de las señales representativas

    """

    indexes = []

    for (sigma, lcm) in zip(sampled_sigmas, sampled_lcms)
        row_indices = findall(row -> row.sigmas == sigma && row.lcms == lcm, eachrow(parameters))
        push!(indexes, row_indices)
    end

    # for sigma in sampled_sigmas
    #     for lcm in sampled_lcms
    #         # Find the row where sigma == value1 and lcm == value2
    #         row_indices = findall(row -> row.sigmas == sigma && row.lcms == lcm, eachrow(parameters))
    #         # Extract the rows based on the indices
    #         push!(indexes, row_indices)
    #     end
    # end

    indexes = vcat(indexes...)

    # Obtenemos las señales sampleadas
    signalsSampled = GetSignalsSampled(path, indexes, muestreo_corto, muestreo_largo)

    return signalsSampled
end
