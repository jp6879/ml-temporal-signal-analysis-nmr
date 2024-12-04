# Programa para la generación de una base de datos de señales con ciertos parámetros
# Autor: Juan Pablo Morales

# Importación de paquetes de Julia
using CSV
using DataFrames
using Plots
# Importamos el archivo de los parámetros
include("./Parameters.jl")
# Importamos el archivo para la generación de señales
include("./Signals.jl")

# Ahora generamos los datos en CSV para cada combinación de parámetros en el path especificado, este va a ser el mismo que use para leer los datos
function genCSVSignalHahn(path :: String)
    """Función que genera un archivo CSV con las señales de Hahn para cada combinación de parámetros
    recibe los parámetros de la simulación desde el archivo Parameters.jl

    Parámetros:
        path (String): ruta donde se guardan los archivos CSV

    Retorna:
        Archivo CSV con las señales de Hahn
    """    

    # Creamos un arreglo para guardar las señales
    Signals = zeros(length(lcms) * length(σs), length(times))

    # Ponemos en las columnas 
    for (i, lcm) in enumerate(lcms)
        for (j, σ) in enumerate(σs)
            S = genSignalHanh(lcm, σ, times, G)
            Signals[(i - 1) * length(σs) + j, :] = S
        end
    end

    # Creamos el directorio si no existe
    if !isdir(path)
        mkdir(path)
    end

    # Guardamos los datos en un archivo CSV
    df = DataFrame(Signals, :auto)
    CSV.write(joinpath(path, "SimpleSignalHahn_TE_$(te)_G_$(G).csv"), df)

end

function main(path, gen = true, lcm = 0.5, σ = 0.5)
    """Función principal que genera los datos de las señales de Hahn
    Parámetros:
        gen (Bool): si es verdadero genera los datos, si es falso no
        path (String): ruta donde se guardan los archivos CSV
        lcm (float): tamaño medio de compartimiento
        σ (float): desviación estándar de compartimiento
    Retorna:
        Archivos CSV con las señales de Hahn
    """
    if gen
        println("Generando datos")
        genCSVSignalHahn(path)
    else
        S = genSignalHanh(lcm, σ, times, G)
        pl = plot(times, S, label = "Signal\nG = $G\nσ = $σ\nlcm=$lcm", xlabel = "t [s]", ylabel = "S(t)", title = "Señal de Hahn",  lw = 2, tickfontsize=12, labelfontsize=15, legendfontsize=11, framestyle =:box, gridlinewidth=1, xminorticks=10, yminorticks=10)
        display(pl)
        savefig(pl, "SignalHahn.png")
    end
end

# Modo de uso

# Si se quiere generar los datos
main("./Data/")

# Si se quiere visualizar una señal
# main("./Data/", false, 1.5, 0.5)