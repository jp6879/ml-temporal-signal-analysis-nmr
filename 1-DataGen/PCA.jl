# Programa que realiza el análisis de componentes principales (PCA) para los datos generados.
# Autor: Juan Pablo Morales

# Importamos los paquetes necesarios
using Plots
using Measures
using MultivariateStats
using DataFrames
using CSV
using Statistics

include("./Parameters.jl")

# Leemos los datos de las señales que generamos

function getSignals(path)
    """Función que lee los datos que generamos
    Parametros
        path: string con la dirección donde se encuentran los datos
    Retorna
        dataSignals: matriz con los datos de las señales NxM donde M es el número de señales y N el número de datos por señal (formato que pide PCA)
    """

    # Leemos los datos
    dataSignals = Matrix(CSV.read(path, DataFrame))
    
    return dataSignals
end

# En cada columna tenmos los datos de las señales, centramos estas columnas para que tengan media 0
function centerData(matrix)
    """Función que centra los datos de las columnas de una matriz para que tengan media 0
    Parametros
        matrix: matriz con los datos a centrar
    Retorna
        centered_data: matriz con los datos centrados
    """
    col_mean = mean(matrix, dims=1)
    centered_data = matrix .- col_mean
    return centered_data
    
end

# Función que realiza PCA sobre los datos de entrada y grafica la varianza explicada por cada componente principal
function dataPCA(dataIN)
    """Función que realiza PCA sobre los datos de entrada y grafica la varianza explicada por cada componente principal

    Parametros
        dataIN: matriz con los datos a los que se les va a realizar PCA
    Retorna
        reduced_dataIN: datos reducidos por PCA
        pca_model: modelo de PCA que se puede usar para reconstruir los datos originales, además contiene información sobre los componentes principales
    """

    # Primero centramos los datos
    dataIN_C = centerData(dataIN)

    # Esto ya hace PCA sobre la matriz dada donde cada observación es una columna de la matriz
    pca_model = fit(PCA, dataIN_C)

    # Esta instancia de PCA tiene distintas funciones como las siguientes

    #projIN = projection(pca_model) # Proyección de los datos sobre los componentes principales

    # Vector con las contribuciones de cada componente (es decir los autovalores)
    pcsIN = principalvars(pca_model)

    # Obtenemos la variaza en porcentaje para cada componente principal
    explained_varianceIN = pcsIN / sum(pcsIN) * 100

    # Grafiquemos esto para ver que tan importante es cada componente principal
    
    if !isdir("./Plots_PCA")
        mkdir("./Plots_PCA")
    end

    explainedVar = Plots.bar(explained_varianceIN, title="Varianza en porcentaje datos entrada",label = false, xlabel="Componente principal", ylabel="Varianza (%)")
    Plots.savefig("./Plots_PCA/ExplainedVariance_G_$(G)_te_$(te).png")

    reducedIN = MultivariateStats.transform(pca_model, dataIN_C)

    return reducedIN, pca_model
    
end

function main(path_read, path_save, maxVarS)
    """Función principal que realiza PCA sobre los datos de las señales y las distribuciones de probabilidad
    Parametros
        Path (string): dirección donde se encuentran los datos
        maxVarS (float): varianza acumulada deseada para las señales
    Retorna
        nothing
    """
    
    dataSignals = getSignals(path_read)

    # Realizamos PCA sobre los datos de las señales y las distribuciones de probabilidad y guardamos los datos reducidos y el modelo de PCA
    reduced_dataSignals, pca_model_signals = dataPCA(dataSignals)

    pl = Plots.plot(cumsum(principalvars(pca_model_signals)) / sum(principalvars(pca_model_signals)) * 100, label = "Varianza acumulada señales", legend = :bottomright, xlabel = "Componentes principales tomadas", ylabel = "Varianza acumulada (%)", tickfontsize=11, labelfontsize=13, legendfontsize=9, framestyle =:box, gridlinewidth=1, xminorticks=10, yminorticks=10, right_margin=5mm, marker = "o")
    savefig(pl, "./Plots_PCA/CumulativeVarianceSignals_G_$(G)_te_$(te).png")

    # Quiero ver hasta que componente hay una varianza acumulada del 98% para las señales y del 80% para las distribuciones de probabilidad
    pcs_vars_s = principalvars(pca_model_signals)

    limdim_S = 0
    # Buscamos el número de componentes principales que nos da la varianza acumulada deseada
    for i in 1:length(pcs_vars_s)
        if sum(pcs_vars_s[1:i]) / sum(pcs_vars_s) * 100 > maxVarS
            println("La varianza acumulada de las señales es del ", sum(pcs_vars_s[1:i]) / sum(pcs_vars_s) * 100, "% con ", i, " componentes principales")
            limdim_S = i
            break
        end
    end

    
    # Guardamos los datos de los componentes principales en un DataFrame
    df_PCA_Signals = DataFrame(reduced_dataSignals, :auto)

    # Limitamos el número de componentes principales a los que nos dan la varianza acumulada deseada
    df_PCA_Signals = df_PCA_Signals[1:limdim_S,:]

    # Identificación de los datos reducidos según los parámetros utilizados para generar los datos originales lcm y σ
    dimlcm = length(lcms)
    dimσ = length(σs)

    column_lcm = zeros(dimlcm*dimσ)
    column_σs = zeros(dimlcm*dimσ)
    aux_lcm = collect(lcms)
    aux_σs = collect(σs)

    for i in 1:dimlcm
        for j in 1:dimσ
            column_lcm[(i - 1)*dimσ + j] = aux_lcm[i]
            column_σs[(i - 1)*dimσ + j] = aux_σs[j]
        end
    end

    # Guardamos tres componentes principales y la identificacion en CSV

    df_PCA_Signals = DataFrame(
        pc1 = reduced_dataSignals[1, :],
        pc2 = reduced_dataSignals[2, :],
        pc3 = reduced_dataSignals[3, :],
        σs = column_σs,
        lcm = column_lcm,)
    
    # Guardamos estos datos en CSV
    CSV.write(path_save * "PCA_Signals_G_$(G)_TE_$(te).csv", df_PCA_Signals)

    return nothing
end

# main("./Data/SignalHahn_TE_1_G_8.73e-7_forPCA.csv", "./Data/", 98.0)

##########################################################################################
# Esto permite reconstruir los datos originales a partir de los datos reducidos
# Datos reconstruidos

#reconstruct(M::PCA, y::AbstractVecOrMat{<:Real})

dataSignals = getSignals("C:/Users/Propietario/OneDrive/Escritorio/ib/Tesis_V1/MLonNMR/1-GeneracionDatos/Data/SignalHahn_TE_1_G_8.73e-7_forPCA.csv")

# Realizamos PCA sobre los datos de las señales y las distribuciones de probabilidad y guardamos los datos reducidos y el modelo de PCA
reduced_dataSignals, pca_model_signals = dataPCA(dataSignals)

dataSignals

re_signals = reconstruct(pca_model_signals, reduced_dataSignals)

using Flux: mse

# Calculamos el error cuadrático medio entre los datos originales y los datos reconstruidos
error = mse(re_signals, dataSignals)

# Ejemplo de reconstrucción de datos originales a partir de los datos reducidos

# Plots.scatter(times, re_signals[:,0*100 + 1], label = "lcm = $(lcms[1]), σ = $(σs[1])", markersize = 2)
# Plots.scatter!(times, dataSignals[:,0*100 + 1], label = "lcm = $(lcms[1]), σ = $(σs[1]) real", markersize = 2)
# Plots.scatter!(times, re_signals[:,0*100 + 20], label = "lcm = $(lcms[1]), σ = $(σs[20])", markersize = 2)
# Plots.scatter!(times, re_signals[:,0*100 + 100], label = "lcm = $(lcms[1]), σ = $(σs[100])", markersize = 2)
# Plots.scatter!(t,re_signals[:,(20 - 1)*100 + 1], label = "lcm = $(lcms[20]), σ = $(σs[1])", markersize = 2)
# Plots.scatter!(t,re_signals[:,(20 - 1)*100 + 100], label = "lcm = $(lcms[20]), σ = $(σs[100])", markersize = 2)

# Plots.scatter(lc,re_probd[:,(50-1)*100 + 20], label = "lcm = $(lcms[50]), σ = $(σs[20])", markersize = 0.5)
# Plots.scatter!(lc,re_probd[:,(60-1)*100 + 100], label = "lcm = $(lcms[50]), σ = $(σs[100])", markersize = 0.5)
# Plots.scatter!(lc,re_probd[:,(551 - 1)*100 + 1], label = "lcm = $(lcms[551]), σ = $(σs[1])", markersize = 0.5)
# Plots.scatter!(lc,re_probd[:,(551 - 1)*100 + 100], label = "lcm = $(lcms[551]), σ = $(σs[100])", markersize = 0.001)