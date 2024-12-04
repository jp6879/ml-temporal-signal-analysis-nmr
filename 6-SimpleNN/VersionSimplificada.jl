using Flux
using Flux: train!
using DataFrames
using CSV
using SciMLSensitivity
using ComponentArrays, Optimization, OptimizationOptimJL, OptimizationFlux
using IterTools: ncycle
using MLDataUtils
using Random
using Plots

include("../1-GeneracionDatos/Parameters.jl")
path_read = "C:/Users/Propietario/OneDrive/Escritorio/ib/Tesis_V1/MLonNMR/1-GeneracionDatos/Data/SignalHahn_TE_1_G_8.73e-7_forPCA.csv"

function GetSignals(path_read)
    """Función que lee las señales desde un archivo CSV
    Args:
        path_read (string): Ruta del archivo CSV de señales simuladas
    Returns:
        dataSignals (Matrix): Matriz con las señales en las columnas
    """
    dataSignals = CSV.read(path_read, DataFrame)
    dataSignals = Matrix(dataSignals)
    return dataSignals
end

function LabelSignals(lcms, σs)
    """Función que etiqueta las señales con los parámetros con los que se generaron
    Args:
        lcms (range): Rango de lcms
        σs (range): Rango de σs
    Returns:
        column_lcm (Array): Arreglo de lcms utilizados
        column_sigmas (Array): Arreglo de σs utilizados
    """
    dim_lcm = length(lcms)
    dim_sigma = length(σs)

    column_lcm = zeros(dim_lcm*dim_sigma)
    column_sigma = zeros(dim_lcm*dim_sigma)
    column_index = zeros(dim_lcm*dim_sigma)

    aux_lcms = collect(lcms)
    aux_sigmas = collect(σs)

    for i in 1:dim_lcm
        for j in 1:dim_sigma
            column_index[(i - 1)*dim_sigma + j] = (i - 1)*dim_sigma + j
            column_lcm[(i - 1)*dim_sigma + j] = aux_lcms[i]
            column_sigma[(i - 1)*dim_sigma + j] = aux_sigmas[j]
        end
    end

    return DataFrame(
        idx = column_index,
        lcm = column_lcm,
        σ = column_sigma
    )
end

function TrainTestSplit(dataset, perTest = 0.2)
    n = size(dataset)[1]
    println(n)
    idx = shuffle(1:n)
    at = 1 - perTest
    train_idx = view(idx, 1:floor(Int, at*n))
    test_idx = view(idx, (floor(Int, at*n)+1):n)
    dataset[train_idx,:], dataset[test_idx,:], train_idx, test_idx
end

function GetSignalsSampled(dataParams, shortTimeRange, longTimeRange)
    indexes = Int.(dataParams[!,1])
    someSignals = Float32.(transpose(Matrix(dataSignals[:,indexes])))
    someSignalsShort = someSignals[:,1:shortTimeRange:1000]
    someSignalsLong = someSignals[:,1001:longTimeRange:end]
    someSignals = hcat(someSignalsShort, someSignalsLong)
    return someSignals
end

# Tenemos las simulaciones de estos datos
lcms = 0.5:0.01:6
σs = 0.01:0.01:1

lcms_explore = 0.5:0.01:6
σs_explore = 0.01:0.01:1

println("Cantidad de lcms: ", length(lcms_explore))
println("Cantidad de σs ", length(σs_explore))

dataSignals = GetSignals(path_read)
dataParams = LabelSignals(lcms_explore, σs_explore)

dataParamsTrain, dataParamsTest, train_idx, test_idx = TrainTestSplit(dataParams, 0.2)

println("Señales de entrenamiento: ")
for i in 1:size(dataParamsTrain)[1]
    println("lcm ", dataParamsTrain[i,:].lcm, " σ ", dataParamsTrain[i,:].σ)
end

println("Señales de prueba: ")
for i in 1:size(dataParamsTest)[1]
    println("lcm ", dataParamsTest[i,:].lcm, " σ ", dataParamsTest[i,:].σ)
end

# Muestreo de señales
muestreo_corto =  25 # Cada cuantos tiempos tomamos un timepo para entrenar la NODE
muestreo_largo = 10
# Esto da 50 tiempos 40 puntos desde 0 a 0.1 y 10 puntos desde 0.1 a 1
t_shortaux = t_short[1:muestreo_corto:end]
t_longaux = t_long[1:muestreo_largo:end]
t = vcat(t_shortaux, t_longaux)

signalsTrain = GetSignalsSampled(dataParamsTrain, muestreo_corto, muestreo_largo)
signalsTest = GetSignalsSampled(dataParamsTest, muestreo_corto, muestreo_largo)

# Numero de puntos para la predicción
n_usados = 25

# Paso para tomar los tiempos de entrenamiento y validación
step = floor(Int, length(t) / n_usados) + 1

t_usados = t[1:step:end]

# En la validación y en el train tenemos que tener el primer y último tiempo
t_usados = vcat(t_usados, t[end])

indexes_usados = [i for i in 1:length(t) if t[i] in t_usados]

signalsTrainPuntos = Float32.(transpose(signalsTrain[:,indexes_usados]))
signalsTestPuntos = Float32.(transpose(signalsTest[:,indexes_usados]))

dataParamsTrainY = Float32.(transpose(Matrix(dataParamsTrain[!, [:lcm, :σ]])))
dataParamsTestY = Float32.(transpose(Matrix(dataParamsTest[!, [:lcm, :σ]])))

model = Flux.Chain(Flux.Dense(size(signalsTrainPuntos, 1), 32, relu),
                Flux.Dense(32, 64, relu),
                Flux.Dense(64, 32, relu),
                Flux.Dense(32, 2, softplus))

batch_size = 10

dataTrain = Flux.DataLoader((signalsTrainPuntos, dataParamsTrainY), batchsize = batch_size, shuffle = true)
dataTest = Flux.DataLoader((signalsTestPuntos, dataParamsTestY), batchsize = batch_size, shuffle = true)

# Función de loss
function loss(x,y)
    return Flux.mse(model(x), y)
end


# Definimos una funcion de callback para ver el progreso del entrenamiento
iter = 0
callback = function ()
    global iter += 1
    if iter % length(dataTrain) == 0
        epoch = iter ÷ length(dataTrain)
        if epoch % 5 == 0
            actual_loss = loss(dataTrain.data[1], dataTrain.data[2])
            # actual_valid_loss = loss(dataValid.data[1], dataValid.data[2])
            # println("Epoch $epoch || Loss = $actual_loss || Valid Loss = $actual_valid_loss")
            println("Epoch $epoch || Loss = $actual_loss")
        end
    end
end;

opt = AdamW(0.1)

# Entrenamos la red neuronal con el loss mse variando la tasa de aprendizaje cada 500 épocas
for epoch in 1:epochs
    Flux.train!(loss, Flux.params(model, opt), dataTrain, opt, cb=callback)
    if epoch % 1000 == 0
        η = η * 0.2
        opt = ADAM(η)
    end
end
