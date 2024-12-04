"""

This script performs the following tasks:

1. **Imports necessary libraries**:
    - Flux: For building and training neural networks.
    - DataFrames, CSV: For handling data frames and CSV files.
    - SciMLSensitivity, ComponentArrays, Optimization, OptimizationOptimJL, OptimizationFlux: For sensitivity analysis and optimization.
    - IterTools, Random, StatsBase: For various utility functions.

2. **Includes external Julia files**:
    - Parameters.jl: Contains parameter definitions.
    - UtilidadesSeñales.jl: Contains utility functions for signal processing.

3. **Sets up the environment**:
    - Sets a random seed for reproducibility.
    - Defines the path to the CSV file containing signal data.
    - Defines ranges for correlation sizes (lcms) and standard deviations (sigmas).
    - Creates a DataFrame of parameters from the Cartesian product of lcms and sigmas.

4. **Defines functions**:
    - `SplitDataSets`: Splits the dataset into training, validation, and test sets.
    - `create_model`: Creates a neural network model with specified layers, activation function, and dropout rate.
    - `main`: The main function that orchestrates the entire process:
        - Defines various configurations for the neural network.
        - Splits the dataset.
        - Prepares the training, validation, and test signals.
        - Creates the neural network model.
        - Defines the loss function and optimizer.
        - Trains the model and saves the results.

5. **Main execution**:
    - Calls the `main` function to execute the entire process.
"""
using Flux
using Flux: train!
using DataFrames
using CSV
using SciMLSensitivity
using ComponentArrays, Optimization, OptimizationOptimJL, OptimizationFlux
using IterTools: ncycle
using Random
using StatsBase

# include("/home/juan.morales/VersionSimpleExploracion/Parameters.jl");
# include("/home/juan.morales/VersionSimpleExploracion/UtilidadesSeñales.jl")
include("./Parameters.jl")
include("./UtilidadesSeñales.jl")


rng = Random.seed!(1234)  # Set seed for reproducibility
path_read = "C:/Users/Propietario/OneDrive/Escritorio/ib/Tesis_V1/MLonNMR/1-GeneracionDatos/Data/SignalHahn_TE_1_G_8.73e-7_forPCA.csv"
# path_read = "/home/juan.morales/datos_PCA/dataSignals.csv"

# Rango de tamaños medios de correlación en μm
lcms = 0.5:0.01:6
# Desviaciones estándar
sigmas = 0.01:0.01:1

parameters = zeros(length(lcms) * length(sigmas), 2)

# Obtenemos los parámetros de las señales
for i in 1:length(lcms)
    for j in 1:length(sigmas)
        parameters[(i - 1) * length(sigmas) + j, 1] = sigmas[j]
        parameters[(i - 1) * length(sigmas) + j, 2] = lcms[i]
    end
end

parameters = DataFrame(
    sigmas = parameters[:,1],
    lcms = parameters[:,2],
)

function SplitDataSets(n_samples, lcms, sigmas, train_ratio, val_ratio)

    # Generate the Cartesian product of lcms and sigmas
    pairs = [(lcm, sigma) for lcm in lcms, sigma in sigmas]

    lower_bound = first(pairs)
    upper_bound = last(pairs)
    middle_pairs = pairs[2:end-1]

    # Downsample the middle values
    downsampled_middle = sample(rng, middle_pairs, n_samples - 2, replace = false)

    # Combine lower, middle, and upper pairs
    downsampled_pairs = [lower_bound; downsampled_middle; upper_bound]

    # Shuffle the downsampled pairs
    shuffled_pairs = shuffle(rng, downsampled_pairs)

    n_total = length(shuffled_pairs)
    n_train = round(Int, train_ratio * n_total)
    n_val = round(Int, val_ratio * n_total)
    n_test = n_total - n_train - n_val  # Ensure all elements are used
    
    # Split the shuffled pairs
    train_set = shuffled_pairs[1:n_train]
    val_set = shuffled_pairs[n_train+1:n_train+n_val]
    test_set = shuffled_pairs[n_train+n_val+1:end]
    
    return train_set, val_set, test_set

end

function create_model(layers, activation, dropout_rate)
    """Función para crear el modelo de la red neuronal
    Args:
        layers (Array{Int}): Arreglo con la cantidad de neuronas por capa
        activation (Function): Función de activación
        dropout_rate (Float): Porcentaje de dropout
    
    Returns:
        model (Chain): Modelo de la red neuronal
    """
    activations = [activation for i in 1:length(layers) - 2]
    startlayer = Flux.Dense(layers[1], layers[2])
    hiddenlayers = hcat([[Flux.Dense(layers[i], layers[i+1], activations[i]), Dropout(dropout_rate)] for i in 2:length(layers) - 2]...)
    endlayer = Flux.Dense(layers[end-1], layers[end], softplus)
    return Flux.Chain(startlayer, hiddenlayers..., endlayer)
end

function main()
    # Puntos de la señal usados
    n_puntos_usados = [10, 20, 30]

    # Optimizadores que vamos a utilizar
    optimizers = [opt for opt in [Flux.Optimise.AdamW]]
    
    # Dropout rates
    dropout_rates = [0.0]

    # Lambdas regularizacion L2
    lambdas = [0.0, 0.1]

    # Numero de señales en el dataset
    num_signals = [55100]

    configuraciones = []

    for n_puntos in n_puntos_usados, opt in optimizers, lambd in lambdas, dropout_rate in dropout_rates, num_signal in num_signals
        if n_puntos == 5 || n_puntos == 10
            input_size = n_puntos + 1
        elseif n_puntos == 30
            input_size = n_puntos - 1
        else
            input_size = n_puntos
        end
        
            architectures = [
            [[input_size, 16, 32, 16, 2], relu], # Tres capas ocultas
            [[input_size, 32, 64, 16, 2], relu], # Tres capas ocultas con aun más neuonras
            [[input_size, 16, 32, 16, 8, 2], relu], # Cuatro capas ocultas
            [[input_size, 32, 64, 16, 8, 2], relu], # Cuatro capas ocultas mas neuronas
            [[input_size, 30, 25, 20, 15, 10, 2], relu], # Cinco capas ocultas
            [[input_size, 32, 64, 100, 64, 32, 2], relu], # Cinco capas ocultas mas neuronass
            ]
            
        for architecture in architectures
            push!(configuraciones, [n_puntos, opt, lambd, dropout_rate, architecture, num_signal])
        end
    end

    println("Cantidad de configuraciones: ", length(configuraciones))

    # id = parse(Int128, ARGS[1])
    # n_usados, opt, lambd, dropout_rate, architecture, NUMSIGNALS = configuraciones[parse(Int128, ARGS[1])]
    id = 1
    n_usados, opt, lambd, dropout_rate, architecture, NUMSIGNALS = configuraciones[1]
    
    println("Puntos usados: ", n_usados, " Optimizador: ", opt, " Lambda: ", lambd, " Dropout rate: ", dropout_rate, " Arquitectura: ", architecture, "Numero de señales: ", NUMSIGNALS)

    # División del dataset en entrenamiento, validación y test
    train_set, val_set, test_set = SplitDataSets(NUMSIGNALS, lcms, sigmas, 0.8, 0.1)

    train_lcms = map(x -> x[1], train_set)
    train_sigmas = map(x -> x[2], train_set)

    val_lcms = map(x -> x[1], val_set)
    val_sigmas = map(x -> x[2], val_set)

    test_lcms = map(x -> x[1], test_set)
    test_sigmas = map(x -> x[2], test_set)

    # Vamos a tomar un subconjunto de t para hacer el entrenamiento de la NODE para agilizar los tiempos de entrenamiento
    muestreo_corto = 55 # Cada cuantos tiempos tomamos un timepo para entrenar la NODE
    muestreo_largo = 35

    # Esto da 60 tiempos 50 puntos desde 0 a 0.1 y 10 puntos desde 0.1 a 1
    t_shortaux = t_short[1:muestreo_corto:end]
    t_longaux = t_long[1:muestreo_largo:end]

    t = vcat(t_shortaux, t_longaux)

    # Obtenemos las señales con el dominio de tiempo reducido
    signalsTrain = GetSignalsDataSet(path_read, parameters, train_sigmas, train_lcms, muestreo_corto, muestreo_largo, t)
    signalsVal = GetSignalsDataSet(path_read, parameters, val_sigmas, val_lcms, muestreo_corto, muestreo_largo, t)
    signalsTest = GetSignalsDataSet(path_read, parameters, test_sigmas, test_lcms, muestreo_corto, muestreo_largo, t)
    
    idx_end = length(t)

    n_usados_short = floor(Int, n_usados * 0.9)
    n_usados_long = ceil(Int, n_usados * 0.1)

    step_short = ceil(Int, 51 / n_usados_short)
    step_long = ceil(Int, 9 / n_usados_long)

    t_usados = t[1:step_short:51]
    t_usados = vcat(t_usados, t[51:step_long:idx_end])

    if t_usados[end] != t[end]
        push!(t_usados, t[end])
    end

    t_usados = unique(t_usados)
    indexes_usados = [i for i in 1:length(t) if t[i] in t_usados]

    println("Numero de puntos de entrada para $n_usados es $(length(indexes_usados))")

    signalsTrainPuntos = Float32.(transpose(signalsTrain[:,indexes_usados]))
    signalsValPuntos = Float32.(transpose(signalsVal[:,indexes_usados]))
    signalsTestPuntos = Float32.(transpose(signalsTest[:,indexes_usados]))
    
    dataParamsTrainY = zeros32(2, size(train_lcms, 1))
    dataParamsValidY = zeros32(2, size(val_lcms, 1))
    dataParamsTestY = zeros32(2, size(test_lcms, 1))

    dataParamsTrainY[1, :] = train_lcms
    dataParamsTrainY[2, :] = train_sigmas

    dataParamsValidY[1, :] = val_lcms
    dataParamsValidY[2, :] = val_sigmas

    dataParamsTestY[1, :] = test_lcms
    dataParamsTestY[2, :] = test_sigmas

    # Configuraciones
    layers = architecture[1]
    activation = architecture[2]

    # Para guardar los resultados
    if activation == relu
        activation_string = "relu"
    elseif activation == tanh_fast
        activation_string = "tanh"
    else
        activation_string = "swish"
    end
    
    # Para guardar los resultados
    if opt == AdamW
        opt_string = "AdamW"
    elseif opt == Adam
        opt_string = "Adam"
    end
    
    # Creamos el modelo con las configuraciones actuales
    model = create_model(layers, activation, dropout_rate)
    
    # Creamos los dataloaders
    dataTrain = Flux.DataLoader((signalsTrainPuntos, dataParamsTrainY), batchsize = n_usados, shuffle = false)
    dataValid = Flux.DataLoader((signalsValPuntos, dataParamsValidY), batchsize = n_usados, shuffle = false)
    dataTest = Flux.DataLoader((signalsTestPuntos, dataParamsTestY), batchsize = n_usados, shuffle = false)
    
    # L2 Penalty
    pen_l2(x::AbstractArray) = sum(abs2, x) / 2

    # Loss function
    function loss(x, y)
        return Flux.mse(model(x), y) + lambd * sum(pen_l2, Flux.params(model))
    end
    
    # Definimos una funcion de callback para ver el progreso del entrenamiento
    history_loss_train = []
    history_loss_val = []
    
    global iter = 0
    callback = function ()
        global iter += 1
        if iter % length(dataTrain) == 0
            epoch = iter ÷ length(dataTrain)
            push!(history_loss_train, loss(dataTrain.data[1], dataTrain.data[2]))
            push!(history_loss_val, loss(dataValid.data[1], dataValid.data[2]))
            
            if length(history_loss_val) > 10
                if history_loss_val[end] < history_loss_val[end - 5]
                    params_nn, re_nn = Flux.destructure(model)
                    df_theta = DataFrame(reshape(params_nn, length(params_nn), 1), :auto)
                    CSV.write("/home/juan.morales/VersionSimpleExploracion/Parameters/0$(id)_parameters_epoch_$(epoch).csv", df_theta)
                end
            end

            if epoch % 500 == 0
                actual_loss = loss(dataTrain.data[1], dataTrain.data[2])
                actual_val_loss = loss(dataValid.data[1], dataValid.data[2])
                println("Epoch $epoch || Loss = $actual_loss || Test Loss = $actual_val_loss")
            end
            GC.gc()
        end
    end
    
    η = 0.01
    epochs = 10000
    optim = opt(η)
    
    for epoch in 1:epochs
        Flux.train!(loss, Flux.params(model, optim), dataTrain, optim, cb=callback)
        if epoch % 1000 == 0
            println("Reducing learning rate on plateau")
            η = η * 0.2
            optim = opt(η)
        end
    end
    
    # Guardamos los hiperparámetros del entrenamiento y el loss final
    actual_layer = string(layers)
    actual_activation = activation_string
    actual_optimizer = opt_string
    actual_puntos_usados = n_usados
    actual_lambda = lambd
    actual_loss_final_train = round(history_loss_train[end], sigdigits = 4)
    actual_loss_final_val = round(history_loss_val[end], sigdigits = 4)
    actual_loss_test = round(loss(dataTest.data[1], dataTest.data[2]), sigdigits = 4)
    
    
    # Guardamos los resultados en un archivo csv
    df_results_total = DataFrame(ID = id, 
                                Arq = actual_layer, 
                                Activ = actual_activation,
                                Opt = actual_optimizer,
                                PuntosUsados = actual_puntos_usados,
                                Lambd = actual_lambda,
                                dp_rate = dropout_rate,
                                MSE_Train = actual_loss_final_train,
                                MSE_Val = actual_loss_final_val,
                                MSE_Test = actual_loss_test,
                                NSeñales = NUMSIGNALS
                                )
    
    
    df_losses = DataFrame(
        loss_train = history_loss_train,
        loss_val = history_loss_val
    )
    
    CSV.write("/home/juan.morales/VersionSimpleExploracion/Losses/0$(id)_losses.csv", df_losses)
    CSV.write("/home/juan.morales/VersionSimpleExploracion/Results/0$(id)_results.csv", df_results_total)
    
    # CSV.write("C:/Users/Propietario/OneDrive/Escritorio/ib/Tesis_V1/MLonNMR/5-Simplificación/Losses/0$(id)_losses.csv", df_losses)
    # CSV.write("C:/Users/Propietario/OneDrive/Escritorio/ib/Tesis_V1/MLonNMR/5-Simplificación/Results/0$(id)_results.csv", df_results_total)
    # CSV.write("C:/Users/Propietario/OneDrive/Escritorio/ib/Tesis_V1/MLonNMR/5-Simplificación/Parameters/0$(id)_parameters.csv", df_theta)
end


main()

