# Progarma para la exploración de hiperparámetros en una Neural Ordinary Differential Equations (NODE) con Mini-Batchs
using Flux
using Flux: train!
using DataFrames
using CSV
using DifferentialEquations
using SciMLSensitivity
using ComponentArrays, Optimization, OptimizationOptimJL, OptimizationFlux
using Interpolations
using OrdinaryDiffEq
using IterTools: ncycle
using BSplineKit
using Random
using StatsBase
include("/home/juan.morales/PANODE/UtilidadesSeñales.jl")
include("/home/juan.morales/PANODE/Parameters.jl")

rng = Random.seed!(1234)

# Función que crea el modelo de la red neuronal que va a estar dentro de la ODE
# Se añade además un dropout en cada capa oculta
function create_model(layers::Vector{Int}, activation::Function, dropout_rate)
    activations = [activation for i in 1:length(layers) - 2]
    startlayer = Dense(layers[1], layers[2])
    # hiddenlayers = hcat([[Dense(layers[i], layers[i+1], activations[i]), Dropout(dropout_rate)] for i in 2:length(layers) - 2]...)
    hiddenlayers = hcat([[Dense(layers[i], layers[i+1], activations[i])] for i in 2:length(layers) - 2]...)
    endlayer = Dense(layers[end-1], layers[end])
    return Chain(startlayer, hiddenlayers..., endlayer)
end

# Función que entrena la NODE con mini-batchs
function Train_Neural_ODE(nn, U0, U0_valid, extra_input, extra_input_valid,
                            epochs, train_loader, opt, eta, signalsTrain, signalsVal, t, lambd, actual_id, extra_dim)
    
    # Tiempo sobre el cual resolver
    tspan = (0f0, 1f0)
    
    # Para entrenar la red tenemos que extraer los parametros de la red neuronal en su condicion inicial
    p_nn, re = Flux.destructure(nn) 

    # Si existe el archivo con los parámetros de la red previamente los cargamos
    if isfile("/home/juan.morales/PANODE/Parameters/0$(actual_id)_Parameters.csv")
        theta = CSV.read("/home/juan.morales/PANODE/Parameters/0$(actual_id)_Parameters.csv", DataFrame)
        p = @views Float32.(theta[:,1])
    else
        println("No se encontraron los parámetros de la red neuronal")
    end
    
    # Optimizardor
    optim = opt(eta)

    function SolveANODE(u0, extra, time_batch)

        # Definimos la función que resuelve la ODE
        function dSdt!(u, p, t)
            S = u[1]
            a = u[2]
            entrada = [t, S, a, extra...]
            return re(p)(entrada)
        end

        # Definimos el problema de la ODE
        prob = ODEProblem(dSdt!, u0, tspan)

        # Resolvemos la ODE, devolvemos únicamente la solución S(t)
        solution = @views Array(solve(prob, Tsit5(), p = p_nn, saveat = time_batch))[1, :]

        return solution
    end

    # Función que predice las señales para un conjunto de condiciones iniciales y parámetros extra
    function PredictSignals(U0, extra, time_batch)
        predicted_signals = map(1:length(U0[:,1])) do i
            u0 = Float32.(U0[i,:])
            if length(extra) > extra_dim
                SolveANODE(u0, extra[i, :], time_batch)
            else
                SolveANODE(u0, extra[1,:], time_batch)
            end
        end
        
        return @views transpose(reduce(hcat, predicted_signals))
    end
    
    function loss_node(batch, time_batch)
        y = PredictSignals(U0, extra_input, time_batch)
        return Flux.mse(y, batch)
    end
    
    function loss_valid(batch, time_batch)
        y = PredictSignals(U0_valid, extra_input_valid, time_batch)
        return Flux.mse(y, batch)
    end
    
    # Función de callback para guardar el loss en cada época
    global iter = 0
    loss = []
    loss_valid_array = []
    callback = function ()
        global iter += 1
        if iter % (length(train_loader)) == 0
            epoch = Int(iter / length(train_loader))
            actual_loss = loss_node(signalsTrain, t)
            loss_validacion = loss_valid(signalsVal, t)
            println("Epoch = $epoch || Loss: $actual_loss || Loss Valid: $loss_validacion")
            push!(loss, actual_loss)
            push!(loss_valid_array, loss_validacion)
        end
        GC.gc()
        return false
    end

    # Entrenamos la red neuronal con mini-batchs
    Flux.train!(loss_node, Flux.params(p_nn), ncycle(train_loader, epochs), optim, cb = callback)

    return loss, p_nn, loss_valid_array
end



function main()
    # Optimizadores que vamos a utilizar
    optimizers = [opt for opt in [Flux.Optimise.AdamW]]

    # Numero de mini-batchs que vamos a utilizar 
    batchs_size = [10, 15]

    # Dropout rates
    dropout_rates = [0.0]

    # Lambdas regularizacion L2
    lambdas = [0.0]

    # Learning rates
    learning_rates = [0.01]

    # Numero de entradas extra a la red neuronal
    n_usados_vector = [5, 8, 10]

    # Vector de configuraciones que vamos a utilizar
    configuraciones = []

    for opt in optimizers, batch_size in batchs_size, lambd in lambdas, eta in learning_rates, dropout_rate in dropout_rates, n_usados in n_usados_vector
        # Input t, S(t), a(t), extra_input = puntos medidos de la señal
        input_dim = 1 + 2 + (n_usados + 1)
        # Arquitecturas que vamos a utilizar
        architectures = [
            [[input_dim, 32, 64, 2], relu], # Dos capas ocultas simple
            [[input_dim, 64, 128, 2], swish], # Dos capas ocultas con mas neuronas
            [[input_dim, 32, 32, 16, 2], relu], # Tres capas ocultas simple
            [[input_dim, 128, 64, 16, 2], swish], # Tres capas ocultas con mas neuronas
            [[input_dim, 16, 32, 64, 16, 2], relu], # Cuatro capas ocultas simple
            [[input_dim, 32, 64, 32, 16, 2], swish], # Cuatro capas ocultas con mas neuronas
        ]
        for arch in architectures
            push!(configuraciones, (arch, opt, batch_size, lambd, eta, dropout_rate, n_usados))
        end
    end

    println("Cantidad de configuraciones: ", length(configuraciones))

    path_read = "/home/juan.morales/datos_PCA/SimpleSignalHahn_TE_1_G_8.73e-7.csv"
    # path_read = "C:/Users/Propietario/OneDrive/Escritorio/ib/Tesis_V1/MLonNMR/1-GeneracionDatos/Data/SimpleSignalHahn_TE_1_G_8.73e-7.csv"
        
    # Leemos los datos de las señales 
    signalsDF = transpose(Matrix(CSV.read(path_read, DataFrame)))

    # Obtenemos los parámetros de las señales para identificarlas posteriormente
    column_lcm = collect(lcms)
    column_sigma = collect(σs)
    pdistparamsDF = zeros(size(signalsDF)[2], 2)

    for (i, lcm) in enumerate(column_lcm)
        for (j, sigma) in enumerate(column_sigma)
            pdistparamsDF[(i - 1) * length(σs) + j, 1] = sigma
            pdistparamsDF[(i - 1) * length(σs) + j, 2] = lcm
        end
    end

    pdistparamsDF = DataFrame(pdistparamsDF, [:sigma, :lcm]);

    # Mezclamos los datos para que no haya sesgo en el entrenamiento
    perm = shuffle(rng, 1:size(signalsDF, 2))

    signalsDF = signalsDF[:, perm]
    pdistparamsDF = pdistparamsDF[perm, :];

    # Split en entrenamiento, validación y test
    n_signals = size(signalsDF, 2)
    n_train = Int(floor(n_signals*0.7))
    n_val = Int(floor(n_signals*0.15))
    n_test = n_signals - n_train - n_val

    train_signals = Float32.(Matrix(signalsDF[:, 1:n_train]))
    val_signals = Float32.(Matrix(signalsDF[:, n_train+1:n_train+n_val]))
    test_signals = Float32.(Matrix(signalsDF[:, n_train+n_val+1:end]))

    train_params = pdistparamsDF[1:n_train, :]
    val_params = pdistparamsDF[n_train+1:n_train+n_val, :]
    test_params = pdistparamsDF[n_train+n_val+1:end, :];

    # Vamos a tomar un subconjunto de t para hacer el entrenamiento de la NODE para agilizar los tiempos de entrenamiento
    muestreo_corto =  20 # Cada cuantos tiempos tomamos un timepo para entrenar la NODE
    muestreo_largo = 10

    # Esto da 60 tiempos 50 puntos desde 0 a 0.1 y 10 puntos desde 0.1 a 1
    t_shortaux = t_short[1:muestreo_corto:end]
    t_longaux = t_long[1:muestreo_largo:end]

    t = vcat(t_shortaux, t_longaux)

    indexes_t = [i for i in 1:length(times) if times[i] in t]

    # Obtenemos las señales con el dominio de tiempo reducido
    signalsTrain = transpose(train_signals[indexes_t, :])
    signalsVal = transpose(val_signals[indexes_t, :])
    signalsTest = transpose(test_signals[indexes_t, :])

    # Vamos a tomar un subconjunto de t para hacer el entrenamiento de la NODE para agilizar los tiempos de entrenamiento
    muestreo_corto = 20 # Cada cuantos tiempos tomamos un timepo para entrenar la NODE
    muestreo_largo = 10

    # Esto da 60 tiempos 50 puntos desde 0 a 0.1 y 10 puntos desde 0.1 a 1
    t_shortaux = t_short[1:muestreo_corto:end]
    t_longaux = t_long[1:muestreo_largo:end]
    
    t = vcat(t_shortaux, t_longaux)

    # Para el entrenamiento en el cluster vamos iterando sobre las configuraciones y guardamos los resultados en archivos csv
    # architecture, opt, batch_size, lambd, eta, dropout_rate, n_usados = configuraciones[1]
    architecture, opt, batch_size, lambd, eta, dropout_rate, n_usados = configuraciones[parse(Int128, ARGS[1])]

    # Número de modelo
    # actual_id = 1
    actual_id = parse(Int128,ARGS[1])

    idx_end = 60

    println("Numero de puntos usados para la predicción ", n_usados, " hasta el tiempo $(t[idx_end]) ", )

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

    signalsTrainPuntos = signalsTrain[:,indexes_usados]
    signalsValPuntos = signalsVal[:,indexes_usados]
    signalsTestPuntos = signalsTest[:,indexes_usados]

    extra_input = signalsTrainPuntos
    extra_input_valid = signalsValPuntos
    extra_input_test = signalsTestPuntos

    extra_dim = size(extra_input, 2)

    # Todas las señales tienen la misma condición inicial U0 = 1, vamos a usar una ANODE por lo que necesitamos además las condiciónes iniciales a(0) = 0
    U0 = [ones(size(signalsTrain)[1]) zeros(size(signalsTrain)[1])]
    U0_valid = [ones(size(signalsVal)[1]) zeros(size(signalsVal)[1])]
    U0_test = [ones(size(signalsTest)[1]) zeros(size(signalsTest)[1])]

    train_loader = Flux.Data.DataLoader((signalsTrain, t), batchsize = batch_size)

    layers = architecture[1]
    activation = architecture[2]

    if activation == tanh_fast
        activation_string = "tanh_fast"
    elseif activation == relu
        activation_string = "relu"
    elseif activation == swish
        activation_string = "swish"
    end

    if opt == AdamW
        opt_string = "AdamW"
    elseif opt == RMSProp
        opt_string = "RMSProp"
    end

    # Vamos a crear el modelo de la red neuronal que va a estar dentro de la ODE
    nn = create_model(layers, activation, dropout_rate)

    # Epocas de entrenamiento
    epochs = 500

    # Entrenamos una NODE con mini-batchs para cada arquitectura, optimizador y tamaño de mini-batch y guardamos el loss y los parametros de la red neuronal    
    architecture_loss, theta, loss_validacion = Train_Neural_ODE(nn, U0, U0_valid, extra_input, extra_input_valid,
                                                                epochs, train_loader, opt, eta, signalsTrain, signalsVal, t, lambd, actual_id, extra_dim)

    # Guardamos los hiperparámetros del entrenamiento y el loss final
    actual_layer = string(layers)
    actual_activation = activation_string
    actual_optimizer = opt_string
    actual_batch_size = batch_size
    actual_lambda = lambd
    actual_eta = eta
    actual_loss_final_train = round(architecture_loss[end], digits = 4)
    actual_loss_final_valid = round(loss_validacion[end], digits = 4)

    # Guardamos los resultados en un archivo csv
    df_results_total = DataFrame(ID = actual_id, 
                                Arq = actual_layer, 
                                Activ = actual_activation,
                                Opt = actual_optimizer,
                                BS = actual_batch_size,
                                lambd = actual_lambda,
                                Eta = actual_eta,
                                MSETrain = actual_loss_final_train,
                                MSEPred = actual_loss_final_valid,
                                dp = dropout_rate,
                                Num_data = n_usados)

    CSV.write("/home/juan.morales/PANODE/Results/0$(actual_id)_$(actual_layer)_$(actual_activation)_$(actual_optimizer)_$(actual_batch_size)_$(actual_lambda)_$(actual_eta)_$(dropout_rate).csv", df_results_total)

    # Guardamos los loss y los parametros de la red neuronal en archivos csv
    Loss_Matrix = zeros((length(architecture_loss), 2))

    for i in 1:length(architecture_loss)
        Loss_Matrix[i,1] = architecture_loss[i]
        Loss_Matrix[i,2] = loss_validacion[i]
    end

    df_losses = DataFrame(Loss_Matrix, :auto)

    rename!(df_losses, Symbol("x1") => Symbol("Loss_Entrenamiento"))
    rename!(df_losses, Symbol("x2") => Symbol("Loss_Predicción"))

    # Chequeamos si existe previamente un archivo CSV y si existe concatenamos al actual
    if isfile("/home/juan.morales/PANODE/Losses/0$(actual_id)_losses.csv")
        df_losses = vcat(CSV.read("/home/juan.morales/PANODE/Losses/0$(actual_id)_losses.csv", DataFrame), df_losses)
    end
    CSV.write("/home/juan.morales/PANODE/Losses/0$(actual_id)_losses.csv", df_losses)

    df_theta = DataFrame(reshape(theta, length(theta), 1), :auto)
    CSV.write("/home/juan.morales/PANODE/Parameters/0$(actual_id)_Parameters.csv", df_theta)
end

main()