# Programa que explora los parámetros de una red neuronal feedfoward para la identificación de lcm y σ a partir de señales en PCA de Hahn
# Se varían la arquitectura de la red, la función de activación y el optimizador, se analizan las redes con la métrica RMAE tanto globalmente como punto a punto de los datos PCA
# Autor: Juan Pablo Morales

# Paquetes necesarios
using Flux
using Statistics
using Flux: train!
using Plots
using Distributions
using DataFrames
using CSV
using LaTeXStrings
using LinearAlgebra
using CUDA
using Random
# Parámetros con los que funciona esta red neuronal
include("../1-GeneracionDatos/Parameters.jl")

# Función para generar directorio de trabajo
function create_dir()
    """Función que crea un directorio de trabajo"""
    if !isdir("./G_$(G)_TE_$(te)_AE")
        mkdir("./G_$(G)_TE_$(te)_AE")
        mkdir("./G_$(G)_TE_$(te)_AE/Losses")
        mkdir("./G_$(G)_TE_$(te)_AE/Params")
        mkdir("./G_$(G)_TE_$(te)_AE/Predictions")
        mkdir("./G_$(G)_TE_$(te)_AE/Results")
    else
        println("El directorio ya existe")
    end
end

create_dir()

# Distribucion de probabilidad log-normal se puede utilizar para añadir a la función de costo final, toma demasiado tiempo.
function Pln(lcm::Float32, σ::Float32)
    """Función que genera la distribución log-normal de tamaños de compartimientos lc con media lcm y desviación estándar σ para N valores de lc
    Parámetros:
        lcm (float): tamaño medio de compartimiento
        σ (float): desviación estándar de compartimiento

    Retorna:
        P (Array{float}): distribución log-normal
    """
    return [(exp(-(log(lc) - log(lcm))^2 / (2σ^2))) / (lc * σ * sqrt(2π)) for lc in lcs]
end

# Metricas de validacion de la red neuronal, error medio absoluto el cual se divide por la media de los datos reales para obtener el error relativo
# Relative Mean Squared error
function RMSE(predicted, real)
    return sqrt(Flux.mse(predicted, real))
end

# Regularizaciones L1 y L2 para la red neuronal
pen_l2(x::AbstractArray) = Float32.(sum(abs2, x) / 2)
pen_l1(x::AbstractArray) = Float32.(sum(abs, x) / 2)

# Función para crear los modelos de la red neuronal
function create_model(layers::Vector{Int}, activation::Function, dropout_rate::Float32)
    """Función que crea un modelo de red neuronal feedforward
    Se crean capas densas con el número de neuronas en cada capa y la función de activación en todas las capas de la red. Termina con una sofplus para asegurarnos positivad en las salidas.
    Parámetros:
        layers (Vector{Int}): vector con el número de neuronas en cada capa
        activation (Function): función de activación de las capas ocultas
        dropout_rate (Float): tasa de dropout
    Retorna:
        model (Chain): modelo de red neuronal
    """
    activations = [activation for i in 1:length(layers) - 2]
    startlayer = Dense(layers[1], layers[2])
    hiddenlayers = hcat([[Dense(layers[i], layers[i+1], activations[i]), Dropout(dropout_rate)] for i in 2:length(layers) - 2]...)
    endlayer = Dense(layers[end-1], layers[end], softplus)
    return Chain(startlayer, hiddenlayers..., endlayer)
end

# Función para cargar los datos de entrenamiento, validacion
function load_data(x_train, y_train, x_valid, y_valid, batchsize::Int, shuffle::Bool)
    data = Flux.Data.DataLoader((x_train, y_train), batchsize = batchsize, shuffle = shuffle)
    data_valid = Flux.Data.DataLoader((x_valid, y_valid), batchsize = batchsize, shuffle = shuffle)
    return data, data_valid
end

# Función para entrenar la red neuronal
function train_model(model, id::String, epochs::Int, learning_rate::Float32, opt, data, data_valid, L1::Bool, L2::Bool, λ)
    """Función para el entrenamiento de una de las redes neuronales creadas
    Parametros:
        model (Chain): modelo de red neuronal
        id (String): identificador de la arquitectura
        epochs (Int): número de épocas
        learning_rate (Float): tasa de aprendizaje
        opt (Function): optimizador
        data (DataLoader): datos de entrenamiento
        data_valid (DataLoader): datos de validación
        L1 (Bool): regularización L1
        L2 (Bool): regularización L2
        λ (Float): factor de regularización
    Retorna:
        rmae_global_train (Float): RMAE global en los datos de entrenamiento
        rmae_global_valid (Float): RMAE global en los datos de validación
    """
    η = learning_rate

    if opt == ADAM
        opt = ADAM(η)
    elseif opt == Descent
        opt = Descent(η)
    elseif opt == RMSProp
        opt = RMSProp(η)
    end

    # Función de costo de la red neuronal

    function loss(x,y)
        if L1
            return Flux.mse(model(x), y) + λ * pen_l1(params(model))
        elseif L2
            return Flux.mse(model(x), y) + λ * pen_l2(params(model))
        else
            return Flux.mse(model(x), y)
        end        
    end

    # Parámetros de la red neuronal
    params = Flux.params(model)

    # Guardamos los valores de la función de costo en cada época
    losses = []
    losses_valid = []
    
    # Definimos una funcion de callback para ver el progreso del entrenamiento cada 1000 épocas
    global iter = 0
    cb = function()
        global iter += 1
        if iter % length(data) == 0
            epoch = iter ÷ length(data)
            actual_loss = loss(data.data[1], data.data[2])
            actual_valid_loss = loss(data_valid.data[1], data_valid.data[2])
            if epoch % 1000 == 0
                println("Epoch $epoch || Loss = $actual_loss || Valid Loss = $actual_valid_loss")
            end
            push!(losses, actual_loss)
            push!(losses_valid, actual_valid_loss)
        end
    end;

    # Entrenamos la red neuronal haciendo ReduceLROnPlateau cada 500 épocas
    for epoch in 1:epochs
        train!(loss, params, data, opt, cb = cb)
        if epoch % 500 == 0
            η = η * 0.2
            if opt == ADAM
                opt = ADAM(η)
            elseif opt == Descent
                opt = Descent(η)
            elseif opt == RMSProp
                opt = RMSProp(η)
            end
        end
    end

    println("Entrenamiento finalizado")

    # Guardamos la función costo en cada época
    save_loss(losses, "0$(id)_loss_train.csv")
    save_loss(losses_valid, "0$(id)_loss_valid.csv")

    # Evaluamos el modelo en los datos de entrenamiento y validación
    rmae_global_train = eval_model(model, data.data[1], data.data[2])
    rmae_global_valid = eval_model(model, data_valid.data[1], data_valid.data[2])

    rmae_scores_train = eval_model_point(model, data.data[1], data.data[2])
    rmae_scores_valid = eval_model_point(model, data_valid.data[1], data_valid.data[2])

    # Guardamos los parámetros de la red neuronal
    save_params(params, "0$(id)_params.csv")

    # Guardamos las predicciones del modelo y el RMAE en cada punto
    save_predictions(model(data.data[1]), rmae_scores_train, "0$(id)_predictions_train.csv")
    save_predictions(model(data_valid.data[1]), rmae_scores_valid, "0$(id)_predictions_valid.csv")

    # Devolvemos los RMAE globales para globales
    return rmae_global_train, rmae_global_valid

end

##########################################################################################
# Tenemos dos formas de evaluar el modelo, una es el RMAE global y la otra es el RMAE punto a punto

# Función que evalua el RMAE global
function eval_model(model, x, y)
    y_pred = model(x)
    rmse = RMSE(y_pred, y)
    return rmse
end

# Función que evalua el RMAE punto a punto
function eval_model_point(model, x, y)
    y_pred = model(x)
    N = length(y_pred[1,:])

    rmse_scores = zeros(N)

    for i in 1:N
        rmse_scores[i] = RMSE(y_pred[:,i], y[:,i])
    end
    
    return rmse_scores
end

##########################################################################################

# Funciones de guardado de datos
function save_loss(loss_vector, filename::String)
    CSV.write("./G_$(G)_TE_$(te)_AE/Losses/"*filename, DataFrame(loss = loss_vector))
end

# Función para guardar las predicciones del modelo y el RMAE en cada punto
function save_predictions(predictions, rmse_scores, filename::String)
    df = DataFrame(x1 = predictions[1,:], x2 = predictions[2,:], rmae_scores = rmse_scores)
    CSV.write("./G_$(G)_TE_$(te)_AE/Predictions/"*filename, df)
end

# Función para guardar los parámetros de la red neuronal
function save_params(params, filename::String)
    """Función para guardar los parámetros de la red neuronal en un archivo CSV con los pesos como array numérico
    Parámetros:
        params (Flux.params) : Parámetros de la red neuronal
        filename (String) : Nombre del archivo csv donde se guardan los parámetros

    Retorna:
        nothing
    """
    params = collect(params)
    theta = []

    for layers in params
        for neuron in layers
            push!(theta, Float32(neuron))
        end
    end
    df_theta = DataFrame(reshape(theta, length(theta), 1), :auto)
    CSV.write("./G_$(G)_TE_$(te)_AE/Params/"*filename, df_theta)
end

# Función que lee los datos de PCA
function read_data_PCA(path_read::String, step_valid::Int64)
    """Función que lee los datos de PCA de las señales de Hahn
    separa además los datos en entrenamiento y validación, para esto tomamos un paso entre los datos según un paso de validación
    que es calculado según el porcentaje de datos para validación

    Parámetros:
        path_read (String): ruta donde se encuentran los datos
        step_valid (Int): paso para los datos de validación
    Retorna:
        datasignals (Array{Float32}): señales de PCA
        dataparams (Array{Float32}): parámetros de las señales de PCA
        datasignals_valid (Array{Float32}): señales de PCA de validación
        dataparams_valid (Array{Float32}): parámetros de las señales de PCA de validación
    """

    df_datasignals = CSV.read("C:/Users/Propietario/OneDrive/Escritorio/ib/Tesis_V1/MLonNMR/2-2-Autoencoders/ReducedDataNLData.csv", DataFrame)
    num_datos = size(df_datasignals)[1]
    k = 7 # Comienzo de los datos de validación
    datasignals_valid = Float32.(Matrix(df_datasignals[k^2:step_valid:num_datos,1:3])')
    datasignals = Float32.(Matrix(df_datasignals[setdiff(1:num_datos, k^2:step_valid:num_datos),1:3])')

    σ_valid = df_datasignals[k^2:step_valid:num_datos,4]
    lcm_valid = df_datasignals[k^2:step_valid:num_datos,5]
    
    σ_col = df_datasignals[setdiff(1:num_datos, k^2:step_valid:num_datos),4]
    lcm_col = df_datasignals[setdiff(1:num_datos, k^2:step_valid:num_datos),5]
    
    dataparams = hcat(lcm_col, σ_col)'
    dataparams_valid = hcat(lcm_valid, σ_valid)'
    
    return datasignals, dataparams, datasignals_valid, dataparams_valid
end

##########################################################################################

# Función principal que realiza la exploración de los parámetros de la red neuronal

function main()
    # Arquitecturas que vamos a utilizar
    # Utilizo solo las que andaban mejor como muestra
    architectures = [
        [[3, 8, 2], relu], # Una capa oculta con pocas neuronas
        [[3, 16, 2], relu], # Una capa oculta con más neuronas
        [[3, 16, 8, 2], relu], # Dos capas ocultas
        [[3, 16, 16, 2], relu], # Dos capas ocultas con aún más neuronas
        [[3, 8, 16, 8, 2], relu], # Tres capas ocultas
        [[3, 16, 32, 16, 2], relu], # Tres capas ocultas con más neuronas
        [[3, 32, 64, 16, 2], relu], # Tres capas ocultas con aun más neuonras
        [[3, 16, 32, 16, 8, 2], relu], # Cuatro capas ocultas
        [[3, 32, 64, 8, 8, 2], relu], # Cuatro capas ocultas mas neuronas
        [[3, 32, 64, 32, 16, 2], relu], # Cuatro capas ocultas con aun mas neuronas
        [[3, 30, 25, 20, 15, 10, 2], relu], # Cinco capas ocultas, mayor complejidad
        [[3, 16, 8, 2], tanh_fast], # Variando función de activación a tanh
        [[3, 16, 32, 16, 2], tanh_fast], # Tres capas ocultas con más neuronas
        [[3, 32, 64, 16, 2], tanh_fast], # Tres capas ocultas con aun más neuonras
        [[3, 32, 64, 8, 8, 2], tanh_fast], # Cuatro capas ocultas mas neuronas
        [[3, 32, 64, 32, 16, 2], tanh_fast], # Cuatro capas ocultas con aun mas neuronas
        [[3, 30, 25, 20, 15, 10, 2], σ], # Cinco capas ocultas σ
        [[3, 32, 64, 8, 8, 2], swish], # Cuatro capas ocultas mas neuronas
        [[3, 32, 64, 32, 16, 2], swish], # Cuatro capas ocultas con aun mas neuronas
        [[3, 30, 25, 20, 15, 10, 2], swish] # Cinco capas ocultas σ
        ]

    # Metodos de optimización que vamos a utilizar
    optimizers = [opt for opt in [ADAM]]

    # Lectura de los datos de PCA de las señales de Hahn
    path_read = "C:Users/Propietario/OneDrive/Escritorio/ib/Tesis_V1/MLonNMR/2-2-Autoencoders/"
    
    # Fraccion de datos que se van a utilizar para validación
    percent_valid = 0.1
    step_valid = Int(1 / percent_valid)

    train_signals, train_params, valid_signals, valid_params = read_data_PCA(path_read, step_valid)

    # Cargamos los datos de entrenamiento y validación
    batchsize = 100
    shuffle = true
    data, data_valid = load_data(train_signals, train_params, valid_signals, valid_params, batchsize, shuffle)

    ########### Si se van a hacer mas exploraciones cambiar esto por el id de la ultima arquitectura usada.#################
    id = 1 # Ultima arquitectura usada
   
    id_column = []
    layers_column = []
    activation_column = []
    optimizer_column = []
    dropout_column = []
    regularization_column = []
    λ_column = []
    rmae_global_train_column = []
    rmae_global_valid_column = []

    for architecture in architectures
        for opt in optimizers
            id += 1
            string_id = string(id)
            layers = architecture[1]
            activation = architecture[2]
            
            if activation == σ
                activation_string = "σ"
            elseif activation == tanh_fast
                activation_string = "tanh"
            elseif activation == relu
                activation_string = "relu"
            elseif activation == swish
                activation_string = "swish"
            end

            if opt == ADAM
                opt_string = "ADAM"
            elseif opt == Descent
                opt_string = "Descent"
            elseif opt == RMSProp
                opt_string = "RMSProp"
            end

            # Definimos la tasa de dropout
            dropout_rate = Float32(0.0)

            # Creamos el modelo
            model = create_model(layers, activation, dropout_rate)

            # Definimos el learning rate inicial, ya que este va a ser variable cada 500 épocas
            learning_rate = Float32(1e-4)

            # Definimos si se va a utilizar regularización L1 o L2
            L1 = false
            L2 = false

            # Definimos el factor de regularización
            λ = Float32(1e-4)

            # Definimos el número de épocas
            epochs = 3000

            # Entrenamos el modelo
            rmae_global_train, rmae_global_valid = train_model(model, string_id, epochs, learning_rate, opt, data, data_valid, L1, L2, λ)
    
            # Guardamos los datos de la arquitectura

            push!(id_column, id)
            push!(layers_column, layers)
            push!(activation_column, activation_string)
            push!(optimizer_column, opt_string)
            push!(dropout_column, dropout_rate)
            push!(rmae_global_train_column, rmae_global_train)
            push!(rmae_global_valid_column, rmae_global_valid)
            
            if L1
                push!(regularization_column, "L1")
                push!(λ_column, λ)
            elseif L2
                push!(regularization_column, "L2")
                push!(λ_column, λ)
            else
                push!(regularization_column, "None")
                push!(λ_column, 0)
            end
        end
    end

    df = DataFrame( id = id_column,
                    layers = layers_column,
                    activation = activation_column,
                    optimizer = optimizer_column,
                    dropout = dropout_column,
                    regularization = regularization_column,
                    λ = λ_column,
                    rmae_global_train = rmae_global_train_column,
                    mae_global_valid = rmae_global_valid_column)

    existing_csv_file = "./G_$(G)_TE_$(te)_AE/Results/Registro.csv"

    if isfile(existing_csv_file)
        df_old = CSV.read(existing_csv_file, DataFrame)
        df = vcat(df_old, df)
    end

    CSV.write("./G_$(G)_TE_$(te)_AE/Results/Registro.csv", df)

end

# Llamamos a la función principal
main()

# Fin del programa