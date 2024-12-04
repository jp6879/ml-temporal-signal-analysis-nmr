"""
# Programa para la integración de ambos modelos mediante una función de loss conjunta

This script integrates two models using a combined loss function for training. It includes the following steps:

1. **Import Libraries**: Import necessary libraries including Flux, DataFrames, CSV, DifferentialEquations, and others.
2. **Set Random Seed**: Set a random seed for reproducibility.
3. **Import Parameters and Utilities**: Include external Julia files for experiment parameters and signal utilities.
4. **Read Signal Data**: Read signal data from a CSV file and transpose it.
5. **Extract Signal Parameters**: Extract parameters from the signals for identification.
6. **Shuffle Data**: Shuffle the data to avoid bias during training.
7. **Split Data**: Split the data into training, validation, and test sets.
8. **Prepare Training Data**: Prepare the training, validation, and test signals and parameters.
9. **Reduce Time Domain**: Reduce the time domain for faster training of the NODE.
10. **Define NODE Functions**: Define functions for the Neural Ordinary Differential Equation (NODE) model.
11. **Define Autoencoder Functions**: Define functions for the Autoencoder model.
12. **Define Neural Network Functions**: Define functions for the Neural Network model.
13. **Center Data**: Center the data columns to have a mean of 0.
14. **End-to-End Model**: Define the end-to-end model combining NODE, Autoencoder, and Neural Network.
15. **Loss Functions**: Define loss functions for signal regularization and combined loss.
16. **Training Loop**: Train the model using the combined loss function and save the best parameters.

The script also includes functions for:
- Solving the NODE.
- Predicting signals.
- Centering data.
- Calculating signal and derivative regularization.
- Calculating combined loss.

The training loop iterates over epochs, updating model parameters using the ADAM optimizer, and implements early stopping based on validation loss.
"""
using Flux
using Flux: train!
using DataFrames
using CSV
using DifferentialEquations
using Statistics
using Random
using LinearAlgebra
using SciMLSensitivity
using ComponentArrays, Optimization, OptimizationOptimJL, OptimizationFlux
using OrdinaryDiffEq
using Measures

rng = Random.seed!(1234)

# Importamos los parámetros del experimento
include("/home/juan.morales/EntrenamientoConjunto/Hahn/Parameters.jl")

# Importamos las utilidades para las señales
include("/home/juan.morales/EntrenamientoConjunto/Hahn/UtilidadesSeñales.jl")

# Path desde donde se leen los datos de las señales de Hahn
path_read = "/home/juan.morales/datos_PCA/SimpleSignalHahn_TE_1_G_8.73e-7.csv"

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
perm = randperm(rng, size(signalsDF, 2))

signalsDF_shuffled = signalsDF[:, perm]
pdistparamsDF_shuffled = pdistparamsDF[perm, :]

# Split en entrenamiento, validación y test
n_signals = size(signalsDF, 2)
n_train = Int(floor(n_signals*0.7))
n_val = Int(floor(n_signals*0.15))
n_test = n_signals - n_train - n_val

train_signals = Flux.Float32.(Matrix(signalsDF_shuffled[:, 1:n_train]))
val_signals = Flux.Float32.(Matrix(signalsDF_shuffled[:, n_train+1:n_train+n_val]))
test_signals = Flux.Float32.(Matrix(signalsDF_shuffled[:, n_train+n_val+1:end]))

# Output objetivo del modelo
train_params = Flux.Float32.(Matrix(pdistparamsDF_shuffled[1:n_train, :])[:,end:-1:1]) 
val_params = Flux.Float32.(Matrix(pdistparamsDF_shuffled[n_train+1:n_train+n_val, :])[:,end:-1:1])
test_params = Flux.Flux.Float32.(Matrix(pdistparamsDF_shuffled[n_train+n_val+1:end, :])[:,end:-1:1])

# Vamos a tomar un subconjunto de t para hacer el entrenamiento de la NODE para agilizar los tiempos de entrenamiento
muestreo_corto = 20 # Cada cuantos tiempos tomamos un timepo para entrenar la NODE
muestreo_largo = 10

# Esto da 60 tiempos 50 puntos desde 0 a 0.1 y 10 puntos desde 0.1 a 1
t_shortaux = t_short[1:muestreo_corto:end]
t_longaux = t_long[1:muestreo_largo:end]

t = vcat(t_shortaux, t_longaux)

indexes_t = [i for i in 1:length(times) if times[i] in t]

# Obtenemos las señales con el dominio de tiempo reducido (Input del modelo)
signalsTrain = transpose(train_signals[indexes_t, :])
signalsVal = transpose(val_signals[indexes_t, :])
signalsTest = transpose(test_signals[indexes_t, :])

############################# Funciones NODE ####################################

# Numero de puntos para la predicción
n_usados = 8

# Indice final de los tiempos
idx_end = length(t)

# Tomamos 8 de los tiempos que tenemos para la predicción como parámetros extra para la ANODE
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

# Obtenemos las interpolaciones de las señales y de las derivadas
extra_input = signalsTrainPuntos
extra_input_valid = signalsValPuntos
extra_input_test = signalsTestPuntos

# Todas las señales tienen la misma condición inicial U0 = 1, vamos a usar una ANODE por lo que necesitamos además las condiciónes iniciales a(0) = 0
U0 = [ones(size(signalsTrain)[1]) zeros(size(signalsTrain)[1])]
U0_valid = [ones(size(signalsVal)[1]) zeros(size(signalsVal)[1])]
U0_test = [ones(size(signalsTest)[1]) zeros(size(signalsTest)[1])]

# Función de activación
activation = relu

# 27 Arquitecura: Any[[12, 32, 32, 16, 2], relu]
extra_dim = size(extra_input, 2)

# Input t, S(t), a(t), extra_input = puntos medidos de la señal
input_size = 1 + 2 + extra_dim

nn = Flux.Chain(Flux.Dense(input_size, 32),
            Flux.Dense(32, 32, activation),
            Flux.Dense(32, 16, activation),
            Flux.Dense(16, 2),
            )
# Output S(t), a(t)

p_anode, re_anode = Flux.destructure(nn)

# Leemos los parámetros de la red ya entrenada si es que existen
if isfile("/home/juan.morales/EntrenamientoConjunto/Hahn/Models/PANODE/027_Parameters.csv")
    theta_nn = CSV.read("/home/juan.morales/EntrenamientoConjunto/Hahn/Models/PANODE/027_Parameters.csv", DataFrame)
    p_anode = @views Flux.Float32.(theta_nn[:,1])
else
    println("No se encontraron los parámetros de la red neuronal")
end

# Tiempo sobre el cual resolver la ODE
tspan = (0f0, 1f0)

function SolveANODE(u0, extra, time_batch)
    # Definimos la función que resuelve la ODE
    function dSdt!(u, p, t)
        S = u[1]
        a = u[2]
        entrada = [t, S, a, extra...]
        return re_anode(p)(entrada)
    end

    # Definimos el problema de la ODE
    prob = ODEProblem(dSdt!, u0, tspan)

    # Resolvemos la ODE, devolvemos únicamente la solución S(t)
    solution = @views Array(solve(prob, Tsit5(), p = p_anode, saveat = time_batch, dtmin = 1e-9))[1, :]

    return solution
end

# Función que predice las señales para un conjunto de condiciones iniciales y parámetros extra
function PredictSignals(U0, extra, time_batch)
    predicted_signals = map(1:length(U0[:,1])) do i
        u0 = Flux.Float32.(U0[i,:])
        if length(extra) > extra_dim
            SolveANODE(u0, extra[i, :], time_batch)
        else
            SolveANODE(u0, extra[1,:], time_batch)
        end
    end
    
    return @views transpose(reduce(hcat, predicted_signals))
end


############################# Funciones AE ####################################
n_times = size(train_signals, 1)

# Definimos el encoder
encoder = Flux.Chain(Flux.Dense(n_times, 500, bias = false), Flux.Dense(500, 100, relu), Flux.Dense(100, 50, relu), Flux.Dense(50, 3))
decoder = Flux.Chain(Flux.Dense(3, 50, bias = false), Flux.Dense(50, 100, relu), Flux.Dense(100, 500, relu), Flux.Dense(500, n_times))

# Cargamos los parámetros del encoder y decoder
p_encoder, re_encoder = Flux.destructure(encoder)
p_decoder, re_decoder = Flux.destructure(decoder)

# Definimos el autoencoder
if isfile("/home/juan.morales/EntrenamientoConjunto/Hahn/Models/Autoencoder/deeperNLParamsE.csv")
    theta_encoder = @views Flux.Float32.(CSV.read("/home/juan.morales/EntrenamientoConjunto/Hahn/Models/Autoencoder/deeperNLParamsE.csv", DataFrame)[:,1])
    p_encoder = theta_encoder
else
    println("No se encontraron los parámetros del encoder")
end

if isfile("/home/juan.morales/EntrenamientoConjunto/Hahn/Models/Autoencoder/deeperNLParamsD.csv")
    theta_decoder = @views Flux.Float32.(CSV.read("/home/juan.morales/EntrenamientoConjunto/Hahn/Models/Autoencoder/deeperNLParamsD.csv", DataFrame)[:,1])
    p_decoder = theta_decoder
else
    println("No se encontraron los parámetros del decoder")
end

# Aunque solo vamos a usar el encoder, vamos a chequear la función de loss del autoencoder

############################# Funciones NN ####################################

# Definimos la red neuronal
model_nn = Flux.Chain(
    Dense(3, 32),
    Dense(32, 64, tanh_fast),
    Dense(64, 32, tanh_fast),
    Dense(32, 16, tanh_fast),
    Dense(16, 2, softplus),
)

# Obtenemos los parámetros de la red neuronal
p_nn, re_nn = Flux.destructure(model_nn)

path_params_nn = "/home/juan.morales/EntrenamientoConjunto/Hahn/Models/NeuralNetwork/017_params.csv"
if isfile(path_params_nn)
    theta = CSV.read(path_params_nn, DataFrame)
    p_nn = @views Flux.Float32.(theta[:,1])
else
    println("No se encontraron los parámetros de la red neuronal")
end

# En cada columna tenmos los datos de las señales, centramos estas columnas para que tengan media 0
function centerData(matrix)
    """Función que centra los datos de las columnas de una matriz para que tengan media 0
    Parametros
        matrix: matriz con los datos a centrar
    Retorna
        centered_data: matriz con los datos centrados
    """
    col_mean = mean(matrix, dims=2)
    centered_data = matrix .- col_mean
    return Flux.Float32.(centered_data), Flux.Float32.(col_mean)
end

function model_end_to_end(U0, extra_parameters)
    # Obtenemos las señales predichas por la NODE
    signals_predicted, signals_mean = centerData(PredictSignals(U0, extra_parameters, times))
    # Obtenemos las componentes reducidas de las señales predichas
    components = re_encoder(p_encoder)(signals_predicted')
    # Obtenemos los parámetros predichos por la red neuronal
    parameters_predicted = re_nn(p_nn)(components)'
    return parameters_predicted, signals_predicted .+ signals_mean
end


# Chequeo de que los modelos fueron cargados correctamente
# ANODE
# function loss_anode(batch, time_batch)
#     y = PredictSignals(U0, extra_input, time_batch)
#     return Flux.mse(y, batch)
# end

# signalsTrain = transpose(train_signals[indexes_t, :])
# println("Loss anode con parametros cargados: ", loss_anode(signalsTrain, t))


# # Autoencoder
# train_signals_C = centerData(train_signals')[1]
# train_signals_reconstruct = re_decoder(p_decoder)(re_encoder(p_encoder)(train_signals_C'))
# println("Loss autoencoder parametros cargados: ", Flux.mse(train_signals_C, train_signals_reconstruct'))

# # NN
# train_params_predicted = re_nn(p_nn)(re_encoder(p_encoder)(train_signals_C'))
# train_params = Flux.Float32.(Matrix(pdistparamsDF[1:n_train, :])[:,end:-1:1])'

# println("Loss NN parametros cargados: ", Flux.mse(train_params, train_params_predicted))

# Signal regularization
function S_reg(signal)
    # \sum_j^M \sum_i^N |S_j(t_i)| / M
    return Flux.Float32(sum(abs, signal) / size(signal)[1])
end

# Derivate regularization
function S_prime_reg(signal, times)
    # \sum_j^M \sum_i^N |S_j(t_i) - S_j(t_{i-1} / dt| / M
    s_primes = diff(signal, dims = 2)' ./ diff(times)
    return Flux.Float32(sum(abs, s_primes) / size(s_primes)[1])
end

λ_s = 1e-3
λ_s_prime = 1e-3

# Loss end to end
function combined_loss(U0, extra_parameters, real_components)
    parameters_prediction, signals_prediction = model_end_to_end(U0, extra_parameters)
    return Flux.mse(real_components, parameters_prediction) + 1/size(signals_prediction, 1) * (λ_s * S_reg(signals_prediction) + λ_s_prime * S_prime_reg(signals_prediction, times))
end


# All parameters of the model and data
all_params = Flux.params([p_encoder, p_anode, p_nn])
data_loader = Flux.DataLoader((signalsTrainPuntos', train_params', U0'), batchsize = 100, shuffle = true)

# Training parameters
lr = 0.01

# Optimizer
global opt = ADAM(lr)

num_epochs = 100
patience_epochs = 30

losses = []
losses_val = []

for epoch in 1:num_epochs
    for (signals_puntos, true_parameters, U0_batch) in data_loader
        grads = Flux.gradient(() -> combined_loss(U0_batch', signals_puntos', true_parameters'), all_params)
        Flux.update!(opt, all_params, grads)
        GC.gc()
    end

    actual_loss = combined_loss(U0, signalsTrainPuntos, train_params)
    actual_val_loss = combined_loss(U0_valid, signalsValPuntos, val_params)

    if epoch > 1 && actual_val_loss < minimum(losses_val)
        df_encoder = DataFrame(reshape(all_params[1], length(all_params[1]), 1), :auto)
        df_anode = DataFrame(reshape(all_params[2], length(all_params[2]), 1), :auto)
        df_nn = DataFrame(reshape(all_params[3], length(all_params[3]), 1), :auto)
        
        CSV.write("/home/juan.morales/EntrenamientoConjunto/Hahn/Models/BestParams/TrainedParamsE_WR.csv", df_encoder)
        CSV.write("/home/juan.morales/EntrenamientoConjunto/Hahn/Models/BestParams/TrainedParamsANODE_WR.csv", df_anode)
        CSV.write("/home/juan.morales/EntrenamientoConjunto/Hahn/Models/BestParams/TrainedParamsNN_WR.csv", df_nn)
    end

    push!(losses, actual_loss)
    push!(losses_val, actual_val_loss)
    println("Epoch: $epoch | Loss: $actual_loss | Loss Val: $actual_val_loss")

    if epoch % patience_epochs == 0
        global opt = ADAM(lr * 0.1)
        loss_val_prev = losses_val[end-patience_epochs+1]
        if actual_val_loss > loss_val_prev
            println("Early stopping at epoch: $epoch, because the validation loss is not decreasing after $patience_epochs epochs")
            break
        end
    end
end

df_losses = DataFrame(MSEtrain = losses,
                      MSEvalid = losses_val)

CSV.write("/home/juan.morales/EntrenamientoConjunto/Hahn/Models/Losses/Losses_WR.csv", df_losses)