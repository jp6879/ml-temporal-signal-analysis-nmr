# Programa para visualizar los resultados de cierta arquitectura ya entrenada
# Importamos los paquetes necesarios
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

include("/home/juan.morales/PANODE/UtilidadesSeñales.jl")
include("/home/juan.morales/PANODE/Parameters.jl")

rng = Random.seed!(1234)  # Set seed for reproducibility

# Path desde donde se leen los datos
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
muestreo_corto = 20 # Cada cuantos tiempos tomamos un timepo para entrenar la NODE
muestreo_largo = 10

# Esto da 24 tiempos 50 puntos desde 0 a 0.1 y 10 puntos desde 0.1 a 1
t_shortaux = t_short[1:muestreo_corto:end]
t_longaux = t_long[1:muestreo_largo:end]

t = vcat(t_shortaux, t_longaux)

times = unique(times)

indexes_t = [i for i in 1:length(times) if times[i] in t]

# Obtenemos las señales con el dominio de tiempo reducido
signalsTrain = transpose(train_signals[indexes_t, :])
signalsVal = transpose(val_signals[indexes_t, :])
signalsTest = transpose(test_signals[indexes_t, :])

# Numero de puntos para la predicción
n_usados = 8

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

# Obtenemos las interpolaciones de las señales y de las derivadas
extra_input = signalsTrainPuntos
extra_input_valid = signalsValPuntos
extra_input_test = signalsTestPuntos

# Todas las señales tienen la misma condición inicial U0 = 1, vamos a usar una ANODE por lo que necesitamos además las condiciónes iniciales a(0) = 0
U0 = [ones(size(signalsTrain)[1]) zeros(size(signalsTrain)[1])]
U0_valid = [ones(size(signalsVal)[1]) zeros(size(signalsVal)[1])]
U0_test = [ones(size(signalsTest)[1]) zeros(size(signalsTest)[1])]

# id actual de la red
actual_id = 27

#Definimos el batch size
batch_size = 15

# Vamos a crear el dataloader para el entrenamiento de la NODE con mini-batchs
train_loader = Flux.Data.DataLoader((signalsTrain, t), batchsize = batch_size)

# Función de activación
activation = relu

# Definimos el rate de dropout en la primera capa
dropout_rate = 0

# 27 Arquitecura: Any[[12, 32, 32, 16, 2], NNlib.relu] Optimizador: AdamW Tamaño de mini-batch: 15 Lambda: 0.0 Eta: 0.01 Dropout: 0.0

extra_dim = size(extra_input, 2)

# Input t, S(t), a(t), extra_input = puntos medidos de la señal
input_size = 1 + 2 + extra_dim

nn = Flux.Chain(Flux.Dense(input_size, 32),
            Flux.Dense(32, 32, activation),
            Flux.Dense(32, 16, activation),
            Flux.Dense(16, 2),
            )

# Output S(t), a(t)

# Tomamos un learning rate de 0.001
η = 5e-3

# Vamos a tomar 1000 épocas para entrenar todas las arquitecturas
epochs = 2

p_nn, re = Flux.destructure(nn) # Para entrenar la red tenemos que extraer los parametros de la red neuronal en su condicion inicial

# Leemos los parámetros de la red ya entrenada si es que existen
if isfile("/home/juan.morales/PANODE/Parameters/0$(actual_id)_Parameters.csv")
    theta = CSV.read("/home/juan.morales/PANODE/Parameters/0$(actual_id)_Parameters.csv", DataFrame)
    p_nn = Float32.(theta[:,1])
else
    println("No se encontraron los parámetros de la red neuronal")
end

# Optimizardor
opt = AdamW(η)

# Tiempo sobre el cual resolver la ODE
tspan = (0f0, 1f0)

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

function loss_test(batch, time_batch)
    y = PredictSignals(U0_test, extra_input_test, time_batch)
    return Flux.mse(y, batch)
end

println("Iniciales")
println("Loss MSE NeuralODE en el set de entrenamiento: ", loss_node(signalsTrain, t))
println("Loss MSE NeuralODE en el set de validación: ", loss_valid(signalsVal, t))
println("Loss MSE NeuralODE en el set de test: ", loss_test(signalsTest, t))

println("Loss RMSE NeuralODE en el set de entrenamiento: ", sqrt(loss_node(signalsTrain, t)))
println("Loss RMSE NeuralODE en el set de validación: ", sqrt(loss_valid(signalsVal, t)))
println("Loss RMSE NeuralODE en el set de test: ", sqrt(loss_test(signalsTest, t)))

# Función de callback para guardar el loss en cada época
global iter = 0
loss = []
loss_valid_array = []
callback = function ()
    global iter += 1
    if iter % (length(train_loader)) == 0
        epoch = Int(iter / length(train_loader))
        actual_loss = loss_node(signalsTrain, t)
        forecast_loss = loss_valid(signalsVal, t)
        println("Epoch = $epoch || Loss: $actual_loss || Loss val: $forecast_loss")
        push!(loss, actual_loss)
        push!(loss_valid_array, forecast_loss)
    end
    GC.gc()
    return false
end

###############################################################################################################

epochs = 700

# Entrenamos la red neuronal
Flux.train!(loss_node, Flux.params(p_nn), ncycle(train_loader, epochs), opt, cb = callback)

# Guardamos los parámetros
df_parameters = DataFrame(reshape(p_nn, length(p_nn), 1), :auto)

CSV.write("/home/juan.morales/PANODE/Arq27/0$(actual_id)_Parameters.csv", df_parameters)

# Guardamos las funciónes de loss
Loss_Matrix = zeros((length(loss), 2))

for i in 1:length(loss)
    Loss_Matrix[i,1] = loss[i]
    Loss_Matrix[i,2] = loss_valid_array[i]
end

df_losses = DataFrame(Loss_Matrix, :auto)
rename!(df_losses, Symbol("x1") => Symbol("Loss_Entrenamiento"))
rename!(df_losses, Symbol("x2") => Symbol("Loss_Predicción"))

if isfile("/home/juan.morales/PANODE/Losses/0$(actual_id)_losses.csv")
   df_losses = CSV.read("/home/juan.morales/PANODE/Losses/0$(actual_id)_losses.csv", DataFrame)
   loss = df_losses[:,1]
   loss_valid_array = df_losses[:,2]
else
    println("No se encontraron loss previos de la red neuronal")
end

CSV.write("/home/juan.morales/PANODE/Arq27/0$(actual_id)_losses.csv", df_losses)