# Autoencoder para las señales de Hahn
using CSV
using DataFrames
using Flux
using Random
using Statistics
using IterTools: ncycle
using LinearAlgebra

include("/home/juan.morales/Autoencoder/Parameters.jl")

rng = Random.seed!(1234)

path_signals = "/home/juan.morales/datos_PCA/SimpleSignalHahn_TE_1_G_8.73e-7.csv"
signalsDF = transpose(Matrix(CSV.read(path_signals, DataFrame)))

column_lcm = collect(lcms)
column_sigma = collect(σs)

pdistparamsDF = zeros(size(signalsDF)[2], 2)

for (i, lcm) in enumerate(column_lcm)
    for (j, sigma) in enumerate(column_sigma)
        pdistparamsDF[(i - 1) * length(σs) + j, 1] = sigma
        pdistparamsDF[(i - 1) * length(σs) + j, 2] = lcm
    end
end

pdistparamsDF = DataFrame(pdistparamsDF, [:sigma, :lcm])

perm = shuffle(rng, 1:size(signalsDF, 2))

# En cada columna tenmos los datos de las señales, centramos estas columnas para que tengan media 0
function centerData(matrix)
    """Función que centra los datos de las columnas de una matriz para que tengan media 0
    Parametros
        matrix: matriz con los datos a centrar
    Retorna
        centered_data: matriz con los datos centrados
    """
    col_means = mean(matrix, dims=1)
    centered_data = matrix .- col_means
    return centered_data, col_means
end


signalsDF_C, col_means = centerData(signalsDF)
# Random Shuflle the data
# signalsDF = signalsDF[:, perm]
signalsDF_C = signalsDF_C[:, perm]
pdistparamsDF = pdistparamsDF[perm, :];

# Split the data in training, validation and test
n_signals = size(signalsDF_C, 2)
n_train = Int(floor(n_signals*0.7))
n_val = Int(floor(n_signals*0.15))
n_test = n_signals - n_train - n_val

train_signals = Float32.(signalsDF_C[:, 1:n_train])
val_signals = Float32.(signalsDF_C[:, n_train+1:n_train+n_val])
test_signals = Float32.(signalsDF_C[:, n_train+n_val+1:end])

train_params = pdistparamsDF[1:n_train, :]
val_params = pdistparamsDF[n_train+1:n_train+n_val, :]
test_params = pdistparamsDF[n_train+n_val+1:end, :];

# Input size is the number of time points
n_times = size(train_signals, 1)

# Función para crear los modelos de la red neuronal
function create_model(layers, activation, dropout_rate)
    encoder_start = Dense(layers[1], layers[2], bias = false)
    if length(layers) > 2
        encoder_hidden = hcat([[Dense(layers[i], layers[i+1], activation), Dropout(dropout_rate)] for i in 2:length(layers) - 2]...)
        encoder_end = Dense(layers[end-1], layers[end])
        encoder = Chain(encoder_start, encoder_hidden..., encoder_end)
    else
        encoder = Chain(encoder_start)
    end
    decoder_start = Dense(layers[end], layers[end-1], bias = false)
    if length(layers) > 2            
        decoder_hidden = hcat([[Dense(layers[i+1], layers[i], activation), Dropout(dropout_rate)] for i in length(layers)-2:-1:2]...)
        decoder_end = Dense(layers[2], layers[1])
        decoder = Chain(decoder_start, decoder_hidden..., decoder_end)
    else
        decoder = Chain(decoder_start)
    end

    autoencoder = Chain(encoder, decoder)

    return encoder, decoder, autoencoder
end

# Architectures 

names = ["simpleNL", "deepNL", "deeperNL", "simpleL", "deepL", "deeperL"]

architectures = [
        [[n_times, 3], relu],
        [[n_times, 100, 50, 3], relu],
        [[n_times, 500, 100, 50, 3], relu],
        [[n_times, 3], identity],
        [[n_times, 100, 50, 3], identity],
        [[n_times, 500, 100, 50, 3], identity]
        ]

dp_rates = Float32.([0.0])
lambda_orthogonality = Float32.([0.0])

# Creamos el vector de configuraciones

configurations = []
for (arch, name) in zip(architectures, names), dp in dp_rates, lo in lambda_orthogonality
    push!(configurations, (arch, dp, lo, name))
end

println("Number of configurations: ", length(configurations))

# Seleccionamos la configuración
actual_configuration = configurations[parse(Int128, ARGS[1])]
println("Actual configuration: ", actual_configuration)

layers = actual_configuration[1][1]
activation = actual_configuration[1][2]
dp_rate = actual_configuration[2]
lambda_orth = actual_configuration[3]
name = actual_configuration[4]

encoder, decoder, autoencoder = create_model(layers, activation, dp_rate)

loader = Flux.DataLoader((train_signals, train_signals), batchsize=100, shuffle=true)
lr = 1e-4
optim = Flux.setup(Flux.AdamW(lr), autoencoder)

function loss_reconstruction(x, y)
    return Flux.mse(a_re(a_params)(x), y)
end

# function weight_orthogonality(W)
#     product = W * W'
#     sz = size(product)
#     return sqrt(sum(abs2, product - Matrix{Float32}(I, sz)))
# end

num_epochs = 5000
patience_epochs = 250

losses = []
losses_val = []

for epoch in 1:num_epochs
    for (x, y) in loader
        loss, grads = Flux.withgradient(autoencoder) do m
            Flux.mse(m(x), y)
        end
        Flux.update!(optim, autoencoder, grads[1])
    end
    actual_loss = Flux.mse(autoencoder(train_signals), train_signals)
    actual_val_loss = Flux.mse(autoencoder(val_signals), val_signals)

    if epoch > 1 && actual_val_loss < minimum(losses_val)
        encoder_params, encoder_re = Flux.destructure(autoencoder[1])
        decoder_params, decoder_re = Flux.destructure(autoencoder[2])

        df_encoder = DataFrame(reshape(encoder_params, length(encoder_params), 1), :auto)
        df_decoder = DataFrame(reshape(decoder_params, length(decoder_params), 1), :auto)

        CSV.write("/home/juan.morales/Autoencoder/Models/BestParams/min$(name)ParamsE.csv", df_encoder)
        CSV.write("/home/juan.morales/Autoencoder/Models/BestParams/min$(name)ParamsD.csv", df_decoder)
    end

    push!(losses, actual_loss)
    push!(losses_val, actual_val_loss)
    println("Epoch: $epoch | Loss: $actual_loss | Loss Val: $actual_val_loss")

    if epoch % patience_epochs == 0
        Flux.adjust!(optim, lr * 0.1)
        loss_val_prev = losses_val[end-patience_epochs+1]
        if actual_val_loss > loss_val_prev
            println("Early stopping at epoch: $epoch, because the validation loss is not decreasing after $patience_epochs epochs")
            break
        end
    end

    GC.gc()
end

encoder_params, encoder_re = Flux.destructure(autoencoder[1])
decoder_params, decoder_re = Flux.destructure(autoencoder[2])

df_encoder = DataFrame(reshape(encoder_params, length(encoder_params), 1), :auto)
df_decoder = DataFrame(reshape(decoder_params, length(decoder_params), 1), :auto)

df_losses = DataFrame(MSEtrain = losses,
                      MSEvalid = losses_val)


# Guardamos los datos en CSV
CSV.write("/home/juan.morales/Autoencoder/Models/Losses/$(name)Losses.csv", df_losses)

# Guardamos los parámetros de las redes en CSV
CSV.write("/home/juan.morales/Autoencoder/Models/$(name)ParamsE.csv", df_encoder)
CSV.write("/home/juan.morales/Autoencoder/Models/$(name)ParamsD.csv", df_decoder)
println("Loss de validación mínimo: ", minimum(losses_val), " en la época: ", argmin(losses_val))

# reduced_signals_train = encoder_re(encoder_params)(train_signals)
# reduced_signals_val = encoder_re(encoder_params)(val_signals)
# reduced_signals_test = encoder_re(encoder_params)(test_signals)

# df_train = DataFrame(
#     component1 = reduced_signals_train[1, :],
#     component2 = reduced_signals_train[2, :],
#     component3 = reduced_signals_train[3, :],
#     sigma = train_params.sigma,
#     lcm = train_params.lcm
# )

# df_val = DataFrame(
#     component1 = reduced_signals_val[1, :],
#     component2 = reduced_signals_val[2, :],
#     component3 = reduced_signals_val[3, :],
#     sigma = val_params.sigma,
#     lcm = val_params.lcm
# )

# df_test = DataFrame(
#     component1 = reduced_signals_test[1, :],
#     component2 = reduced_signals_test[2, :],
#     component3 = reduced_signals_test[3, :],
#     sigma = test_params.sigma,
#     lcm = test_params.lcm
# )


# # Plotting with grouping by 'lcm' and coloring by 'sigma'
# scatter(df.component1, df.component2, group = df.lcm, 
#         title = "Train Data", xlabel = "Component 1", ylabel = "Component 2", label = false)