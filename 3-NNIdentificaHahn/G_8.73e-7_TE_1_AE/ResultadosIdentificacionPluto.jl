### A Pluto.jl notebook ###
# v0.19.42

using Markdown
using InteractiveUtils

# ╔═╡ 019f3fd1-fc27-4d40-a672-a8e2bd90dfa2
using PlutoUI

# ╔═╡ af30ef40-7c37-11ef-10ba-ef6a12ad1b39
md"# Entrenamiento de una red neuronal para la identificación de los parámetros de la señal que caracterizan la señal $l_{cm}$ y $\sigma$ a partir de la representación del autoencoder

* Ahora que cambió la representación en baja dimensionalidad porque cambiamos la forma en que se hacía tenemos que reentrenar una red neuronal para poder identificar los parámetros de las distribuciones de tamaños de donde provienen las señales. Para esto se hizo una exploración de hiperparámetros teniendo en cuenta distintas arquitecturas y funciones de activación. Al final se llegó a una red bastante similar a la anterior con respecto a la arquitectura pero esta vez con la función de activación tahn.
"

# ╔═╡ 0f1a37ce-0f2d-4671-9d0a-b3a56d129a4b
# Importamos las librerias necesarias
using Flux
using Statistics
using Flux: train!
using Plots
using DataFrames
using CSV
using StatsPlots
using LaTeXStrings
using LinearAlgebra
using PlotlyJS
using CUDA
using Random
using Measures
# Importamos los parámetros del experimento
include("../1-GeneracionDatos/Parameters.jl");

# ╔═╡ 31f093df-2571-4321-89be-0eaa744279ab
# Distribucion de probabilidad log-normal que se puede agregar a la función costo de la red neuronal, lleva mucho tiempo de entrenamiento

function Pln(lcm, σ)
    return [(exp(-(log(lc) - log(lcm))^2 / (2σ^2))) / (lc * σ * sqrt(2π)) for lc in lcs]
end

####################################################################################################################
# Funciones de pre procesamiento para escalar los datos y estandarizarlos

# Normalización Max-Min
function MaxMin(data)
    min_vals = minimum(data, dims=1)
    max_vals = maximum(data, dims=1)
    scaled_data = (data .- min_vals) ./ (max_vals .- min_vals)
    return scaled_data

end

# Estandarización Z
function Standarize(data)
    mean_vals = mean(data, dims=1)
    std_devs = std(data, dims=1)
    standardized_data = (data .- mean_vals) ./ std_devs
    return standardized_data
end

####################################################################################################################
# Metricas de validacion de la red neuronal, solo utilice RMAE

# Root Mean Squared Error
function RMSE(predicted, real)
    return sqrt(Flux.mse(predicted, real))
end

# Mean Absolute Error
function MAE(predicted, real)
    return sum(abs.(predicted .- real)) / length(predicted)
end

# R2 score
function R2_score(predicted, real)
    return 1 - sum((predicted .- real).^2) / sum((real .- mean(real)).^2)
end

# Realetive Root Mean Squared Error
function RRMSE(predicted, real)
    return sqrt(mean((predicted .- real).^2)) / mean(real)
end

# Relative Mean Absolute Error
function RMAE(predicted, real)
    return mean(abs.(predicted .- real)) / mean(real)
end

# Mean Absolute Percentaje Error
function MAPE(predicted, real)
    return mean(abs.((predicted .- real) ./ real))
end

####################################################################################################################

# Regularizaciones L1 y L2
pen_l2(x::AbstractArray) = Float32.(sum(abs2, x) / 2)
pen_l1(x::AbstractArray) = Float32.(sum(abs, x) / 2)

####################################################################################################################

# Funciones de pre procesamiento para escalar los datos y estandarizarlos

# Normalización Max-Min
function MaxMin(data)
    min_vals = minimum(data, dims=1)
    max_vals = maximum(data, dims=1)
    scaled_data = (data .- min_vals) ./ (max_vals .- min_vals)
    return scaled_data

end

# Estandarización Z
function Standarize(data)
    mean_vals = mean(data, dims=1)
    print(mean_vals)
    std_devs = std(data, dims=1)
    print(std_devs)
    standardized_data = (data .- mean_vals) ./ std_devs
    return standardized_data
end;

# ╔═╡ ac7b08a7-41ec-476f-8ca6-b2f7825989b3
# Leemos los datos de las señales en PCA
path_read = "../1-GeneracionDatos/Data/"
df_datasignals = CSV.read("C:/Users/Propietario/OneDrive/Escritorio/ib/Tesis_V1/MLonNMR/2-2-Autoencoders/Models/ReducedDataNLData.csv", DataFrame)

# ╔═╡ 2ba1c854-9d0a-4d7f-8442-29ffc5a888d7
md"## K-folds

Usamos la técnica de validación cruzada K-folds para prevenir el sobreajuste de la red neuronal. Este método divide el conjunto de datos en K subconjuntos, y el modelo se entrena K veces, utilizando en cada iteración un subconjunto diferente como conjunto de validación y el resto como conjunto de entrenamiento. Al final, los resultados obtenidos en cada iteración se promedian, esto nos permite obtener una mejor estimación de cómo funcionará el modelo con datos no vistos previamente.

El proceso de separación de los datos es el siguiente:

Los datos se dividen en K subconjuntos de igual tamaño.
Cada subconjunto requiere de una representación adecuada de los datos, eligiendo señales de validación basadas en diferentes valores de $\sigma$ y $l_{cm}$ de la siguiente manera:
Se comienza con $l_{cm} = 0.5$, pero en cada fold se utiliza un valor distinto de $\sigma$. Posteriormente, se seleccionan valores equiespaciados de $l_{cm}$ y $\sigma$ de modo que el 20% de los datos se destinen a la validación. Así, al comenzar cada fold con un valor diferente de $\sigma$, se obtienen K conjuntos disjuntos para validar.
Después de entrenar el modelo K veces, cambiando el conjunto de validación en cada iteración, se calcula el rendimiento promedio de la red. Para esto, se obtienen la media y la desviación estándar de los errores de validación RMSE de los K folds.
"

# ╔═╡ 0617918a-c3b2-428f-b6a7-41e37f9801e7
# Utilizamos la técnica k-fold de validación cruzada para prevenir el overfitting de la red neuronal
# Definimos el número de folds
folds = 5
percent_valid = 0.2
step_valid = Int(1 / percent_valid)
num_datos = Int(size(df_datasignals, 1))

# Guardamos los datos de validacion de cada NN en cada fold
out_of_sample_data = []
out_of_sample_pred = []

# Guardamos la metrica de validación de cada NN en cada fold
scores_RMSE = []

# Definimos desde donde empezamos los datos de testing
idx_startTest = 6

# Primero sacamos los datos de testing de los datos de señales, estos seran un 3er conjunto de datos que no se usara para entrenar ni validar la red
df_datasignals_out = df_datasignals[idx_startTest:step_valid:num_datos,:]
# Datos de Testing
df_datasignals_minus_out = df_datasignals[setdiff(1:num_datos, idx_startTest:step_valid:num_datos),:]
# Nuevo numero de datos que tenemos para entrenamiento + validacion
num_datos_new = Int(size(df_datasignals_minus_out, 1))

# ╔═╡ bdd66e7d-7508-4c6f-97c3-7a659216936c
# Método de K-Folds

for k in 1:folds
    # Usamos 5 conjuntos disjuntos de datos de validación del 5% total de los datos para cada fold

    datasignals_valid = Float32.(Matrix(df_datasignals_minus_out[k^2 + 10:step_valid:num_datos_new,1:3])')
    datasignals = Float32.(Matrix(df_datasignals_minus_out[setdiff(1:num_datos_new, k^2 + 10:step_valid:num_datos_new),1:3])')

    σ_valid = df_datasignals_minus_out[k^2 + 10:step_valid:num_datos_new,4]
    lcm_valid = df_datasignals_minus_out[k^2 + 10:step_valid:num_datos_new,5]
    
    σ_col = df_datasignals_minus_out[setdiff(1:num_datos_new, k^2 + 10:step_valid:num_datos_new),4]
    lcm_col = df_datasignals_minus_out[setdiff(1:num_datos_new, k^2 + 10:step_valid:num_datos_new),5]
    
    dataparams = hcat(lcm_col, σ_col)'
    dataparams_valid = hcat(lcm_valid, σ_valid)'

    # Uno de los modelos que dio buenos resultados en la exploración anterior
    # 17,"[3, 32, 64, 32, 16, 2]",tanh,ADAM,0.0,None,0,0.017810457567638147,0.01427131790632415

    id = 17

    # Definimos la red neuronal
    model = Chain(
        Dense(3, 32),
        Dense(32, 64, tanh_fast),
        Dense(64, 32, tanh_fast),
        Dense(32, 16, tanh_fast),
        Dense(16, 2, softplus),
    )

    # Cargamos los parámetros del modelo
    path = "C:/Users/Propietario/OneDrive/Escritorio/ib/Tesis_V1/MLonNMR/2-NNIdentificaHahn/G_8.73e-7_TE_1_AE/Params/017_params.csv"

    p, re = Flux.destructure(model)
    
    theta = CSV.read(path, DataFrame)
    p = @views Float32.(theta[:,1])

    # Función de loss
    function loss(x,y)
        return Flux.mse(re(p)(x), y)
    end

    # Definimos el metodo de aprendizaje y la tasa de aprendizaje
    η = 1e-4
    opt = ADAM(η)

    # Definimos el número de épocas
    epochs = 5000

    # Definimos el tamaño del batch
    batch_size = 100

    # Usamos dataloader para cargar los datos
    data = Flux.DataLoader((datasignals, dataparams), batchsize = batch_size, shuffle = true)
    data_valid = Flux.DataLoader((datasignals_valid, dataparams_valid), batchsize = batch_size, shuffle = true)

    # Definimos una funcion de callback para ver el progreso del entrenamiento
    global iter = 0
    cb = function()
        global iter += 1
        if iter % length(data) == 0
            epoch = iter ÷ length(data)
            if epoch % 500 == 0
                actual_loss = loss(data.data[1], data.data[2])
                actual_valid_loss = loss(data_valid.data[1], data_valid.data[2])
                println("Epoch $epoch || Loss = $actual_loss || Valid Loss = $actual_valid_loss")
            end
        end
    end;

    # Entrenamos la red neuronal con el loss mse variando la tasa de aprendizaje cada 500 épocas
    for epoch in 1:epochs
        Flux.train!(loss, Flux.params(p), data, opt, cb=cb)
        if epoch % 500 == 0
            η = η * 0.2
            opt = ADAM(η)
        end
    end

    # Predicción de la red en la validacion
    predictions_valid = re(p)(datasignals_valid)

    # Métricas de validación de la red
    RMSE_valid = RMSE(predictions_valid, dataparams_valid)

    push!(scores_RMSE, RMSE_valid)

    # Guardamos los datos de validación y las predicciones de la red
    push!(out_of_sample_data, dataparams_valid)
    push!(out_of_sample_pred, predictions_valid)

    println("Fold $k terminado con score de validación RMSE = $RMSE_valid")

end

# ╔═╡ d93a98b1-6896-47cd-8391-0017ba9a55c1
md"### El promedio de la metrica RMSE en los conjuntos 5 folds de validación es 0.014 y el desvio estandar es 0.0011"

# ╔═╡ 6919310f-96a1-4437-ba22-a36ece690fe5
md"* Al haber testeado en todos estos datos y seguir obteniendo valores similares podemos asegurar que el modelo no está sobreajustando

* Podemos reentrenar el modelo con todos los datos y obtener un modelo más robusto"

# ╔═╡ 1f697ef0-2201-4a5f-9a01-e1419b7b4d26
# Convertimos las señales de test y train a una matriz de Float32
datasignals_test = Float32.(Matrix(df_datasignals_out[:,1:3])')
datasignals = Float32.(Matrix(df_datasignals_minus_out[:,1:3])')

# Extraemos las columnas objetivo de test y train
σ_test = df_datasignals_out[:,4]
lcm_test = df_datasignals_out[:,5]

σ_col = df_datasignals_minus_out[:,4]
lcm_col = df_datasignals_minus_out[:,5]

# Las concatenamos en dataparams
dataparams = hcat(lcm_col, σ_col)'
dataparams_test = hcat(lcm_test, σ_test)';

# ╔═╡ 484aa171-b846-4a27-825d-4967213c8ae1
# Función de loss
function loss(x,y)
    y_hat = re(p)(x)
    return Flux.mse(y_hat, y)
end

# Loss compuesto para hacer un fine tune comparando la predicción de la red con la distribución log-normal
function composed_loss(x,y)
    y_hat = model(x)
    Pln_predicted = Pln.(y_hat[1,:], y_hat[2,:])
    Pln_real = Pln.(y[1,:], y[2,:])
    return mean(Flux.mse.(Pln_predicted,Pln_real)) + Flux.mse(y_hat, y)
end

# ╔═╡ 041cd100-9e5a-48ac-b377-476fcd30a17e
# Definimos la red neuronal
model = Chain(
    Dense(3, 32),
    Dense(32, 64, tanh_fast),
    Dense(64, 32, tanh_fast),
    Dense(32, 16, tanh_fast),
    Dense(16, 2, softplus),
)

# Cargamos los parámetros del modelo
path = "C:/Users/Propietario/OneDrive/Escritorio/ib/Tesis_V1/MLonNMR/2-NNIdentificaHahn/G_8.73e-7_TE_1_AE/Params/017_params.csv"

p, re = Flux.destructure(model)

# Definimos el tamaño de los mini batchs para entrenar
batch_size = 100

# Usamos dataloader para cargar los datos
data = Flux.DataLoader((datasignals, dataparams), batchsize = batch_size, shuffle = true)

# Parámetros de la red neuronal
params = Flux.params(model)

# Definimos el vector donde guardamos la pérdida
losses = []

# Definimos una funcion de callback para ver el progreso del entrenamiento
iter = 0
cb = function()
    global iter += 1
    if iter % length(data) == 0
        epoch = iter ÷ length(data)
        actual_loss = loss(data.data[1], data.data[2])
        if epoch%100 == 0
            println("Epoch $epoch || Loss = $actual_loss")
        end
        push!(losses, actual_loss)
    end
end;

losses_composed = []
cb2 = function()
    global iter += 1
    epoch = iter ÷ length(data)
    actual_loss = composed_loss(data.data[1], data.data[2])
    println("Epoch $epoch || Loss = $actual_loss")
    push!(losses_composed, actual_loss)
end;

# Definimos el modo de aprendizaje y la tasa de aprendizaje
η = 1e-4
opt = ADAM(η)

# Definimos el número de épocas
epochs = 5000;

println("Se va a entrenar una red con \n batch_size: $(batch_size) \n η: $(η) \n Optimizer: $(opt) \n Epochs: $(epochs)")

# ╔═╡ 6095ef3e-8bb4-4f05-9ea3-44a507ab955c
# Entrenamos la red neuronal con el loss mse
for epoch in 1:epochs
    Flux.train!(loss, Flux.params(p), data, opt, cb=cb)
end

# Entrenamiento con loss compuesto
# for epoch in 1:1
#     Flux.train!(composed_loss, Flux.params(model, opt), data, opt, cb=cb2)
# end

# ╔═╡ 3c2d8096-5e5e-4532-b7a4-cd4329650dc8
md"### Loss luego de reentrenar con los datos de validación sabiendo que no hay overfitting"

# ╔═╡ 6d4a3a7d-150b-47ff-8bbf-834ef72e823b
PlutoUI.Resource("https://imgur.com/5nBJcOT.png")

# ╔═╡ 3425db90-32bf-41e7-af9a-1ae6ca6c5391
md"* El error raíz cuadrática media de la red neuronal en el conjunto de entrenamiento es 0.02
"

# ╔═╡ 19d03741-42ab-4b64-b7d6-5db3b2e0a80f
PlutoUI.Resource("https://imgur.com/VDm8zpN.png")

# ╔═╡ 97150e65-7c94-47a3-a328-b40c918a871d
PlutoUI.Resource("https://imgur.com/bewxS7r.png")

# ╔═╡ 8f2842a6-687f-404b-a324-c49ca1baaf2a
md"* Este modelo funciona aún mejor que el anterior quizas porque esta vez la representación en baja dimensionalidad de donde entran los datos a esta red proviene de un autoencoder.

* El error RMSE en el conjunto de test es 0.02.

"

# ╔═╡ 3c3f4da7-90df-4d4f-bda1-49ba0612bfe0
PlutoUI.Resource("https://imgur.com/bNXEJ00.png")

# ╔═╡ 66745956-a54e-4f5b-bbdd-4ba3360aba09
md"Recordar que la represantación aunque cambió sigue siendo similar en el hecho que $\sigma$ y $l_{cm}$ pequeños es donde mas separadas y van aumentando a medida que se acercan a la zona donde hay mas puntos acumulados. Dejo un HTML en el trello con el gráfico que indica cuando pasas el mouse el sigma y en color el lcm" 

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"

[compat]
PlutoUI = "~0.7.59"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.9.1"
manifest_format = "2.0"
project_hash = "6e7bcec4be6e95d1f85627422d78f10c0391f199"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "b10d0b65641d57b8b4d5e234446582de5047050d"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.5"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.2+0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "b6d6bfdd7ce25b0f9b2f6b3dd56b2673a66c8770"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.5"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.10.11"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.21+4"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "8489905bcdbcfac64d1daa51ca07c0d8f0283821"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.1"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.9.0"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "ab55ee1510ad2af0ff674dbcced5e94921f867a9"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.59"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "9306f6085165d270f7e3db02af26a400d580f5c6"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.3"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.9.0"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "Pkg", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "5.10.1+6"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.Tricks]]
git-tree-sha1 = "eae1bb484cd63b36999ee58be2de6c178105112f"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.8"

[[deps.URIs]]
git-tree-sha1 = "67db6cc7b3821e19ebe75791a9dd19c9b1188f2b"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.8.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"
"""

# ╔═╡ Cell order:
# ╟─af30ef40-7c37-11ef-10ba-ef6a12ad1b39
# ╠═0f1a37ce-0f2d-4671-9d0a-b3a56d129a4b
# ╠═31f093df-2571-4321-89be-0eaa744279ab
# ╠═ac7b08a7-41ec-476f-8ca6-b2f7825989b3
# ╟─2ba1c854-9d0a-4d7f-8442-29ffc5a888d7
# ╠═0617918a-c3b2-428f-b6a7-41e37f9801e7
# ╠═bdd66e7d-7508-4c6f-97c3-7a659216936c
# ╟─d93a98b1-6896-47cd-8391-0017ba9a55c1
# ╟─6919310f-96a1-4437-ba22-a36ece690fe5
# ╠═1f697ef0-2201-4a5f-9a01-e1419b7b4d26
# ╠═484aa171-b846-4a27-825d-4967213c8ae1
# ╠═041cd100-9e5a-48ac-b377-476fcd30a17e
# ╠═6095ef3e-8bb4-4f05-9ea3-44a507ab955c
# ╟─3c2d8096-5e5e-4532-b7a4-cd4329650dc8
# ╠═019f3fd1-fc27-4d40-a672-a8e2bd90dfa2
# ╟─6d4a3a7d-150b-47ff-8bbf-834ef72e823b
# ╟─3425db90-32bf-41e7-af9a-1ae6ca6c5391
# ╟─19d03741-42ab-4b64-b7d6-5db3b2e0a80f
# ╟─97150e65-7c94-47a3-a328-b40c918a871d
# ╟─8f2842a6-687f-404b-a324-c49ca1baaf2a
# ╟─3c3f4da7-90df-4d4f-bda1-49ba0612bfe0
# ╟─66745956-a54e-4f5b-bbdd-4ba3360aba09
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
