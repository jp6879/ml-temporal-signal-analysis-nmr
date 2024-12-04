### A Pluto.jl notebook ###
# v0.19.42

using Markdown
using InteractiveUtils

# ╔═╡ eb0fcdd6-d5ba-4481-8e0d-15e35c961f08
using PlutoUI

# ╔═╡ 39f617a0-7c28-11ef-1026-0f229a92f088
md"# Autoencoders

* Para reemplazar PCA en un contexto en el cual necesitamos un modelo end-to-end totalmente diferenciable podemos usar autoencoders para reducir la dimensionalidad.

* Los autoencoders son redes neuronales con una estructura particular que nos permiten aprender una representación de menor dimensionalidad de los datos de entrada. Este espacio latente es una representación densa de los datos de entrada que puede ser usada para reconstruir los datos originales.

* En este caso sabemos que PCA da buenos resultados para reducir la dimensionalidad de los datos, por lo que vamos a comparar primero PCA con autoencoders con activaciones no lineales y luego con autoencoders con activaciones lineales. La métrica que vamos a utilizar es comparar el error de reconstrucción entre las señales originales y las reconstruidas."

# ╔═╡ 5d4d6f98-6740-46ff-847e-e9bf1a396da5
md"## PCA

* Utilizamos la implementación de PCA que habiamos hecho anteriormente."

# ╔═╡ 68f07600-1236-48b5-881f-332a36caf139
using MultivariateStats
using DataFrames
using CSV
using Statistics
using Flux
using Plots
using Random
using IterTools: ncycle
using ProgressMetera
using LinearAlgebra
using Measures
include("../1-GeneracionDatos/Parameters.jl");

# ╔═╡ 0a0c0cb0-4040-4942-8487-3283264186ea
signalsDF = transpose(Matrix(CSV.read("C:/Users/Propietario/OneDrive/Escritorio/ib/Tesis_V1/MLonNMR/1-GeneracionDatos/Data/SimpleSignalHahn_TE_1_G_8.73e-7.csv", DataFrame)))

# ╔═╡ b3328025-2f8e-42e3-b9b8-32dcf7adc162
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

# ╔═╡ bb07dbb8-8e10-4197-b9ee-1e47e3e6b339
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

# ╔═╡ f214c4f3-289b-481f-8e0a-e6cdf2f0caf0
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
    dataIN_C, _ = centerData(dataIN)

    # Esto ya hace PCA sobre la matriz dada donde cada observación es una columna de la matriz
    pca_model = fit(PCA, dataIN_C, maxoutdim=3)

    # Esta instancia de PCA tiene distintas funciones como las siguientes

    #projIN = projection(pca_model) # Proyección de los datos sobre los componentes principales

    # Vector con las contribuciones de cada componente (es decir los autovalores)
    pcsIN = principalvars(pca_model)

    # Obtenemos la variaza en porcentaje para cada componente principal
    explained_varianceIN = pcsIN / sum(pcsIN) * 100

    reducedIN = MultivariateStats.transform(pca_model, dataIN_C)

    return reducedIN, pca_model
end

# ╔═╡ 20a6febb-92a3-455c-bec4-baed13f52138
md"* En PCA es mejor centrar los datos para que tengan media 0 antes de aplicarlo, esto es algo que también vamos a repetir en los autoencoders."

# ╔═╡ 51b07e2b-25f0-44ef-a5a4-a81497572d21
PlutoUI.Resource("https://imgur.com/wFG7OXT.png")

# ╔═╡ e641b446-d6f9-4449-8d32-220f873d75c3
md"* Obtenemos el mismo resultado de siempre."

# ╔═╡ 22608d60-20b9-46e6-ad97-15ee9b90913a
md"## Error de Reconstrucción

* Usamos la inversa de la matriz que usamos para reducir la dimensionalidad con PCA y así reconstruir los datos originales y ver si se pierde información en este proceso de reconstrucción.

* La métrica que vamos a usar de error de reconstrucción es la raíz del error cuadrático medio (RMSE) calculada como $\sqrt{\frac{1}{N} \sum_{i = 1}^{N} \frac{1}{M} \sum_{j = 1}^{M} (S_j - \hat{S}_j)^2_i}$ donde $(S_j - \hat{S}_j)^2_i$ es la resta cuadrática entre de los puntos $j$ de la señal $i$ verdaderos contra la reconstrucción. $N$ es la cantidad de señales y $M$ el número de puntos en cada señal"

# ╔═╡ ccf19f32-8860-4145-bb96-e39262cb14e3
md" * Error MSE de reconstrucción de PCA: 4.7125315997926665e-5
* Error RMSE de reconstrucción de PCA: 0.006864788124765882"

# ╔═╡ 00f97ae4-b1b5-436a-839e-e62861261bab
md"## Autoencoders

* A diferencia de PCA, los autoencoders pueden aprender representaciones no lineales de los datos, sin embargo hay un gran paralelismo entre PCA y autoencoders cuando estos tienen una sola capa oculta con una capa de activación lineal. Como dijimos antes los autoencoders tienen una estructura particular que consiste en dos redes neuronales un encoder y un decoder. El encoder toma los datos de entrada como M dimensiones y los reduce a un espacio latente de menor dimensionalidad. El decoder toma el espacio latente y lo reconstruye a las M dimensiones originales. En este caso el espacio latente que vamos a querer aprender es de 3 dimensiones que es lo que nos da PCA por defecto sin perder casi nada de información.

![autoencoder](https://upload.wikimedia.org/wikipedia/commons/2/28/Autoencoder_structure.png)

* Vamos a explorar tres autoencoders distintos uno simple con una única capa oculta, otro con 4 capas ocultas y otro con 6 capas ocultas. En todos los casos vamos a comparar autoencoders con activaciones lineales y no lineales.

* Como vimos que PCA funciona bien con 3 dimensiones la dimension del espacio latente en los autoencoders va a ser de 3.

* Para evitar el overfitting vamos a utilizar regularización L2 en los pesos de las capas ocultas, además usamos early stopping que consiste en detener el entrenamiento cuando el error de validación deja de disminuir durante 100 épcas.

* El error de reconstrucción lo vamos a calcular de la misma forma que en PCA para poder comparar los resultados.

* También los datos van a ser comparados con la señal centrada "

# ╔═╡ 384f6e6f-c253-4491-8fa3-127d58eac24c
md"# Autoencoder Simple

* Este autoencoder tiene una sola capa oculta con 3 neuronas para aprender la representación de menor dimensionalidad de las señales de entrada."

# ╔═╡ f23ac298-c21d-419a-8d50-d200237d9a5e
md"### Con activaciones lineales"

# ╔═╡ 342b181d-f6a5-4f3e-b34b-d4d185e6aec7
# Split en training, validation y test
n_signals = size(signalsDF_C, 2)
n_train = Int(floor(n_signals*0.7))
n_val = Int(floor(n_signals*0.15))
n_test = n_signals - n_train - n_val

train_signals = Float32.(Matrix(signalsDF_C[:, 1:n_train]))
val_signals = Float32.(Matrix(signalsDF_C[:, n_train+1:n_train+n_val]))
test_signals = Float32.(Matrix(signalsDF_C[:, n_train+n_val+1:end]))

train_params = pdistparamsDF[1:n_train, :]
val_params = pdistparamsDF[n_train+1:n_train+n_val, :]
test_params = pdistparamsDF[n_train+n_val+1:end, :];

# Number of dimension of the unreduced data
n_times = size(train_signals, 1)

# Autoencoder simple
encoderSimpleLineal = Chain(Dense(n_times, 3, identity, bias = false))
decoderSimpleLineal = Chain(Dense(3, n_times, identity, bias = false))
autoencoderSimpleLineal = Chain(encoderSimpleLineal, decoderSimpleLineal)

# Destructure in parameters and NN structure
s_params, s_re = Flux.destructure(autoencoderSimple)

# Dataloader and mini-batching for traininig
loader = Flux.Data.DataLoader((train_signals, train_signals), batchsize=50, shuffle=true)

# Initial learning rate
lr = 0.001

# Optimizer used
optim = Flux.setup(Flux.AdamW(lr), autoencoderSimple)

# Number of epochs
num_epochs = 500
# Patience for reducing learning rate and checking if the val loss has decresed
patience_epochs = 100

# Loss function
function loss_re(x, y)
    return Flux.mse(s_re(x), y)
end

# Losses history
lossesSimple = []
lossesSimpleVal = []

# Training loop
for epoch in 1:num_epochs
    for (x, y) in loader
        loss, grads = Flux.withgradient(autoencoderSimple) do m
            Flux.mse(m(x), y)
        end 
        Flux.update!(optim, autoencoderSimple, grads[1])
    end
    actual_loss = Flux.mse(autoencoderSimple(train_signals), train_signals)
    actual_loss_val = Flux.mse(autoencoderSimple(val_signals), val_signals)

	# Saving best model
    if epoch > 1  && (actual_loss_val < minimum(lossesSimpleVal))

        encoder_params, encoder_re = Flux.destructure(autoencoderSimple[1])
        decoder_params, decoder_re = Flux.destructure(autoencoderSimple[2])

        df_encoder = DataFrame(reshape(encoder_params, length(encoder_params), 1), :auto)
        df_decoder = DataFrame(reshape(decoder_params, length(decoder_params), 1), :auto)

        CSV.write("C:/Users/Propietario/OneDrive/Escritorio/ib/Tesis_V1/MLonNMR/2-2-Autoencoders/Models/minAE_PCAParamsE.csv", df_encoder)
        CSV.write("C:/Users/Propietario/OneDrive/Escritorio/ib/Tesis_V1/MLonNMR/2-2-Autoencoders/Models/minAE_PCAParamsD.csv", df_decoder)
    end
	
    push!(lossesSimple, actual_loss)
    push!(lossesSimpleVal, actual_loss_val)
    println("Epoch: ", epoch, " Loss: ", actual_loss, " Loss Val: ", actual_loss_val)

	# Early stopping and decreasing learning rate after patience epochs
    if epoch % patience_epochs == 0
        Flux.adjust!(optim, lr * 0.1)
        loss_val_prev = lossesSimpleVal[end-patience_epochs+1]
        if actual_loss_val > loss_val_prev
            println("Early stopping at epoch: $epoch, because the validation loss is not decreasing after $patience_epochs epochs")
            println("Loss de validación mínimo: ", minimum(lossesSimpleVal), " en la época: ", argmin(lossesSimpleVal))
            break
        end
    end

	# Cleaning garbage memory	
    GC.gc()
end

# ╔═╡ 75f273a0-c6a6-48fe-b50c-efb0bd412b2c
PlutoUI.Resource("https://imgur.com/j3BShYh.png")

# ╔═╡ cdc6ca3e-4822-41ae-b30e-59f8d9351d52
md"* El loss de validación y de entrenamiento son muy similares por lo que no hay peligro de overfitting"

# ╔═╡ 59416449-bc31-44ed-a3dc-aee7998581b0
md" * El error de recosntrucción RMSE del autoencoder simple en el conjunto de señales de test es de: 0.008394564
* Comparado con el error de reconstrucción de PCA (0.006864788124765882) este error del autoencoder simple con activacion lineal es mayor pero del orden."

# ╔═╡ f663720e-82c1-47fa-80f3-7d59302d01e2
md"* Grafiquemos el espacio reducido en 2D distinguiendo las señales por $l_{cm}$ como hicimos con PCA."

# ╔═╡ 6e00a071-e29e-4197-bded-420f43d9fd4d
PlutoUI.Resource("https://imgur.com/S8tUZFD.png")

# ╔═╡ ec9802e6-7d31-4be5-9990-3b2987131396
PlutoUI.Resource("https://imgur.com/EqRwI2z.png")

# ╔═╡ 5b37c3fd-d806-4d8b-9aba-9beca1cf229f
PlutoUI.Resource("https://imgur.com/XXKbF7g.png")

# ╔═╡ 90d505c5-b281-4430-a9e1-621e5e69f30e
md"* Las representaciones de las señales en el espacio latente son muy similares a las de PCA como era de esperarse ya que con activaciones lineales esto es casi equivalente a PCA. Si lo graficamos con plotly podemos tener una mejor visualización."

# ╔═╡ c14ed2f3-8d89-42d8-8979-fcba9dc1e131
PlutoUI.Resource("https://imgur.com/8ysaTrD.png")

# ╔═╡ c6e77f4d-7714-4ce9-8e42-1a54713a8ac4
PlutoUI.Resource("https://imgur.com/6iQO2uY.png")

# ╔═╡ b10f33a4-2a42-41c9-a84e-43b5e0efcc12
md"### Autoencoder simple con activaciones no lineales"

# ╔═╡ 235fc2d0-f6fc-4b19-8048-715fb58dbc0a
# Autoencoder simple
encoderSimpleNoLineal = Chain(Dense(n_times, 3, relu, bias = false))
decoderSimpleNoLineal = Chain(Dense(3, n_times, relu, bias = false))
autoencoderSimpleNoLineal = Chain(encoderSimpleNoLineal, decoderSimpleNoLineal)

# ╔═╡ 426b7b30-a2a2-4197-b637-847ba2b79980
PlutoUI.Resource("https://imgur.com/icuAGu5.png")

# ╔═╡ 91b53a16-5c25-446f-a982-e842b016b5f5
md"* Lo mismo pasa con activaciones no lineales"

# ╔═╡ 67edab72-1756-4cef-851a-3f5cc163f340
md"* Calculando el error de recosntrucción RMSE del autoencoder simple en el conjunto de test: 0.008394564
* Comparado con el error de reconstrucción de PCA (0.006864788124765882) este error del autoencoder simple con activacion no lineal es mayor.
* Cuando graficamos la reducción de dimensionalidad queda algo similar al caso anterior con activaciones lineales, esto puede deberse a que la estructura del autoencoder es similar.
"

# ╔═╡ dcac69d1-5aaa-4e20-8a62-d9f1af7a52eb
md"# Autoencoder con 4 capas ocultas

## Caso con activaciones lineales
* En este caso vamos a ver la representación de las señales en el espacio latente con el autoencoder con 4 capas ocultas y activaciones lineales.
* Vamos a comparar el error de reconstrucción con PCA."

# ╔═╡ ebd60f2c-90f7-4191-b704-149fe7cf40e4
encoderDeepLineal = Chain(Dense(n_times, 100, identity, bias = false), Dense(100, 50, identity), Dense(50, 3, identity))

decoderDeepLineal = Chain(Dense(3, 50, identity, bias = false), Dense(50, 100, identity), Dense(100, n_times, identity))

autoencoderDeepLineal = Chain(encoderDeepLineal, decoderDeepLineal)

# ╔═╡ e751bcf6-3a90-47f3-ba71-723fc2962a7c
PlutoUI.Resource("https://imgur.com/23jVgZv.png")

# ╔═╡ b3e3a075-3d51-4c3b-b956-858a32c688d7
md"* Error de recosntrucción RMSE del autoencoder simple en el conjunto de test: 0.006836749
* Comparado con el error de reconstrucción de PCA (0.006864788124765882) este error del autoencoder con 4 capas ocultas y activacion lineal es casi igual.
"

# ╔═╡ 24684b56-e849-4c46-88bd-60f80c70cc87
md"Si vemos la recosntrucción de los datos:"

# ╔═╡ f1c8147a-fa45-4434-9f8a-2a73872ff623
PlutoUI.Resource("https://imgur.com/9iXpxYG.png")

# ╔═╡ 33da9113-864f-4382-ae5b-fde13a45a379
md"También es similar al PCA original"

# ╔═╡ 268412d1-af6e-4338-8034-278bbf8f84f1
md"## Caso con activaciones no lineales"

# ╔═╡ b67d9e2b-510e-4880-9dcc-b0e4fc1b2886
encoderDeepNoLineal = Chain(Dense(n_times, 100, bias = false), Dense(100, 50, relu), Dense(50, 3))
decoderDeepNoLineal = Chain(Dense(3, 50, bias = false), Dense(50, 100, relu), Dense(100, n_times))
autoencoderDeepNoLineal = Chain(encoderDeepNoLineal, decoderDeepNoLineal)

# ╔═╡ df6aa69f-06ad-4864-a2ca-66d513748869
PlutoUI.Resource("https://imgur.com/fHt4YNB.png")

# ╔═╡ 0478b11a-9fe6-400c-8df7-bd6a31e54889
md"* Error de recosntrucción RMSE del autoencoder no lineal profundo en el conjunto de test: 0.0016314732
* Comparado con el error de reconstrucción de PCA (0.006864788124765882) este error del autoencoder con 4 capas ocultas y activacion no lineal es menor pero también del orden.
"

# ╔═╡ 7e326706-12d9-4e92-8c57-116487e6f67f
PlutoUI.Resource("https://imgur.com/abj0N1j.png")

# ╔═╡ bb901ecd-3d47-4eb8-9b76-9dc7c7d13564
md"Los cambios son que con las activaciones no lineales ahora los datos parecen estar menos amontonados para valores de lcm y sigma altos. sin embargo el error de reconstrucción es similar al de PCA"

# ╔═╡ e228d648-8936-4855-bcc6-510689215392
md"# Autoencoder con 6 capas ocultas

* En este caso vamos a ver la representación de las señales en el espacio latente con el autoencoder con 6 capas ocultas y activaciones lineales."

# ╔═╡ 1ed4c2de-ac44-4b92-862e-419fec589a85
PlutoUI.Resource("https://imgur.com/RBCJEvE.png")

# ╔═╡ 6542d89f-bb0a-4a6a-8ae5-6a77b973514f
md"* Error de recosntrucción RMSE del autoencoder simple en el conjunto de test: 0.006838203
* Comparado con el error de reconstrucción de PCA (0.006864788124765882) este error del autoencoder con 6 capas ocultas y activacion lineal es similar.

* El error aumentó al agregar capas siendo un autoencoder con activaciones lineales.
"

# ╔═╡ 9d92fee1-15eb-497f-8a52-2487ca456984
PlutoUI.Resource("https://imgur.com/tDQ1WTI.png")

# ╔═╡ ee8c7420-31d4-424d-839c-fc0829411f7c
md"Ahora la representación está cambiando en el sentido que parece rotarse con respecto a las otras componentes, esto tiene sentido en el hecho de que los autoencoders no tienen necesariamente las mismas restricciones de ortogonalidad que impone el PCA, lo que genera diferencias en las representaciones."

# ╔═╡ ef3cc6fd-2190-46f4-a173-946a4da6dc10
md"En el caso del error este aumentó con respecto al anteriro autoencoder lineal o bueno es casi igual, por lo que probablente sea el límite y seguir aumentando las capas lleve al overfitting."

# ╔═╡ 22e47b3e-19d6-40ad-a1f3-ba2a15f601ce
md"## Caso con activaciones no lineales"

# ╔═╡ 34c66675-db5c-4670-87a1-2ad065634aa8
encoderDeeperNoLineal = Chain(Dense(n_times, 500, bias = false), Dense(500, 100, relu), Dense(100, 50, relu), Dense(50, 3))
decoderDeeperNoLineal = Chain(Dense(3, 50, bias = false), Dense(50, 100, relu), Dense(100, 500, relu), Dense(500, n_times))
autoencoderDeeperNoLineal = Chain(encoderDeeperNoLineal, decoderDeeperNoLineal)

# ╔═╡ 3fa5dfe0-065d-4c66-b27c-863ef84e1f2d
PlutoUI.Resource("https://imgur.com/3g6BJOu.png")

# ╔═╡ 6106bd41-16c3-4128-b890-dc044ef32d51
md"* Error de recosntrucción RMSE del autoencoder simple en el conjunto de test: 0.00056004466
* Comparado con el error de reconstrucción de PCA (0.006864788124765882) este error del autoencoder con 6 capas ocultas y activacion no lineal es mucho menor, incluso menor que el autoencoder no lineal con 4 capas ocultas.
"

# ╔═╡ 1db5edbd-75f7-4b45-98c5-e9016755f595
PlutoUI.Resource("https://imgur.com/mVrrJw6.png")

# ╔═╡ b53c9d48-469f-48a1-a821-49fc5d62c596
PlutoUI.Resource("https://imgur.com/NjgLsly.png")

# ╔═╡ 37be9491-5fb8-4c83-9cc5-99cb0dab6ab2
md"Ahora la represenación si camibia con respecto a PCA y la transformación que se aprende puede involucrar cambios no lineales en el espacio latente que no tienen una interpretación tan directa como en el PCA"

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
# ╟─39f617a0-7c28-11ef-1026-0f229a92f088
# ╟─5d4d6f98-6740-46ff-847e-e9bf1a396da5
# ╠═eb0fcdd6-d5ba-4481-8e0d-15e35c961f08
# ╠═68f07600-1236-48b5-881f-332a36caf139
# ╠═0a0c0cb0-4040-4942-8487-3283264186ea
# ╠═b3328025-2f8e-42e3-b9b8-32dcf7adc162
# ╠═bb07dbb8-8e10-4197-b9ee-1e47e3e6b339
# ╠═f214c4f3-289b-481f-8e0a-e6cdf2f0caf0
# ╟─20a6febb-92a3-455c-bec4-baed13f52138
# ╟─51b07e2b-25f0-44ef-a5a4-a81497572d21
# ╟─e641b446-d6f9-4449-8d32-220f873d75c3
# ╟─22608d60-20b9-46e6-ad97-15ee9b90913a
# ╟─ccf19f32-8860-4145-bb96-e39262cb14e3
# ╟─00f97ae4-b1b5-436a-839e-e62861261bab
# ╟─384f6e6f-c253-4491-8fa3-127d58eac24c
# ╟─f23ac298-c21d-419a-8d50-d200237d9a5e
# ╠═342b181d-f6a5-4f3e-b34b-d4d185e6aec7
# ╟─75f273a0-c6a6-48fe-b50c-efb0bd412b2c
# ╟─cdc6ca3e-4822-41ae-b30e-59f8d9351d52
# ╟─59416449-bc31-44ed-a3dc-aee7998581b0
# ╟─f663720e-82c1-47fa-80f3-7d59302d01e2
# ╟─6e00a071-e29e-4197-bded-420f43d9fd4d
# ╟─ec9802e6-7d31-4be5-9990-3b2987131396
# ╟─5b37c3fd-d806-4d8b-9aba-9beca1cf229f
# ╟─90d505c5-b281-4430-a9e1-621e5e69f30e
# ╟─c14ed2f3-8d89-42d8-8979-fcba9dc1e131
# ╟─c6e77f4d-7714-4ce9-8e42-1a54713a8ac4
# ╟─b10f33a4-2a42-41c9-a84e-43b5e0efcc12
# ╠═235fc2d0-f6fc-4b19-8048-715fb58dbc0a
# ╟─426b7b30-a2a2-4197-b637-847ba2b79980
# ╟─91b53a16-5c25-446f-a982-e842b016b5f5
# ╟─67edab72-1756-4cef-851a-3f5cc163f340
# ╟─dcac69d1-5aaa-4e20-8a62-d9f1af7a52eb
# ╠═ebd60f2c-90f7-4191-b704-149fe7cf40e4
# ╟─e751bcf6-3a90-47f3-ba71-723fc2962a7c
# ╟─b3e3a075-3d51-4c3b-b956-858a32c688d7
# ╟─24684b56-e849-4c46-88bd-60f80c70cc87
# ╟─f1c8147a-fa45-4434-9f8a-2a73872ff623
# ╟─33da9113-864f-4382-ae5b-fde13a45a379
# ╟─268412d1-af6e-4338-8034-278bbf8f84f1
# ╠═b67d9e2b-510e-4880-9dcc-b0e4fc1b2886
# ╟─df6aa69f-06ad-4864-a2ca-66d513748869
# ╟─0478b11a-9fe6-400c-8df7-bd6a31e54889
# ╟─7e326706-12d9-4e92-8c57-116487e6f67f
# ╟─bb901ecd-3d47-4eb8-9b76-9dc7c7d13564
# ╟─e228d648-8936-4855-bcc6-510689215392
# ╟─1ed4c2de-ac44-4b92-862e-419fec589a85
# ╟─6542d89f-bb0a-4a6a-8ae5-6a77b973514f
# ╟─9d92fee1-15eb-497f-8a52-2487ca456984
# ╟─ee8c7420-31d4-424d-839c-fc0829411f7c
# ╟─ef3cc6fd-2190-46f4-a173-946a4da6dc10
# ╟─22e47b3e-19d6-40ad-a1f3-ba2a15f601ce
# ╠═34c66675-db5c-4670-87a1-2ad065634aa8
# ╟─3fa5dfe0-065d-4c66-b27c-863ef84e1f2d
# ╟─6106bd41-16c3-4128-b890-dc044ef32d51
# ╟─1db5edbd-75f7-4b45-98c5-e9016755f595
# ╟─b53c9d48-469f-48a1-a821-49fc5d62c596
# ╟─37be9491-5fb8-4c83-9cc5-99cb0dab6ab2
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
