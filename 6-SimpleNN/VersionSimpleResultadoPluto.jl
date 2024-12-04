### A Pluto.jl notebook ###
# v0.19.42

using Markdown
using InteractiveUtils

# ╔═╡ 00512935-e488-436e-be6b-f1cd34de0eab
using PlutoUI

# ╔═╡ b7fecde0-6f9e-11ef-2d7f-e55652d38868
md"# Modelo simplificado

* Una de las ideas era no pasar ni por NODE, ni reducción de dimensionalidad ni nada. Simplemente usar puntos de las señales de Hahn para obtener los parámetros de las distribuciones de probabilidad subyacentes.

* Esta simplificación parecía no tener sentido porque por lo menos para que la red neuronal funcione con la entrada en dimensionalidad reducida se necesitaban muchos datos de la señal.
"

# ╔═╡ 67e69b0e-079e-4bc7-84d9-05b932559a22
md"Si comenzamos con esta idea, una pregunta sería: ¿qué tiempos tomar para medir la señal?

Esto es complicado porque todas las señales son muy distintas entre sí; algunas, al llegar a 0.1 s, ya decayeron completamente, otras aún no, y otras ni siquiera comienzan a mostrar el comportamiento de decaimiento exponencial. Esto depende mucho de los parámetros de las distribuciones de tamaño subyacentes $l_{cm}$ (tamaño de correlación medio) y $\sigma$ (desviación estándar). En este caso, como no queremos perdernos información de ninguna señal, hacemos lo siguiente: si vamos a tomar $N$ puntos de la señal para hacer de entrada a la red, el 90% va a pertenecer al intervalo de tiempo $t \in (0, 0.1)$ s. El resto al intervalo $t \in (0.1, 1)$ s.

Al final tenemos datos de entrada como las que se muestran en las imágenes para $N = 5,~10,~20$. Estos son para algunas señales de nuestro dataset"

# ╔═╡ 77c417cb-e676-4835-8e68-d1a35795df93
PlutoUI.Resource("https://imgur.com/Ut03Wr5.png")

# ╔═╡ eb879ab8-7db4-47cb-b225-f8205d28755f
PlutoUI.Resource("https://imgur.com/mHmIRLy.png")

# ╔═╡ 06f2a86f-0d71-4353-9494-2705afc3636f
PlutoUI.Resource("https://imgur.com/wxof11u.png")

# ╔═╡ ec1ad253-0d35-46b0-afdd-1f5c647daa1c
md"Haciendo una exploración de hiperparámetros donde nuestras redes van desde $N$ hasta un par $(l_{cm},~\sigma)$ donde se explora, tamaño de la arquitecura, numero de datos de entrada, lambda de regularización L2, y cantidad de señales en el dataset (dejé solo las que tienen todo el dataset) tenemos:"

# ╔═╡ cd34de2d-7b05-49b8-82fa-36665d7cb81c
md"$\begin{aligned}
   & \begin{array}{cccccccc}
        ID & Arq & N & Lambd & MSETrain & MSEVal & MSETest & NumS \\ 


        7 & [6,  16,  32,  16,  2] & 5 & 0.0 & 0.06677 & 0.07003 & 0.06706 & 55100 \\ 
        8 & [6,  32,  64,  16,  2] & 5 & 0.0 & 0.0793 & 0.08181 & 0.07917 & 55100 \\ 
        9 & [6,  16,  32,  16,  8,  2] & 5 & 0.0 & 0.09322 & 0.09717 & 0.09262 & 55100 \\ 
        10 & [6,  32,  64,  16,  8,  2] & 5 & 0.0 & 0.09323 & 0.09738 & 0.09224 & 55100 \\
        11 & [6,  30,  25,  20,  15,  10,  2] & 5 & 0.0 & 0.05589 & 0.05868 & 0.05624 & 55100 \\
        12 & [6,  32,  64,  32,  16,  2] & 5 & 0.0 & 0.09216 & 0.09611 & 0.09142 & 55100 \\ 
        19 & [6,  16,  32,  16,  2] & 5 & 0.1 & 0.9587 & 0.9504 & 0.961 & 55100 \\
        20 & [6,  32,  64,  16,  2] & 5 & 0.1 & 0.9587 & 0.9504 & 0.961 & 55100 \\
        21 & [6,  16,  32,  16,  8,  2] & 5 & 0.1 & 0.9859 & 0.978 & 0.9881 & 55100 \\ 
        22 & [6,  32,  64,  16,  8,  2] & 5 & 0.1 & 0.9859 & 0.978 & 0.9881 & 55100 \\ 
        23 & [6,  30,  25,  20,  15,  10,  2] & 5 & 0.1 & 1.021 & 1.014 & 1.023 & 55100 \\ 
        24 & [6,  32,  64,  32,  16,  2] & 5 & 0.1 & 0.9859 & 0.978 & 0.9881 & 55100 \\ 
        31 & [11,  16,  32,  16,  2] & 10 & 0.0 & 0.003579 & 0.003672 & 0.003597 & 55100 \\
        32 & [11,  32,  64,  16,  2] & 10 & 0.0 & 0.003017 & 0.003084 & 0.002987 & 55100 \\
        33 & [11,  16,  32,  16,  8,  2] & 10 & 0.0 & 0.001993 & 0.002094 & 0.002069 & 55100 \\
        34 & [11,  32,  64,  16,  8,  2] & 10 & 0.0 & 0.005792 & 0.005746 & 0.00587 & 55100 \\
        35 & [11,  30,  25,  20,  15,  10,  2] & 10 & 0.0 & 0.003659 & 0.003815 & 0.003662 & 55100 \\
        36 & [11,  32,  64,  32,  16,  2] & 10 & 0.0 & 0.005 & 0.005135 & 0.005094 & 55100 \\
        43 & [11,  16,  32,  16,  2] & 10 & 0.1 & 0.8009 & 0.7974 & 0.8038 & 55100 \\ 

        45 & [11,  16,  32,  16,  8,  2] & 10 & 0.1 & 0.8281 & 0.8251 & 0.8309 & 55100 \\ 
        46 & [11,  32,  64,  16,  8,  2] & 10 & 0.1 & 0.8281 & 0.8251 & 0.8309 & 55100 \\ 
        47 & [11,  30,  25,  20,  15,  10,  2] & 10 & 0.1 & 0.8633 & 0.8607 & 0.8661 & 55100 \\
        48 & [11,  32,  64,  32,  16,  2] & 10 & 0.1 & 0.8281 & 0.8251 & 0.8309 & 55100 \\
        55 & [21,  16,  32,  16,  2] & 20 & 0.0 & 0.0003108 & 0.0003169 & 0.0003186 & 55100 \\
        56 & [21,  32,  64,  16,  2] & 20 & 0.0 & 0.0002175 & 0.0002201 & 0.0002262 & 55100 \\
        57 & [21,  16,  32,  16,  8,  2] & 20 & 0.0 & 0.0001325 & 0.0001383 & 0.0001379 & 55100 \\
        58 & [21,  32,  64,  16,  8,  2] & 20 & 0.0 & 0.0001117 & 0.0001134 & 0.0001144 & 55100 \\
        59 & [21,  30,  25,  20,  15,  10,  2] & 20 & 0.0 & 0.0001084 & 0.0001144 & 0.0001127 & 55100 \\
        60 & [21,  32,  64,  32,  16,  2] & 20 & 0.0 & 7.564e-5 & 7.897e-5 & 7.734e-5 & 55100 \\
        67 & [21,  16,  32,  16,  2] & 20 & 0.1 & 0.7518 & 0.7494 & 0.7542 & 55100 \\ 
        68 & [21,  32,  64,  16,  2] & 20 & 0.1 & 0.7518 & 0.7494 & 0.7542 & 55100 \\ 
        69 & [21,  16,  32,  16,  8,  2] & 20 & 0.1 & 0.7756 & 0.7738 & 0.7778 & 55100 \\ 
        70 & [21,  32,  64,  16,  8,  2] & 20 & 0.1 & 0.7756 & 0.7738 & 0.7778 & 55100 \\ 
        71 & [21,  30,  25,  20,  15,  10,  2] & 20 & 0.1 & 0.8079 & 0.8062 & 0.8099 & 55100 \\ 
        72 & [21,  32,  64,  32,  16,  2] & 20 & 0.1 & 0.7756 & 0.7738 & 0.7778 & 55100 \\
    \end{array}
\end{aligned}$"

# ╔═╡ 1e683a58-199b-42bc-85ec-7a5f3f41d317
md"Como se ve ciertas arquitecturas con 20 puntos de entrada (mas el 1 que siempre es conocido) tienen valores de MSE bajos, tanto que la arquitectura 60, el MSE tanto para entrenamiento, validación y test es del orden de las mejores redes obtenidas al hacer una red que reciba únicamente las 3 componentes principales luego de reducir dimensionalidad. 

El loss MSE en función de las épocas de entrenamiento es este"

# ╔═╡ 56dde49f-27b1-4ed3-b1ed-2de1c7c7631b
PlutoUI.Resource("https://imgur.com/TjSTdWm.png")

# ╔═╡ ecefd3ab-9ae9-49d5-9e16-6e59b23cc6b9
md"Los errores Mean Absolute Error de todo los puntos y Root Mean Absolut error de todas las predicciones son:

* MAE Train: 0.0065
* RMAE Train: 0.0087

* MAE Valid: 0.0067
* RMAE Valid: 0.0089"

# ╔═╡ 6ffa082b-03ed-49a3-b433-d6f1f3de4e71
md"Los mejores y peores errores en la predicción de parámetros"

# ╔═╡ a612d31c-32bd-4c21-8334-c31c42443ca1
PlutoUI.Resource("https://imgur.com/RN1EqDF.png")

# ╔═╡ 0f20df8b-6080-4ffb-ad17-52c59f212ac3
PlutoUI.Resource("https://imgur.com/HcJvYxo.png")

# ╔═╡ cd08bad1-c32a-457e-8de8-80581c65cde7
md"Si observamos el error en la predicción de los parámetros de manera individual utilizando el error RMSE en los datos de test, obtenemos el siguiente mapa de calor. Esta grilla tiene aproximadamente 5000 predicciones de los parámetros en la grilla de $l_{cm}$ y $\sigma$.
"

# ╔═╡ 73192037-732f-4691-9d38-7cbba5577996
PlutoUI.Resource("https://imgur.com/mLNEw3g.png")

# ╔═╡ 48ee13dd-1f72-418d-9509-95de781e6c25
md"Vemos que el máximo en datos de señales con las que la red no fue entrenada el error no pasa el 5% con esta métrica de error, lo cual es suficientemente bueno para lo buscado."

# ╔═╡ 2865d46e-36d6-485f-badf-bae40db57819
md"Teniendo esto en mente se puede hacer un bypass a la reducción de dimensionalidad, porque podemos hacer que a partir de suponiendo 5 puntos una red (NODE u otra) nos de los restantes para poder entrar a la red neruonal entrenada y predecir los parámetros.
"

# ╔═╡ 3cb09aa8-b5d3-498d-8cc2-961d91958a7a
PlutoUI.Resource("https://imgur.com/EpX5nG1.png")

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
# ╟─b7fecde0-6f9e-11ef-2d7f-e55652d38868
# ╟─67e69b0e-079e-4bc7-84d9-05b932559a22
# ╠═00512935-e488-436e-be6b-f1cd34de0eab
# ╟─77c417cb-e676-4835-8e68-d1a35795df93
# ╟─eb879ab8-7db4-47cb-b225-f8205d28755f
# ╟─06f2a86f-0d71-4353-9494-2705afc3636f
# ╟─ec1ad253-0d35-46b0-afdd-1f5c647daa1c
# ╟─cd34de2d-7b05-49b8-82fa-36665d7cb81c
# ╟─1e683a58-199b-42bc-85ec-7a5f3f41d317
# ╟─56dde49f-27b1-4ed3-b1ed-2de1c7c7631b
# ╟─ecefd3ab-9ae9-49d5-9e16-6e59b23cc6b9
# ╟─6ffa082b-03ed-49a3-b433-d6f1f3de4e71
# ╟─a612d31c-32bd-4c21-8334-c31c42443ca1
# ╟─0f20df8b-6080-4ffb-ad17-52c59f212ac3
# ╟─cd08bad1-c32a-457e-8de8-80581c65cde7
# ╟─73192037-732f-4691-9d38-7cbba5577996
# ╟─48ee13dd-1f72-418d-9509-95de781e6c25
# ╟─2865d46e-36d6-485f-badf-bae40db57819
# ╟─3cb09aa8-b5d3-498d-8cc2-961d91958a7a
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
