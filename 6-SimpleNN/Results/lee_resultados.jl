using CSV
using DataFrames

folder_path = "C:/Users/Propietario/OneDrive/Escritorio/ib/Tesis_V1/MLonNMR/5-Simplificación/Results"  # Replace with the actual path to your folder
csv_files = readdir(folder_path, join=true) |> filter(file -> endswith(file, ".csv"))

# En caso que ya se haya corrido el script anterior y se haya generado un archivo con los resultados
pop!(csv_files)

for name in csv_files
    println(name)
end

rows = []

for path in csv_files
    df = CSV.read(path, DataFrame)
    push!(rows, df[1, :])
end

rows

df = DataFrame(rows)

sort!(df)

df[!,"RMSE Entren."] = round.(sqrt.(round.(df[!,"MSE_Train"], sigdigits = 4)), sigdigits = 4)
df[!, "RMSE Val."] = round.(sqrt.(round.(df[!, "MSE_Val"], sigdigits = 4)), sigdigits = 4)
df[!,"RMSE Test"] = round.(sqrt.(round.(df[!,"MSE_Test"], sigdigits = 4)), sigdigits = 4)
df[!, "Arq"] = replace.(df[!, "Arq"], "," => " ")

# Remove the dp_rate column from the dataframe
select!(df, Not(:dp_rate))
select!(df, Not(:Activ))
select!(df, Not(:Opt))
select!(df, Not(:MSE_Train))
select!(df, Not(:MSE_Val))
select!(df, Not(:MSE_Test))

for name in col_names
    println(name)
end

col_names = names(df)

# rename!(df, col_names .=>  ["ID", "Arq.", "Activ.", "Opt.", "Puntos Usados", "λ", "dp rate", "MSE Train", "MSE Test"])

# df[!,"MSE Train"] .= round.(df[!,"MSE Train"], digits=4)
# df[!, "MSE Test"] .= round.(df[!, "MSE Test"], digits=4)

CSV.write(folder_path * "/Resultados.csv", df)

minimo_loss_predict = minimum(df[!,"MSE_Test"])
id_min = df[df.MSE_Train .== minimo_loss_predict, :].ID
println("El minimo valor de Loss_Final_Predicción es: $minimo_loss_predict en la arquitectura con ID: $id_min")

minimo_loss_train = minimum(df[!,"MSE_Train"])
id_min_t = df[df.MSE_Train .== minimo_loss_train, :].ID
println("El minimo valor de Loss_Final_Entrenamiento es: $minimo_loss_train en la arquitectura con ID: $id_min_t")

df