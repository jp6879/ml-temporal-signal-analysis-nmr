using CSV
using DataFrames

folder_path = "C:/Users/Propietario/OneDrive/Escritorio/ib/Tesis_V1/MLonNMR/PANODE/Results"  # Replace with the actual path to your folder
csv_files = readdir(folder_path, join=true) |> filter(file -> endswith(file, ".csv"))

pop!(csv_files)  # Remove the last element of the array, which is the file "Resultados_expl.csv"

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

CSV.write(folder_path * "/Resultados_expl.csv", df)

# Drop dp column
df = select(df, Not(:dp))
df = select(df, Not(:Opt))
df = select(df, Not(:lambd))
df = select(df, Not(:Eta))
# Round the column Loss_Final_Entrenamiento to 4 decimal places
df[!,"Loss_Final_Entrenamiento"] = round.(df[!,"Loss_Final_Entrenamiento"], sigdigits=4)
df[!,"Loss_Final_Predicción"] = round.(df[!,"Loss_Final_Predicción"], sigdigits=4)


CSV.write(folder_path * "/Resultados_expl.csv", df)

# Sort by Num_data
df = sort(df, :Num_data)

minimo_loss_predict = minimum(df[!,"MSEPred"])
id_min = df[df.MSEPred .== minimo_loss_predict, :].ID
println("El minimo valor de Loss_Final_Predicción es: $minimo_loss_predict en la arquitectura con ID: $id_min")

minimo_loss_train = minimum(df[!,"MSETrain"])
id_min_t = df[df.MSETrain .== minimo_loss_train, :].ID
println("El minimo valor de Loss_Final_Entrenamiento es: $minimo_loss_train en la arquitectura con ID: $id_min_t")