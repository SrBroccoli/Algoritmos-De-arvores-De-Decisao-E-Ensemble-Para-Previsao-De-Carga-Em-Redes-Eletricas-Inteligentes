# D_pratico(1).py - código comentado
# Comentários adicionados para explicar a finalidade de cada bloco; a lógica original foi mantida.

# Bibliotecas utilizadas no processamento dos dados, treinamento dos modelos e geração dos gráficos.
import sklearn.tree as sktree
from sklearn.ensemble import RandomForestRegressor
import sklearn.metrics
import pandas as pd
import matplotlib.pyplot as plt
# Bibliotecas utilizadas no processamento dos dados, treinamento dos modelos e geração dos gráficos.
import numpy as np

# Define o arquivo que contém os dados históricos de consumo de energia.
energyPredictionFilePath = 'AEP_hourly.csv'
# Carrega os dados do arquivo CSV para um DataFrame do pandas.
energyData = pd.read_csv(energyPredictionFilePath) 
# Remove linhas que possuem valores ausentes.
energyData = energyData.dropna(axis=0)

# Define os índices que delimitam os conjuntos de treinamento e de previsão.
beginningOfTraining = 0
# Define os índices que delimitam os conjuntos de treinamento e de previsão.
endOfTraining = int(len(energyData)*0.8)
# Define o início da previsão e o intervalo utilizado para visualização dos resultados.
beginningOfPrediction = endOfTraining + 1
endOfPrediction = len(energyData)

# Define o início da previsão e o intervalo utilizado para visualização dos resultados.
beginningOfPredictionPlot = beginningOfPrediction
endOfPredictionPlot = beginningOfPredictionPlot + 168

# Converte a coluna de data e hora para o formato datetime.
energyData["Datetime"] = pd.to_datetime(energyData["Datetime"])

# Extrai informações de ano, mês, dia e hora para utilizar como variáveis de entrada.
energyData["Year"] = energyData["Datetime"].dt.year
energyData["Month"] = energyData["Datetime"].dt.month
energyData["Day"] = energyData["Datetime"].dt.day
energyData["Hour"] = energyData["Datetime"].dt.hour

energyData["AEP_t-1"] = energyData["AEP_MW"].shift(1)
energyData["AEP_t-2"] = energyData["AEP_MW"].shift(2) # Adicional em relação ao A <------

# Cria o lag de 24 horas, permitindo utilizar o consumo do mesmo horário do dia anterior.
energyData["AEP_t-24"] = energyData["AEP_MW"].shift(24)


# Define a variável alvo do treinamento: a demanda real em MW.
trainingSampleY = energyData.loc[range(beginningOfTraining, endOfTraining)]["AEP_MW"]
# Define as variáveis de entrada utilizadas para treinar o modelo.
trainingSampleX = energyData.loc[range(beginningOfTraining, endOfTraining)][["Year", "Month", "Day", "Hour", "AEP_t-1", "AEP_t-2","AEP_t-24"]]

    

forestmodel = []
# Realiza a previsão para o período definido.
predictionforest = []
    
for i in range(0,10):
# Cria o modelo Random Forest usando os valores da partícula como hiperparâmetros.
    forestmodel.append(RandomForestRegressor(random_state=i*10, max_depth=1423, max_leaf_nodes=45000))
# Treina o modelo utilizando os dados de entrada e os valores reais de demanda.
    forestmodel[i].fit(trainingSampleX,trainingSampleY)
    predictionforest.append(forestmodel[i].predict(energyData.loc[range(beginningOfPrediction, endOfPrediction)][["Year", "Month", "Day", "Hour", "AEP_t-1", "AEP_t-2","AEP_t-24"]])) # Prediction of future values

# Calcula a média das previsões dos modelos para obter uma previsão agregada.
predictionforestmean = np.mean(predictionforest, axis=0)

#---- Plotagem de uma fatia dos resultados obtidos
# Inicia a configuração dos gráficos de comparação entre previsão e valor real.
plt.subplot(2,1,1)
# Plota os valores previstos, reais ou a diferença percentual entre eles.
plt.plot(range(beginningOfPredictionPlot, endOfPredictionPlot), predictionforestmean[beginningOfPredictionPlot-beginningOfPrediction:endOfPredictionPlot-beginningOfPrediction], linestyle="dashed", label="Valor previsto")
# Plota os valores previstos, reais ou a diferença percentual entre eles.
plt.plot(range(beginningOfPredictionPlot,endOfPredictionPlot), energyData.loc[range(beginningOfPredictionPlot, endOfPredictionPlot)]["AEP_MW"], label="Valor real")
plt.title("Comparação entre valores reais de consumo e previsão")
plt.ylabel("MW")
plt.legend()
plt.xticks(ticks=[beginningOfPredictionPlot + 24,beginningOfPredictionPlot + 48,beginningOfPredictionPlot + 72,beginningOfPredictionPlot + 96,beginningOfPredictionPlot + 120,beginningOfPredictionPlot + 144,beginningOfPredictionPlot + 168],labels=["Dia 1","Dia 2","Dia 3","Dia 4","Dia 5","Dia 6","Dia 7"])
plt.grid()
plt.tight_layout()

# Inicia a configuração dos gráficos de comparação entre previsão e valor real.
plt.subplot(2,1,2)
plt.title("Diferença entre valor real e previsto")
# Plota os valores previstos, reais ou a diferença percentual entre eles.
plt.plot(range(beginningOfPredictionPlot, endOfPredictionPlot), (predictionforestmean[beginningOfPredictionPlot-beginningOfPrediction:endOfPredictionPlot-beginningOfPrediction]-energyData.loc[range(beginningOfPredictionPlot, endOfPredictionPlot)]["AEP_MW"])/(energyData.loc[range(beginningOfPredictionPlot, endOfPredictionPlot)]["AEP_MW"]))

# Calcula o erro percentual médio utilizado para avaliar a qualidade da previsão.
mediaDoErroPercentual = [sklearn.metrics.mean_absolute_percentage_error(energyData.loc[range(beginningOfPrediction, endOfPrediction)]["AEP_MW"], predictionforestmean[0:endOfPrediction-beginningOfPrediction])]*(endOfPrediction-beginningOfPrediction)
# Formata o erro médio percentual para exibição.
stringMediaDoErroPercentual = f"Erro Médio Percentual: {mediaDoErroPercentual[0] * 100:.2f}%"
print(stringMediaDoErroPercentual)

# Plota os valores previstos, reais ou a diferença percentual entre eles.
plt.plot(range(beginningOfPredictionPlot, endOfPredictionPlot), mediaDoErroPercentual[0:endOfPredictionPlot-beginningOfPredictionPlot], label=stringMediaDoErroPercentual)
plt.yticks(ticks=[-0.05, 0, 0.05, 0.1],labels=["-5%", "0%", "5%", "10%"])
plt.xticks(ticks=[beginningOfPredictionPlot + 24,beginningOfPredictionPlot + 48,beginningOfPredictionPlot + 72,beginningOfPredictionPlot + 96,beginningOfPredictionPlot + 120,beginningOfPredictionPlot + 144,beginningOfPredictionPlot + 168],labels=["Dia 1","Dia 2","Dia 3","Dia 4","Dia 5","Dia 6","Dia 7"])
plt.grid()
plt.legend()
plt.tight_layout()
# Salva a figura gerada em arquivo PNG.
plt.savefig("comparacaoDePrevisoes_D_treinamento15Anos.png", dpi=300)
# Exibe os gráficos na tela.
plt.show()
#----
