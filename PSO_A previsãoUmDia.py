# PSO_A previsãoUmDia.py - código comentado
# Comentários adicionados para explicar a finalidade de cada bloco; a lógica original foi mantida.

# Bibliotecas utilizadas no processamento dos dados, treinamento dos modelos e geração dos gráficos.
import random
from A_previsãoUmDia import DecisionPrediction
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#---- Initial parameters
# Define o número de partículas utilizadas pelo algoritmo PSO.
numberOfParticles = 5
# Define o número máximo de iterações do PSO.
maximumNumberOfIterations = 20
# Coeficiente de influência da melhor posição individual da partícula.
c1 = 1.5
# Coeficiente de influência da melhor posição global encontrada pelo enxame.
c2 = 1.5
# Coeficiente de inércia utilizado no cálculo da velocidade.
w = 0.7

# Define os limites inferior e superior para o hiperparâmetro max_depth.
minimumValueofDepth = 2
# Define os limites inferior e superior para o hiperparâmetro max_leaf_nodes.
minimumValueofNodes = 2
maximumValueOfDepth = 1000
maximumValueOfNodes = 1000

# Armazena a posição de cada partícula, representando uma combinação de hiperparâmetros.
particleVector = [None] * numberOfParticles
# Armazena a velocidade de cada partícula no espaço de busca.
velocityVector = [None] * numberOfParticles
velocityVectorFuture = [None, None]
# Armazena a melhor posição já encontrada por cada partícula.
personalBest = [None] * numberOfParticles
# Indica se cada partícula ainda possui movimento suficiente para continuar as iterações.
iterationContinuityCheck = [None] * numberOfParticles

# Vetores auxiliares para armazenar dados usados no acompanhamento dos resultados do PSO.
XGlobalBest = []
X = []
Y = []
Z = []
# Vetores auxiliares para armazenar dados usados no acompanhamento dos resultados do PSO.
T = []
#----

#---- Definição inicial de valores
# Inicializa aleatoriamente as posições e velocidades das partículas.
for i in range(numberOfParticles):
    particleVector[i] = [random.randint(minimumValueofDepth, maximumValueOfDepth), random.randint(minimumValueofNodes, maximumValueOfNodes)]

    velocityVector[i] = [0.0,0.0]

    personalBest[i] = particleVector[i]

    globalBest = personalBest[0] #Arbitrarily set to the value of the first particle
globalBestValue = DecisionPrediction([int(globalBest[0]), int(globalBest[1])])
#----

#---- Loops de iterações do PSO
# Executa as iterações do algoritmo de otimização por enxame de partículas.
for j in range(maximumNumberOfIterations):
    f = open("resultadosA_UmDia.txt", "w")
    for i in range(0, numberOfParticles):
        
        print("\nIteration number ", j, "\n")
# Avalia o desempenho do modelo na posição atual da partícula.
        currentParticleResult = DecisionPrediction([int(particleVector[i][0]), int(particleVector[i][1])])
        
        print("Particle ",i, " MAE value for position ", particleVector[i],": ",currentParticleResult)
        #--- Global best check
# Verifica se a partícula encontrou uma solução melhor que a melhor solução global atual.
        if currentParticleResult < globalBestValue:
            globalBest = [int(particleVector[i][0]), int(particleVector[i][1])]
            globalBestValue = DecisionPrediction([int(globalBest[0]), int(globalBest[1])])
            print("\nThe new value for a global best is: ")
            print(globalBestValue)
            print(" at position ", globalBest, "\n")
        #---

        #--- Local best check
# Verifica se a posição atual é melhor ou igual ao melhor resultado individual da partícula.
        if currentParticleResult <= DecisionPrediction([int(personalBest[i][0]), int(personalBest[i][1])]):
            personalBest[i] = particleVector[i]
            print("Particle ", i, " now has a Local best at position ", personalBest[i])
        #---

        #--- Position update
# Atualiza a velocidade da partícula considerando inércia, melhor posição individual e melhor posição global.
        velocityVectorFuture[0] = w*velocityVector[i][0] + random.random()*c1*(personalBest[i][0] - particleVector[i][0]) + random.random()*c2*(globalBest[0] - particleVector[i][0])
        velocityVectorFuture[1] = w*velocityVector[i][1] + random.random()*c1*(personalBest[i][1] - particleVector[i][1]) + random.random()*c2*(globalBest[1] - particleVector[i][1])
        velocityVector[i] = [velocityVectorFuture[0], velocityVectorFuture[1]]
        #---

        #--- OUT-OF-BOUNDS CHECK
# Atualiza a posição da partícula e impede que o valor fique abaixo do limite mínimo definido.
        if  particleVector[i][0] + velocityVector[i][0] >= 2:
            particleVector[i][0] = particleVector[i][0] + velocityVector[i][0]
        else:
            particleVector[i][0] = 2
            print("Particle went out-of-bounds")
            
        if particleVector[i][1] + velocityVector[i][1] >= 2:
            particleVector[i][1] = particleVector[i][1] + velocityVector[i][1]
        else:
            particleVector[i][1] = 2
            print("Particle went out-of-bounds")
        #---

        print("\n" + str(["Global best at position ", globalBest,"with a value of ", globalBestValue]))
        print("----------------------------")

        #--- Checks if the program reached a steady state
# Verifica se a velocidade da partícula é suficientemente pequena para considerar que o PSO atingiu estabilidade.
        if velocityVector[i][0] <= 1 and velocityVector[i][0] >= -1 and velocityVector[i][1] >= -1 and velocityVector[i][1] <= 1:
            iterationContinuityCheck[i] = False
        else:
            iterationContinuityCheck[i] = True
        #---
        
        X.append(particleVector[i][0])
        Y.append(particleVector[i][1])
        Z.append(currentParticleResult)
    


        # If the program reached a steady state, ends the iterations
# Encerra o processo caso todas as partículas tenham atingido a condição de estabilidade.
    if not any(iterationContinuityCheck):
        break
# Vetores auxiliares para armazenar dados usados no acompanhamento dos resultados do PSO.
    XGlobalBest.append(globalBestValue)
    T.append(j)
#----

#plt.scatter(T, XGlobalBest)
#plt.xlabel("Iteração")
#plt.ylabel("MAE")
#---- Gráfico para mostrar quais as posições tomadas pelas partículas durante as iterações     
#fig = plt.figure(figsize=(10,7))
#ax = fig.add_subplot(111, projection='3d')

# X, Y, Z, T já convertidos para numpy como acima
#sc = ax.scatter(X, Y, Z, c=T, cmap="plasma", s=8)  # s tamanho dos marcadores

#ax.set_xlabel("Max depth")
#ax.set_ylabel("Max leaf nodes")
#ax.set_zlabel("MAE")

#cbar = plt.colorbar(sc, pad=0.1)
#cbar.set_label("Iteration")

#plt.title("PSO evaluations (depth, nodes, MAE)")
# Exibe os gráficos na tela.
plt.show()
#----


string = "\n" + str(["Global best at position ", globalBest,"with a value of ", globalBestValue])
string = string + "\n" + str(personalBest)
f.write(string)
f.close()
