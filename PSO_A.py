# PSO_A.py - código comentado
# Comentários adicionados para explicar a finalidade de cada bloco; a lógica original foi mantida.

# Bibliotecas utilizadas no processamento dos dados, treinamento dos modelos e geração dos gráficos.
import random
from A import DecisionPrediction
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#Initial parameters

# Define o número de partículas utilizadas pelo algoritmo PSO.
numberOfParticles = 30
# Define o número máximo de iterações do PSO.
maximumNumberOfIterations = 20
# Coeficiente de influência da melhor posição individual da partícula.
c1 = 2
# Coeficiente de influência da melhor posição global encontrada pelo enxame.
c2 = 2
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

#Define initial values of particles
# Inicializa aleatoriamente as posições e velocidades das partículas.
for i in range(numberOfParticles):
    particleVector[i] = [random.randint(minimumValueofDepth, maximumValueOfDepth), random.randint(minimumValueofNodes, maximumValueOfNodes)]

    velocityVector[i] = [0.0,0.0]

    personalBest[i] = particleVector[i]

    globalBest = personalBest[0] #Arbitrarily set to the value of the first particle
globalBestValue = DecisionPrediction([int(globalBest[0]), int(globalBest[1])])


# Executa as iterações do algoritmo de otimização por enxame de partículas.
for j in range(maximumNumberOfIterations):
    f = open("resultadosA.txt", "w")
    string = "\n" + str(["Global best at position ", globalBest,"with a value of ", globalBestValue])
    string = string + "\n" + str(personalBest)
    f.write(string)
    f.close()
    for i in range(0, numberOfParticles):
        
        print("\nIteration number ", j, "\n")
# Avalia o desempenho do modelo na posição atual da partícula.
        currentParticleResult = DecisionPrediction([int(particleVector[i][0]), int(particleVector[i][1])])
        
        print("Particle ",i, " MAPE value for position ", particleVector[i],": ",currentParticleResult)
            # Global best check
# Verifica se a partícula encontrou uma solução melhor que a melhor solução global atual.
        if currentParticleResult < globalBestValue:
            globalBest = [int(particleVector[i][0]), int(particleVector[i][1])]
            globalBestValue = DecisionPrediction([int(globalBest[0]), int(globalBest[1])])
            print("\nThe new value for a global best is: ")
            print(globalBestValue)
            print(" at position ", globalBest, "\n")

            # Local best check
# Verifica se a posição atual é melhor ou igual ao melhor resultado individual da partícula.
        if currentParticleResult <= DecisionPrediction([int(personalBest[i][0]), int(personalBest[i][1])]):
            personalBest[i] = particleVector[i]
            print("Particle ", i, " now has a Local best at position ", personalBest[i])

            # Position update
# Atualiza a velocidade da partícula considerando inércia, melhor posição individual e melhor posição global.
        velocityVectorFuture[0] = w*velocityVector[i][0] + random.random()*c1*(personalBest[i][0] - particleVector[i][0]) + random.random()*c2*(globalBest[0] - particleVector[i][0])
        velocityVectorFuture[1] = w*velocityVector[i][1] + random.random()*c1*(personalBest[i][1] - particleVector[i][1]) + random.random()*c2*(globalBest[1] - particleVector[i][1])
        velocityVector[i] = [velocityVectorFuture[0], velocityVectorFuture[1]]

            # OUT-OF-BOUNDS CHECK
# Atualiza a posição da partícula e impede que o valor fique abaixo do limite mínimo definido.
        if  particleVector[i][0] + velocityVector[i][0] >= 2:
            particleVector[i][0] = particleVector[i][0] + velocityVector[i][0]
        else:
            particleVector[i][0] = 2
            
        if particleVector[i][1] + velocityVector[i][1] >= 2:
            particleVector[i][1] = particleVector[i][1] + velocityVector[i][1]
        else:
            particleVector[i][1] = 2
        print("\n" + str(["Global best at position ", globalBest,"with a value of ", globalBestValue]))
        print("----------------------------")

            # Checks if the program reached a steady state
# Verifica se a velocidade da partícula é suficientemente pequena para considerar que o PSO atingiu estabilidade.
        if velocityVector[i][0] <= 1 and velocityVector[i][0] >= -1 and velocityVector[i][1] >= -1 and velocityVector[i][1] <= 1:
            iterationContinuityCheck[i] = False
        else:
            iterationContinuityCheck[i] = True


        # If the program reached a steady state, ends the iterations
# Encerra o processo caso todas as partículas tenham atingido a condição de estabilidade.
    if not any(iterationContinuityCheck):
        break
