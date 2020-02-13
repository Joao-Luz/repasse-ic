from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

#Carregando nossos dados com formato:
#[[<classe     1>, <pixel 1>, ... , <pixel 784>],
#                       .
#                       .
#                       .
# [<classe 70000>, <pixel 1>, ... , <pixel 784>]]
print('Carregando dados...')
D = np.loadtxt('dados/mnist-fashion-simple.csv', delimiter=',', dtype=int)
print('OK!')
X = D[:, 1:] #Caracteristicas
y = D[:,  0] #Classes
#Nomes das classes
nomes = ['camisa', 'calca', 'sueter', 'vestido', 'casaco', 'sandalia', 'camiseta', 'tenis', 'bolsa', 'bota']


historico = [] #Lista para guardar as performances de cada Fold
melhor_saida = [] #Lista para guardar a melhor saida
melhor_indices = [] #Lista para guardar os melhores indices
melhor_perf = 0

kf = KFold(n_splits=10) #Declarando K-Fold. Faremos 10 permutacoes
print('Realizando treinamento com 10 folds...')
for i, (treino, teste) in enumerate(kf.split(X, y)):

    #Separando os dados utilizados em treino e em teste
    X_treino, X_teste = X[treino], X[teste]
    y_treino, y_teste = y[treino], y[teste]

    #O modelo KNN
    knn = KNeighborsClassifier(n_neighbors=1)

    #Treinando o modelo
    print('\ttreinando modelo...', end='')
    knn.fit(X_treino, y_treino)
    print('\r', end='')

    #Avaliando os dados de teste e comparando com a saida esperada
    print('\tavaliando modelo...', end='')
    y_saida = knn.predict(X_teste)
    performance = accuracy_score(y_teste, y_saida)
    if performance > melhor_perf:
        melhor_saida = y_saida    #melhor saida do modelo
        melhor_perf = performance #melhor performance
        melhor_indices = teste    #indices de teste do melhor caso
    print('\r', end='')

    historico.append(performance)

    print("\tperformance {}: {:.2f}".format(i + 1, performance))

print('Finalizado!')
# Gerando um grafico boxplot do historico do modelo
print('Gerando boxplot...')
plt.title("Historico do modelo")
plt.boxplot(historico)
plt.savefig('saidas/historico.png')
print('OK!')

# Carregando imagens completas
print('Carregando imagens completas...')
X = np.loadtxt('dados/mnist-fashion.csv', delimiter=',', dtype=int)[:, 1:]
print('OK!')

# Mostrando as imagens e suas classificacoes no melhor dos casos do modelo
for (imagem, saida, classe) in zip(X[melhor_indices], melhor_saida, y[melhor_indices]):
    imagem = imagem.reshape((28, 28))
    plt.title('Avaliado: {} - Real: {}'.format(nomes[saida], nomes[classe]))
    plt.imshow(imagem, cmap=plt.cm.binary)
    plt.show()
    plt.clf()



    