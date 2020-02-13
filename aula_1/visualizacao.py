import matplotlib.pyplot as plt
import numpy as np

# Carregando os dados
D = np.loadtxt('mnist-fashion.csv', delimiter=',', dtype=int, max_rows=100)
imagens = D[:, 1:] # Separando as imagens
classes = D[:,  0] # Separando suas respectivas classes
# Nomes das classes
nomes = ['camisa', 'calca', 'sueter', 'vestido', 'casaco', 'sandalia', 'camiseta', 'tenis', 'bolsa', 'bota']

# Para cada imagem, gerar um "plot" com o titulo sendo sua classe
for classe, imagem in zip(classes, imagens):
    imagem = imagem.reshape((28, 28))
    plt.title(nomes[classe])
    plt.imshow(imagem, cmap=plt.cm.binary)
    plt.show()
    plt.clf()

