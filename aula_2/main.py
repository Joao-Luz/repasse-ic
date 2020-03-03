import torch
from datasets import MnistFashionDataset
from classificador_linear import ClassificadorLinear
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def main():
    lr = 0.005
    batch_size = 1024
    regularization_weight = 0
    epochs = 10

    class_names = ["T-Shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    print('Gerando o Dataset...')
    mnist = MnistFashionDataset('dados/mnist-fashion-train.csv')
    print("OK!")
    print('Gerando DataLoader...')
    trainloader = torch.utils.data.DataLoader(mnist, batch_size=batch_size)
    print('Ok!')

    mnist = MnistFashionDataset('dados/mnist-fashion-test.csv')
    testloader = torch.utils.data.DataLoader(mnist, batch_size=batch_size)

    classificador_linear = ClassificadorLinear().cuda()
    funcao_perda = torch.nn.MSELoss()
    otimizador = torch.optim.SGD(classificador_linear.parameters(), lr=lr, weight_decay=regularization_weight)

    i = 0
    score = 0

    #avaliando o classificador nao treinado
    for dado in testloader:
        inputs, targets = dado
        inputs = inputs.cuda()
        _, targets = torch.max(targets, 1)

        outputs = classificador_linear(inputs)
        _, predicted = torch.max(outputs, 1)
        score += accuracy_score(targets.cpu(), predicted.cpu())
        i += 1
    print('Performance Inicial: {:.10f}\n'.format(score/i))

    print('Iniciando treino...')

    #treinando
    for epoca in range(epochs):
        print('\tEpoca {}...'.format(epoca + 1))
        for dado in trainloader:
            inputs, targets = dado
            inputs = inputs.cuda()
            targets = targets.cuda()

            otimizador.zero_grad()
            outputs = classificador_linear(inputs)
            perda = funcao_perda(outputs, targets)
            perda.backward()
            otimizador.step()
    
    i = 0
    score = 0

    #testando
    for dado in testloader:
        inputs, targets = dado
        inputs = inputs.cuda()
        _, targets = torch.max(targets, 1)

        outputs = classificador_linear(inputs)
        _, predicted = torch.max(outputs, 1)
        score += accuracy_score(targets.cpu(), predicted.cpu())
        i += 1
    print('\tPerformance: {:.10f}\n'.format(score/i))

    #gerando as imagens
    fig = plt.figure()
    ax = []
    for i, template in enumerate(classificador_linear.linear.weight):
        ax.append(fig.add_subplot(2, 5, i + 1))
        ax[-1].set_title(class_names[i])

        plt.setp(ax[-1].get_xticklabels(), visible=False)
        plt.setp(ax[-1].get_yticklabels(), visible=False)
        ax[-1].tick_params(axis='both', which='both', length=0)

        template = template.cpu().detach().numpy()
        template = template.reshape((28, 28))
        plt.imshow(template, cmap=plt.cm.binary)
    plt.savefig('templates/templates.png')


if __name__ == "__main__":
    main()

