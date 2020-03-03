import torch

class ClassificadorLinear(torch.nn.Module):
    """Classe que representa um classificador linear simples
       Tem a seguinte forma:\n
       `FC(784, 10)` -> `softmax` -> `out`
    """
    def __init__(self):
        super(ClassificadorLinear, self).__init__()
        self.linear = torch.nn.Linear(784, 10)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, input):
        output = self.linear(input)
        output = self.softmax(output)
        return output