import torch
from torch import nn
from numpy import tile

class CrossEntropyLabelSmooth(nn.Module):

    def __init__(self, num_classes=4, epsilon=0.1, weight=[0.1,0.2,0.3,0.4], use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.weight_temp = weight

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
    
        self.weight = tile(self.weight_temp, (log_probs.size()[0],1))
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / (self.num_classes)
        self.weight = torch.FloatTensor(self.weight)
        targets = targets.cpu()
        log_probs = log_probs.cpu()
        loss = (- targets * log_probs * self.weight).mean(0).sum()
        return loss

# criterion = CrossEntropyLabelSmooth()
# x = torch.rand((4, 4))
# t = torch.tensor([1,1,1,1])
# print(criterion(x,t))