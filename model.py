import torch
import torch.nn as nn


class Model(nn.Module):

    def forward(self, batch):
        raise NotImplementedError

    def forward_loss(self, batch):
        raise NotImplementedError


class LinearProbe(Model):
    def __init__(self, input_dim, output_dim):
        super(LinearProbe, self).__init__()
        self.normalize = nn.LayerNorm(input_dim)
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, batch):
        normalized_inputs = self.normalize(batch["inputs"])
        return self.linear(normalized_inputs)

    def forward_loss(self, batch):
        outputs = self.forward(batch)
        loss = nn.functional.cross_entropy(outputs, batch["labels"])
        return loss


class MLPProbe(Model):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPProbe, self).__init__()
        self.normalize = nn.LayerNorm(input_dim)
        self.linear1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.act = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, batch):
        normalized_inputs = self.normalize(batch["inputs"])
        x = self.linear1(normalized_inputs)
        x = self.act(x)
        return self.linear2(x)

    def forward_loss(self, batch):
        outputs = self.forward(batch)
        loss = nn.functional.cross_entropy(outputs, batch["labels"])
        return loss
