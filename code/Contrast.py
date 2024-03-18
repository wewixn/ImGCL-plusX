import torch
from GCL.losses import Loss


class WithinEmbedContrast(torch.nn.Module):
    def __init__(self, loss: Loss, **kwargs):
        super(WithinEmbedContrast, self).__init__()
        self.loss = loss
        self.kwargs = kwargs

    def forward(self, h1, h2, labels=None):
        l1 = self.loss(anchor=h1, sample=h2, **self.kwargs)
        l2 = self.loss(anchor=h2, sample=h1, **self.kwargs)
        if labels is not None:
            class_weights = self.compute_class_weights(labels)
            l1 = torch.sum(l1 * class_weights)
            l2 = torch.sum(l2 * class_weights)
        return (l1 + l2) * 0.5

    def compute_class_weights(self, labels):
        class_counts = torch.bincount(labels)
        class_weights = 1 / (class_counts.float() + 1e-5)
        class_weights /= torch.sum(class_weights)
        return class_weights
