
import torch.nn as nn
from transformers import ResNetForImageClassification

class PretrainedResNet50(nn.Module):

    def __init__(self, num_classes=21, pretrained_model_name="microsoft/resnet-50"):
        super().__init__()
        self.model = ResNetForImageClassification.from_pretrained(
            pretrained_model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )

    def forward(self, x):
        outputs = self.model(x)
        return outputs.logits