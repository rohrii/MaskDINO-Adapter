import torch.nn as nn
from transformers.adapters import AdapterConfig
from transformers.adapters.modeling import Adapter

class MaskdinoAdapter(nn.Module):
    def __init__(self, d_model, reduction=4, num_adapters=1):
        super().__init__()

        down_sample = d_model // reduction
        
        self.adapters = nn.ModuleList([
            Adapter(
                adapter_name="pfeiffer",
                config=AdapterConfig.load("pfeiffer"),
                input_size=d_model,
                down_sample=down_sample
            ) for _ in range(num_adapters)
        ])

    def forward(self, x, residual_input):
        for adapter in self.adapters:
            x, *_ = adapter(x, residual_input=residual_input)
        
        return x
