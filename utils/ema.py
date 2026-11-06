import copy
import torch
import torch.nn as nn

class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self._ema_model = copy.deepcopy(model)
        self._ema_model.eval()
        for param in self._ema_model.parameters():
            param.requires_grad = False

    @property
    def ema_model(self) -> nn.Module:
        return self._ema_model

    @torch.no_grad()
    def update(self, model: nn.Module):
        # Update the exponential moving average of parameters
        for ema_param, param in zip(self._ema_model.parameters(), model.parameters()):
            ema_param.copy_(self.decay * ema_param + (1.0 - self.decay) * param)
        
        # Copy buffers directly
        for ema_buffer, buffer in zip(self._ema_model.buffers(), model.buffers()):
            ema_buffer.copy_(buffer)

    def state_dict(self) -> dict:
        return self._ema_model.state_dict()

    def load_state_dict(self, state_dict: dict):
        self._ema_model.load_state_dict(state_dict)
