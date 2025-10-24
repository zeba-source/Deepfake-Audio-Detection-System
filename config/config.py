from dataclasses import dataclass, field
from typing import Optional, Dict
import torch


@dataclass
class SchedulerConfig:
    name: str = 'ReduceLROnPlateau'
    patience: int = 5
    factor: float = 0.5


@dataclass
class TrainingConfig:
    model_name: str = 'resnet18_cqcc'
    num_epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    optimizer: str = 'Adam'
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    early_stopping_patience: int = 10
    loss_function: str = 'CrossEntropyLoss'
    class_weights: Optional[Dict[int, float]] = None
    device: str = field(default_factory=lambda: 'cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_dir: str = 'models/checkpoints'
    log_dir: str = 'logs'

    def to_dict(self):
        d = self.__dict__.copy()
        d['scheduler'] = d['scheduler'].__dict__ if hasattr(d['scheduler'], '__dict__') else d['scheduler']
        return d
