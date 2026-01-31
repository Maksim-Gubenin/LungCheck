import logging
from typing import cast

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet

logger = logging.getLogger(__name__)


class ModelLoader:
    def __init__(self, model_name: str = "resnet18") -> None:
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def load_model(self) -> ResNet:
        """Инициализация архитектуры и загрузка весов."""
        logger.info("Initializing %s architecture...", self.model_name)

        # 1. Берем ResNet18
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # 2. Заменяем голову
        num_ftrs = model.fc.in_features
        # На выходе нам нужно 2 класса: [NORMAL, PNEUMONIA]
        model.fc = nn.Linear(num_ftrs, 2)

        # 3. Переносим на CPU и переводим в режим предсказания
        self.model = model.to(self.device)
        model_instance = cast(ResNet, self.model)
        self.model.eval()

        logger.info("Model loaded successfully on %s", self.device)
        return self.model


model_loader = ModelLoader()
