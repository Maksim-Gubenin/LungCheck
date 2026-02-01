import logging
from typing import cast

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet

from app.core.config import settings

logger = logging.getLogger(__name__)


class ModelLoader:
    def __init__(self, model_name: str = "resnet18") -> None:
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def load_model(self) -> ResNet:
        logger.info("Initializing %s architecture...", self.model_name)

        model: ResNet = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)

        weights_path = settings.PROJECT_ROOT / settings.ml_config.model_path

        if weights_path.exists():
            logger.info("Loading trained weights from %s", weights_path)
            model.load_state_dict(torch.load(weights_path, map_location=self.device))
        else:
            logger.warning("Trained weights NOT FOUND at %s", weights_path)

        model.to(self.device)
        model.eval()

        self.model = model

        logger.info("Model loaded successfully on %s", self.device)
        return model


model_loader = ModelLoader()
