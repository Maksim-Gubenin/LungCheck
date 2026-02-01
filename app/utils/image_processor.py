import io

import torch
from PIL import Image
from torchvision import transforms


class ImageProcessor:
    """
    Utility class for medical image preprocessing.

    This class handles the conversion of raw image bytes into a format
    suitable for deep learning inference, ensuring consistency between
    training and production data.
    """
    def __init__(self, img_size: int = 224) -> None:
        """
        Initialize the preprocessing pipeline with standard ResNet transformations.

        Args:
            img_size: Target dimension for image resizing (default: 224).
        """
        self.transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),  # К стандарту размера
                transforms.ToTensor(),  # пиксели в массив чисел (тензор, яркость /255).
                transforms.Normalize(  # Стандарт для ResNet
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def process_image(self, file_content: bytes) -> torch.Tensor:
        """
        Convert raw image bytes into a 4D input tensor.

        The process includes:
        1. Decoding bytes and forcing RGB mode (handles grayscale X-rays).
        2. Applying Resize, Tensor conversion, and Normalization.
        3. Adding a batch dimension (NCHW format) required by PyTorch.

        Args:
            file_content: Raw bytes from the uploaded file.

        Returns:
            torch.Tensor: Preprocessed tensor of shape [1, 3, 224, 224].
        """
        # 1. Перевод ч/б рентгена в RGB
        image = Image.open(io.BytesIO(file_content)).convert("RGB")

        # 2. Применяем трансформации
        tensor: torch.Tensor = self.transform(image)

        # 3. Добавляем размерность батча: из [3, 224, 224] делаем [1, 3, 224, 224]
        return tensor.unsqueeze(0)


# Создаем экземпляр для использования в роутах
image_processor = ImageProcessor()
