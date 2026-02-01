import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

DATA_DIR = "./data"
MODEL_SAVE_PATH = "./models/pneumonia_resnet18.pth"
BATCH_SIZE = 32
EPOCHS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model() -> None:
    # Подготовка данных (Трансформации)
    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # Загрузчик
    train_dataset = datasets.ImageFolder(
        os.path.join(DATA_DIR, "train"), train_transform
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    print(f"Dataset loaded: {len(train_dataset)} images.")
    print(
        f"Classes found: {train_dataset.classes}"
    )  # Должно быть ['NORMAL', 'PNEUMONIA']

    # Инициализация модели (ResNet18)
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(DEVICE)

    # Функция потерь и Оптимизатор
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Цикл обучения
    os.makedirs("./models", exist_ok=True)
    print(f"Starting training on {DEVICE}...")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch + 1}/{EPOCHS}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}"
                )

        print(
            f"Epoch {epoch + 1} finished. Avg Loss: {running_loss / len(train_loader):.4f}"
        )

    # Сохранение
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Success! Model saved to {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train_model()
