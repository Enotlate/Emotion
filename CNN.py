import torch
import torch.nn as nn

device = torch.device("cpu" if torch.cuda.is_available() else "cpu")


class CNN_delay(nn.Module):
    def __init__(self):
        super(CNN_delay, self).__init__()
        # Сверточные слои
        self.conv1 = nn.Conv2d(3, 32, kernel_size=6, padding=1, device=device)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=6, padding=1, device=device)

        # Полносвязные слои
        self.fc1 = nn.Linear(10816, 512, device=device)
        self.fc2 = nn.Linear(512, 128, device=device)
        self.fc3 = nn.Linear(128, 7, device=device)

        # Агрегат
        self.pool = nn.MaxPool2d(kernel_size=6, stride=3, padding=0)

        # Функция активации
        self.relu = nn.LeakyReLU().to(device)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # Прямой проход через сверточные слои
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 10816)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)


        return x


def train_model(model, train_loader, criterion, optimizer, num_epochs, grad_value=0.5):
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0

        for inputs, labels in train_loader:
            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_value)
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")


def top_k_accuracy(model, dataloader):
    model.eval()
    deviation = []
    mse = []
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for inputs, target in dataloader:
            predicted = model(inputs)

            # собираем MSE
            loss = criterion(predicted, target)
            mse.append(loss.item())

            # собираем отклонение
            info = torch.mean(abs((target - predicted) / target))
            deviation.append(info)

    # рассчитываем среднее отклонение
    deviation = torch.tensor(deviation)
    deviation = torch.mean(deviation)

    # рассчитываем среднее mse
    mse = sum(mse) / len(mse)
    return deviation, mse
