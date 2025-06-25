from torch.utils.data import Dataset
from cnn_data_loader import PictureDatasetLoader
from CNN import *
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Подготовка данных
train_dataset = PictureDatasetLoader("train", transform)
test_dataset = PictureDatasetLoader("test", transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=512, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=512, shuffle=True)

device = torch.device("cpu" if torch.cuda.is_available() else "cpu")


#
# model = CNN_delay().to(device)
# # model.load_state_dict(torch.load('weight_serv.pth', map_location=device))
#
# criterion = nn.CrossEntropyLoss().to(device)
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# num_epochs = 12
#
# if __name__ == "__main__":
#     s = time()
#     train_model(model, train_loader, criterion, optimizer, num_epochs)
#     print(time() - s)
#
# torch.save(model.state_dict(), "weight_12pc.pth")


model = CNN_delay().to(device)
model.load_state_dict(torch.load('weight_10.pth', map_location=device))
a = top_k_accuracy(model, test_loader)
print(a)