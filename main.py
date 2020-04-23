import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from model import Autoencoder50, Autoencoder25
import matplotlib.pyplot as plt

N_EPOCHS = 15
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print("1. Model with 50%% compression.\n")

    classify(50)

    print("2. Model with 25%% compression.\n")
    
    classify(25)

def classify(modelType):

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # Normalize the test set same as training set without augmentation
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)                                    
    trainset, validset = torch.utils.data.random_split(trainset, [45000, 5000])

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                                shuffle=True, num_workers=4)
    validloader = torch.utils.data.DataLoader(validset, batch_size=64,
                                                shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                            shuffle=False, num_workers=4)

    if modelType == 50:
        model = Autoencoder50()
    else:
        model = Autoencoder25()

    #Adam optimizer and MSE Loss
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    #Transfer model and criterion to GPU
    model = model.to(device)
    criterion = criterion.to(device)

    train_loss_list = []
    valid_loss_list = []
    best_valid_loss = float('inf')
    PATH = './autoencoder.pth'

    for epoch in range(N_EPOCHS):  # loop over the dataset multiple times

        print("EPOCH: %d" % (epoch+1))
        train_loss = train(model, trainloader, optimizer, criterion)
        valid_loss = evaluate(model, validloader, criterion)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), PATH)

        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)

    #Plot loss and accuracy
    plotLoss(train_loss_list, valid_loss_list)

    if modelType == 50:
        model = Autoencoder50()
    else:
        model = Autoencoder25()
    model.load_state_dict(torch.load(PATH))

    #Transfer model to GPU
    model = model.to(device)

    test_loss = evaluate(model, testloader, criterion, True)
    print('Loss of the network on test images: %f ' % (test_loss))

#Function to train the model
def train(model, dataloader, optimizer, criterion):
    epoch_loss = 0
    running_loss = 0.0
    model.train()

    for i, data in enumerate(dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, _ = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        # print statistics
        running_loss += loss.item()
        if i % 200 == 199:    # print every 2000 mini-batches
            print('loss: %.3f ' % (running_loss / 200))
            running_loss = 0.0
        
    return epoch_loss / len(dataloader)


#Forward pass for classification
def evaluate(model, dataloader, criterion, plotImage=False):
    epoch_loss = 0
    model.eval()
    
    with torch.no_grad():
        for i, data in enumerate(dataloader, 0):
            images, _ = data[0].to(device), data[1].to(device)
            outputs = model(images)
            loss = criterion(outputs, images)
            epoch_loss += loss.item()
            if plotImage and i%20==0:
                inputImage = images[0]
                outputImage = outputs[0]
                plotImages(inputImage, outputImage)
    return epoch_loss / len(dataloader)

#Plot images
def plotImages(inputImage, outputImage):
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).cpu()
    std = torch.tensor([0.2023, 0.1994, 0.2010]).cpu()

    fig, ax = plt.subplots(1, 2)
    
    inputImage = inputImage.permute(1, 2, 0).cpu()
    inputImage = std * inputImage + mean

    outputImage = outputImage.permute(1, 2, 0).cpu()
    outputImage = std * outputImage + mean

    ax[0].imshow(inputImage)
    ax[1].imshow(outputImage)
    plt.show()

#Plot training and validation loss
def plotLoss(train_loss, valid_loss):
    epochs = range(N_EPOCHS)
    plt.plot(epochs, train_loss, 'g', label='Training loss')
    plt.plot(epochs, valid_loss, 'b', label='validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
