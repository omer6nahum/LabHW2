import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models, transforms
import time
import copy
import numpy as np
from tqdm import tqdm
from utils import ImageFolderWithPaths

torch.manual_seed(0)


def create_train_model(train_dir, val_dir):
    # Training hyperparameters
    BATCH_SIZE = 16
    NUM_EPOCHS = 7
    LR = 0.001

    # Resize the samples and transform them into tensors
    data_transforms = transforms.Compose([transforms.Resize([64, 64]), transforms.ToTensor()])

    # Create a pytorch dataset from a directory of images
    train_dataset = ImageFolderWithPaths(train_dir, data_transforms)
    val_dataset = ImageFolderWithPaths(val_dir, data_transforms)

    class_names = train_dataset.classes

    # Dataloaders initialization
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'device is {device}')
    dataloaders = {'train': train_dataloader, 'val': val_dataloader}
    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}

    NUM_CLASSES = len(class_names)

    def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=30):
        """Responsible for running the training and validation phases for the requested model."""
        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        loss_dict = {'train': [], 'val': []}
        acc_dict = {'train': [], 'val': []}

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch + 1, num_epochs))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels, _ in tqdm(dataloaders[phase]):
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                acc_dict[phase].append(epoch_acc.item())
                loss_dict[phase].append(epoch_loss)

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                   phase, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

            print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        model.load_state_dict(best_model_wts)

        return model, loss_dict, acc_dict

    # Use a prebuilt pytorch's ResNet50 model
    model_ft = models.resnet50(pretrained=False)

    # Fit the last layer for our specific task
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    model_ft = model_ft.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=LR)

    # Decay LR by a factor of 0.1 every 5 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)

    # Train the model
    model_ft, loss_dict, acc_dict = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, dataloaders, dataset_sizes, num_epochs=NUM_EPOCHS)

    # Return the prediction of the model on the train set
    y = []
    y_proba = []
    paths = []
    model_ft.eval()
    for X_batch, y_batch, path_batch in train_dataloader:
        X_batch = X_batch.to(device)
        y.append(y_batch.cpu().numpy())
        paths.append(path_batch)
        with torch.no_grad():
            y_proba.append(torch.nn.functional.softmax(model_ft(X_batch), dim=1).cpu().numpy())

    return np.hstack(y), np.vstack(y_proba), np.hstack(paths), class_names
