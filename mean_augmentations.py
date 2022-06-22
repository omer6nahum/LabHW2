import numpy as np
import os
from torchvision import transforms, datasets
import torch
from torchvision.utils import save_image


def np_save_image(image, label, dir_path, name):
    path = os.path.join(dir_path, str(label), f'mean_aug_{name}.png')
    save_image(image, path)


def main():
    train_dir = os.path.join("data", "our_filtered_train")
    dest_dir = os.path.join("data", "our_filtered_train")

    BATCH_SIZE = 16
    data_transforms = transforms.Compose([transforms.Resize([64, 64]), transforms.ToTensor()])
    train_dataset = datasets.ImageFolder(train_dir, data_transforms)
    class_names = train_dataset.classes
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    n_labels = len(class_names)
    X = {l: [] for l in range(n_labels)}
    for imgs, labels in train_dataloader:
        for img, label in zip(imgs, labels):
            X[label.item()].append(img.numpy())

    name = 0
    for label in range(n_labels):
        corr = np.corrcoef([x.flatten() for x in X[label]])
        n = corr.shape[0]
        k = 50
        top_k = sorted([(i, j) for i in range(n) for j in range(n) if j > i],
                       key=lambda t: corr[t[0], t[1]], reverse=True)[:k]
        for i, j in top_k:
            x_new = 0.5 * (X[label][i] + X[label][j])
            image = torch.Tensor(x_new)
            np_save_image(image, class_names[label], dest_dir, name)
            name += 1


if __name__ == '__main__':
    main()
