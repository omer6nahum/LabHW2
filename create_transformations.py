import os
from utils import imshow, copy_dir
import torch
import numpy as np
import torchvision
from torchvision import transforms, datasets
from torchvision.utils import save_image


INDEX = 0


def save_images(inputs, classes, dir_path):
    global INDEX
    for image, label in zip(inputs, classes):
        path = os.path.join(dir_path, str(label), f'aug_{INDEX}.png')
        save_image(image, path)
        INDEX += 1


class AddGaussianNoise(object):
    # based on https://discuss.pytorch.org/t/how-to-add-noise-to-mnist-dataset-when-using-pytorch/59745
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        noise = torch.randn(tensor.size()[2:])  # noise in grayscale
        return tensor + noise * self.std + self.mean


class AddMaskedGaussianNoise(object):
    # based on https://discuss.pytorch.org/t/how-to-add-noise-to-mnist-dataset-when-using-pytorch/59745
    def __init__(self, mean=0., std=1., p=0.1):
        self.std = std
        self.mean = mean
        self.p = p

    def __call__(self, tensor):
        noise = torch.randn(tensor.size()[2:])  # noise in grayscale
        mask = torch.zeros_like(tensor)
        indices = torch.randint(low=0, high=tensor.shape[2], size=[2, int(self.p * tensor.shape[2] * tensor.shape[3])])
        mask[:, :, indices[0], indices[1]] = 1
        return tensor + (noise * self.std + self.mean) * mask


def main():
    flag_plot = False
    flag_augment = not flag_plot
    flag_write = not flag_plot

    BATCH_SIZE = 8

    # Paths to your train and val directories
    train_dir = os.path.join("data", "our_filtered_train")
    train_aug_dir = os.path.join("data", "our_train_aug")

    data_transforms = transforms.Compose([transforms.Resize([64, 64]),
                                          transforms.ToTensor()])
    train_dataset = datasets.ImageFolder(train_dir, data_transforms)
    class_names = train_dataset.classes
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    if flag_write:
        copy_dir(train_dir, train_aug_dir, class_names)

    optional_add_transformation = {
        'blur': transforms.GaussianBlur(kernel_size=15),
        'perspective': transforms.RandomPerspective(p=1, fill=1, interpolation=transforms.InterpolationMode.NEAREST),
        'rotation+': transforms.RandomRotation(degrees=(15, 45), fill=1),
        'rotation-': transforms.RandomRotation(degrees=(-45, -15), fill=1),
        'affine': transforms.RandomAffine(degrees=(-45, 45), scale=(0.3, 1.3), fill=1),
        'autocontrast': transforms.RandomAutocontrast(p=1),
        'gaussian_noise': AddGaussianNoise(std=0.2),  # was .4
        # # 'gaussian_noise2': AddGaussianNoise(std=0.4),  # was .4
        'brighten': AddGaussianNoise(std=0, mean=0.4),
        # # 'darken': AddGaussianNoise(std=0, mean=-0.4),
        'masked_noise': AddMaskedGaussianNoise(std=0.2, p=0.1),
        'crop': transforms.RandomResizedCrop(size=(64, 64), ratio=(0.8, 1), scale=(0.5, 1))
                     }

    max_observations = 10_000
    k = int((max_observations - 2067) / (BATCH_SIZE * len(optional_add_transformation)))
    print(f'k={k}')

    # Plot the transformations on random batches
    if flag_plot:
        for name, add_transform in optional_add_transformation.items():
            # sample batch from loader
            inputs, classes = next(iter(train_dataloader))
            # plot original batch
            out = torchvision.utils.make_grid(inputs)
            imshow(out, title=[f'{name}_{class_names[x]}' for x in classes])
            # transform batch according to add_transform
            inputs = add_transform(inputs)
            # plot transformed batch
            out = torchvision.utils.make_grid(inputs)
            imshow(out, title=[f'{name}_{class_names[x]}_transformed' for x in classes])

    # create transformations on batches and save to 'data/our_train_aug'
    if flag_augment:
        for name, add_transform in optional_add_transformation.items():
            for i in range(k):
                inputs, classes = next(iter(train_dataloader))
                inputs = add_transform(inputs)
                save_images(inputs, np.array(class_names)[classes.numpy()], train_aug_dir)

        print(INDEX)
        print(INDEX + 2067)


if __name__ == '__main__':
    main()
