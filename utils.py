import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import errno
from torchvision import transforms, datasets


def copy_dir(org_dirpath, dest_dirpath, class_names):
    for c in class_names:
        try:
            os.makedirs(os.path.join(dest_dirpath, c))
        except OSError as e:
            if e.errno == errno.EEXIST:
                # override folders
                os.system(f'rm -d -r {os.path.join(dest_dirpath, c)}')
                # create new one
                os.makedirs(os.path.join(dest_dirpath, c))
            else:
                raise e

    # copy train images to train_aug folder
    for c in tqdm(os.listdir(org_dirpath)):
        names = os.listdir(f'{org_dirpath}/{c}')
        for name in names:
            os.system(f'cp {os.path.join(org_dirpath, c, name)} {os.path.join(dest_dirpath, c, name)}')


def imshow2(imgs, titles):
    # taken from: https://pytorch.org/vision/stable/auto_examples/plot_scripted_tensor_transforms.html#sphx-glr-auto-examples-plot-scripted-tensor-transforms-py
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = transforms.ToPILImage()(img.to('cpu'))
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        axs[0, i].set_title(titles[i])


def imshow(inp, title=None):
    """Imshow for Tensors."""
    inp = inp.numpy().transpose((1, 2, 0))
    plt.figure(figsize=(15, 15))
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


class ImageFolderWithPaths(datasets.ImageFolder):
    # taken from: https://gist.github.com/andrewjong/6b02ff237533b3b2c554701fb53d5c4d
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """
    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path
