import os
from our_run_train_eval import create_train_model
from torchvision.io import read_image, ImageReadMode
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
import numpy as np
from utils import copy_dir, imshow2
from scipy.stats import entropy


def main():
    flag_write = True
    flag_plot = False

    sizes = {}
    for c in os.listdir(os.path.join("data", "our_train")):
        sizes[c] = len(os.listdir(os.path.join("data", "our_train", c)))
    print(f'original sizes: {sizes}')

    # train a model based on our_train
    y_train, y_proba, paths, class_names = create_train_model(train_dir=os.path.join("data", "our_train"),
                                                              val_dir=os.path.join("data", "our_val"))
    d = y_proba.shape[1]
    counter = 0
    bad_indices = []

    for i, (y, y_pred, path) in enumerate(zip(y_train, y_proba, paths)):
        filtering1 = y_pred[y] < 2 / d  # wrong label (probably)
        filtering2 = entropy(y_pred) > np.log(d) / 2  # high uncertainty (probably)
        if filtering2 or filtering1:
            counter += 1
            bad_indices.append(i)

    print(f'First filtering: {counter} images')

    if flag_plot:
        # visualize "wrong" images
        data_transforms = transforms.Compose([transforms.Resize([64, 64]),
                                              transforms.ToTensor()])
        images = [data_transforms(transforms.ToPILImage()(read_image(paths[i], ImageReadMode.RGB))) for i in
                  bad_indices]
        images = torch.stack(images)
        labels = [f'{class_names[y_train[i]]}({class_names[np.argmax(y_proba[i])]})' for i in bad_indices]

        for i in range(8, min(50, len(images) - 8), 8):
            imshow2(images[i: i+8], labels[i: i+8])

        plt.show()

    if flag_write:
        class_names = os.listdir(os.path.join("data", "our_filtered_train"))
        # remove images from train (create a new folder named our_filtered_train)
        org_dirpath = os.path.join("data", "our_train")
        dest_dirpath = os.path.join("data", "our_filtered_train")
        copy_dir(org_dirpath, dest_dirpath, class_names)

        for i in bad_indices:
            os.system(f'rm {paths[i].replace(org_dirpath, dest_dirpath)}')

    sizes = {}
    for c in os.listdir(os.path.join("data", "our_filtered_train")):
        sizes[c] = len(os.listdir(os.path.join("data", "our_filtered_train", c)))
    print(f'sizes after filtering: {sizes}')


if __name__ == '__main__':
    main()
