import os
import numpy as np

data_dir = os.path.join('data', 'our_train_aug')

sizes = {}
for c in os.listdir(data_dir):
    sizes[c] = len(os.listdir(os.path.join(data_dir, c)))

min_size = min(min(sizes.values()), int((10_000-619)/10))

for c in os.listdir(data_dir):
    n_remove = sizes[c] - min_size
    filenames = np.random.choice(os.listdir(os.path.join(data_dir, c)), size=n_remove, replace=False)
    for filename in filenames:
        os.system(f'rm {os.path.join(data_dir, c, filename)}')



