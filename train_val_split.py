import os
import numpy as np
from tqdm import tqdm
import errno


def main():
    org_dir = 'data/full_data'
    out_dir_train = 'data/our_train'
    out_dir_val = 'data/our_val'

    p = 0.1

    for c in tqdm(os.listdir(org_dir)):
        for out_dir in [out_dir_train, out_dir_val]:
            try:
                os.makedirs(os.path.join(out_dir, c))
            except OSError as e:
                if e.errno == errno.EEXIST:
                    # override folders
                    os.system(f'rm -d -r {os.path.join(out_dir, c)}')
                    # create new one
                    os.makedirs(os.path.join(out_dir, c))
                else:
                    raise e

        names = os.listdir(f'{org_dir}/{c}')
        n = len(names)
        n_val = int(n * p)
        n_train = n - n_val

        val_indices = np.random.choice(range(n), size=n_val, replace=False)
        train_indices = [i for i in range(n) if i not in val_indices]
        assert (len(val_indices) + len(train_indices) == n)

        for i in train_indices:
            os.system(f'cp {os.path.join(org_dir, c, names[i])} {os.path.join(out_dir_train, c, names[i])}')
        for i in val_indices:
            os.system(f'cp {os.path.join(org_dir, c, names[i])} {os.path.join(out_dir_val, c, names[i])}')


if __name__ == '__main__':
    main()