import numpy as np
import os

class_names = os.listdir(os.path.join("data", "our_filtered_train"))
num_flips = 200

for i in range(num_flips):
    c1 = np.random.choice(class_names)
    image_path = np.random.choice(os.listdir(os.path.join("data", "our_filtered_train", c1)))
    c2 = np.random.choice([c for c in class_names if c != c1])
    os.system(f'cp {os.path.join("data", "our_filtered_train", c1, image_path)} {os.path.join("data", "our_filtered_train", c2, image_path)}')