import os
exec(open("train_val_split.py").read())
print('Clearing confusing images.....')
exec(open("first_filtering.py").read())
print('Creating transformations.....')
exec(open("create_transformations.py").read())
# rename folders
for dirname in ['val', 'train', 'our_train', 'our_filtered_train']:
    os.system(f'rm -r {os.path.join("data", dirname)}')
os.system(f'mv {os.path.join("data", "our_train_aug")} {os.path.join("data", "train")}')
os.system(f'mv {os.path.join("data", "our_val")} {os.path.join("data", "val")}')
print('Training model.....')
exec(open("run_train_eval.py").read())
