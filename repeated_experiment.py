import os

for i in range(100):
    os.system('python train.py' + ' --seed_number ' + str(i))
