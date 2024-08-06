import os
import shutil


for name in os.listdir('.'):
    print(name)
    if not os.path.isdir(name):
        continue
    if not os.path.exists(os.path.join(name, 'network.png')):
        continue
    shutil.copy(
        os.path.join(os.path.join(name, 'network.png')),
        os.path.join(os.path.join('networks', f'{name}.png'))
    )
