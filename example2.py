import numpy as np
import pandas as pd
import PIL.Image

from os.path import exists
from sys import exit
from keras.models import load_model


model = load_model('model.h5')
dataset_path = "your_images/"
classes = np.array(['airplane', 'automobile', 'bird',
                    'cat', 'deer', 'dog', 'frog',
                    'horse', 'ship', 'truck'])

i = 0
images = []

while exists(f'{dataset_path}img{i}.jpg'):
    img = PIL.Image.open(f'{dataset_path}img{i}.jpg')
    img = img.resize((32, 32), PIL.Image.Resampling.LANCZOS)
    images.append(img)
    i += 1

if len(images) == 0:
    exit(f'No images in {dataset_path} directory\n' +
         'Or you named them wrong, you should name ' +
         'them \'img0\', \'img1\', \'img2\', etc\n' +
         'And acceptable format is only \'.jpg\'')

for i in range(len(images)):
    images[i] = np.array(images[i])

images = np.array(images).astype('float32') / 255.0

pred = model.predict(images)

df = pd.DataFrame(np.round(pred * 100, 2))
df.columns = classes

arr = []

for i in range(len(df.index)):
    for item in df:
        for value in df[item]:
            if value == df.iloc[i].max():
                arr.append((item, value))

for i in range(len(arr)):
    print(f'The img{df.index[i]} was predicted to be a ' +
          f'{arr[i][0]} with {np.round(arr[i][1], 2)} percent')
