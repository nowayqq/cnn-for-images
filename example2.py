import numpy as np
import pandas as pd
import PIL.Image
import glob

from sys import exit
from keras.models import load_model


model = load_model('model.h5')
dataset_path = "your_images/"
classes = np.array(['airplane', 'automobile', 'bird',
                    'cat', 'deer', 'dog', 'frog',
                    'horse', 'ship', 'truck'])

i = 0
images = []

for filename in glob.glob(f'{dataset_path}/*.jpg'):
    im = PIL.Image.open(filename)
    im = im.resize((32, 32), PIL.Image.Resampling.LANCZOS)
    images.append(im)

if len(images) == 0:
    exit(f'No images in {dataset_path} directory\n' +
         'Note: \'.jpg\' is only acceptable format')

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
