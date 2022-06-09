import numpy as np
import pandas as pd
import PIL.Image

from keras.models import load_model


model = load_model('model.h5')
dataset_path = "images/"

file_names = ['dog', 'cat', 'automobile',
              'ship', 'dog2', 'cat2']

test_h = []

for i in range(len(file_names)):
    test_h.append(PIL.Image.open(dataset_path + file_names[i] + '.jpg'))
    test_h[i] = test_h[i].resize((32, 32), PIL.Image.ANTIALIAS)
    test_h[i] = np.array(test_h[i])

test_h = np.array(test_h).astype('float32') / 255.0

answers = np.array([np.zeros(10), np.zeros(10),
                    np.zeros(10), np.zeros(10),
                    np.zeros(10), np.zeros(10)])
answers[0][5] = 1
answers[1][3] = 1
answers[2][1] = 1
answers[3][8] = 1
answers[4][5] = 1
answers[5][3] = 1

pred = model.predict(test_h)

result = model.evaluate(test_h, answers, verbose=0)
print("Accuracy: %.2f%%" % (result[1] * 100))

classes = np.array(['airplane', 'automobile', 'bird',
                    'cat', 'deer', 'dog', 'frog',
                    'horse', 'ship', 'truck'])

df = pd.DataFrame()
df.index = classes

for i in range(len(file_names)):
    df[file_names[i]] = np.round(pred[i] * 100, 2)

arr = []
for item in df:
    arr.append(df[df[item] == max(df[item])].index[0])

for i in range(len(arr)):
    print(f'The {df.columns[i]} was predicted to be a {arr[i]} ' +
          f'with {np.round(df[file_names[i]].max(), 2)} percent')
