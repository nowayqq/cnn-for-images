import numpy as np
import pandas as pd
import PIL.Image

from keras.models import load_model


model = load_model('model.h5')
dataset_path = "images/"

dog = PIL.Image.open(dataset_path + 'dog' + '.jpg')
dog = dog.resize((32, 32), PIL.Image.ANTIALIAS)

cat = PIL.Image.open(dataset_path + 'cat' + '.jpg')
cat = cat.resize((32, 32), PIL.Image.ANTIALIAS)

automobile = PIL.Image.open(dataset_path + 'automobile' + '.jpg')
automobile = automobile.resize((32, 32), PIL.Image.ANTIALIAS)

ship = PIL.Image.open(dataset_path + 'ship' + '.jpg')
ship = ship.resize((32, 32), PIL.Image.ANTIALIAS)

dog2 = PIL.Image.open(dataset_path + 'dog2' + '.jpg')
dog2 = dog2.resize((32, 32), PIL.Image.ANTIALIAS)

cat2 = PIL.Image.open(dataset_path + 'cat2' + '.jpg')
cat2 = cat2.resize((32, 32), PIL.Image.ANTIALIAS)

test_h = np.array([np.array(dog), np.array(cat),
                   np.array(automobile), np.array(ship),
                   np.array(dog2), np.array(cat2)])

test_h = test_h.astype('float32')
test_h = test_h / 255.0

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

tmp = pd.Series(np.round(pred[0] * 100), name='dog')
df = pd.DataFrame(tmp)
df.index = classes
df['dog'] = np.round(pred[0] * 100, 2)
df['cat'] = np.round(pred[1] * 100, 2)
df['automobile'] = np.round(pred[2] * 100, 3)
df['ship'] = np.round(pred[3] * 100, 2)
df['dog2'] = np.round(pred[4] * 100, 2)
df['cat2'] = np.round(pred[5] * 100, 2)

arr = []
for item in df:
    arr.append(df[df[item] == max(df[item])].index[0])

for i in range(len(arr)):
    print(f'The {df.columns[i]} was predicted to be a {arr[i]}')
