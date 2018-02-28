from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Embedding, Lambda, TimeDistributed
import keras.backend as K
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import keras
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tqdm import tqdm
import pickle as pkl
from keras.callbacks import TensorBoard
from time import time

#########################################################################################

time_delay = 20 #0
look_back = 50
n_epoch = 50
n_videos = 50
tbCallback = TensorBoard(log_dir="logs/{}".format(time())) # TensorBoard(log_dir='./Graph', histogram_freq=0, batch_size=n_batch, write_graph=True, write_images=True)

#########################################################################################

# Load the files
with open('data/audio_kp/audio_kp1467_mel.pickle', 'rb') as pkl_file:
	audio_kp = pkl.load(pkl_file)
with open('data/pca/pkp1467.pickle', 'rb') as pkl_file:
	video_kp = pkl.load(pkl_file)
with open('data/pca/pca1467.pickle', 'rb') as pkl_file:
	pca = pkl.load(pkl_file)


# Get the data

X, y = [], [] # Create the empty lists
# Get the common keys
keys_audio = audio_kp.keys()
keys_video = video_kp.keys()
keys = sorted(list(set(keys_audio).intersection(set(keys_video))))
# print('Length of common keys:', len(keys), 'First common key:', keys[0])

# X = np.array(X).reshape((-1, 26))
# y = np.array(y).reshape((-1, 8))

for key in tqdm(keys[0:n_videos]):
	audio = audio_kp[key]
	video = video_kp[key]
	if (len(audio) > len(video)):
		audio = audio[0:len(video)]
	else:
		video = video[0:len(audio)]
	start = (time_delay-look_back) if (time_delay-look_back > 0) else 0
	for i in range(start, len(audio)-look_back):
		a = np.array(audio[i:i+look_back])
		v = np.array(video[i+look_back-time_delay]).reshape((1, -1))
		X.append(a)
		y.append(v)

X = np.array(X)
y = np.array(y)
shapeX = X.shape
shapey = y.shape
print('Shapes:', X.shape, y.shape)
X = X.reshape(-1, X.shape[2])
y = y.reshape(-1, y.shape[2])
print('Shapes:', X.shape, y.shape)

scalerX = MinMaxScaler(feature_range=(0, 1))
scalery = MinMaxScaler(feature_range=(0, 1))

X = scalerX.fit_transform(X)
y = scalery.fit_transform(y)


X = X.reshape(shapeX)
y = y.reshape(shapey[0], shapey[2])

print('Shapes:', X.shape, y.shape)
print('X mean:', np.mean(X), 'X var:', np.var(X))
print('y mean:', np.mean(y), 'y var:', np.var(y))

split1 = int(0.8*X.shape[0])
split2 = int(0.9*X.shape[0])

train_X = X[0:split1]
train_y = y[0:split1]
val_X = X[split1:split2]
val_y = y[split1:split2]
test_X = X[split2:]
test_y = y[split2:]




# Initialize the model

model = Sequential()
model.add(LSTM(25, input_shape=(look_back, 26)))
model.add(Dropout(0.25))
model.add(Dense(8))
model.compile(loss='mean_squared_error', optimizer='adam')
print(model.summary())
# model = load_model('my_model.h5')

# train LSTM with validation data
for i in tqdm(range(n_epoch)):
	print('Epoch', (i+1), '/', n_epoch, ' - ', int(100*(i+1)/n_epoch))
	model.fit(train_X, train_y, epochs=1, batch_size=1, 
		verbose=1, shuffle=True, callbacks=[tbCallback], validation_data=(val_X, val_y))
	# model.reset_states()
	test_error = np.mean(np.square(test_y - model.predict(test_X)))
	# model.reset_states()
	print('Test Error: ', test_error)

# Save the model
model.save('my_model.h5')
model.save_weights('my_model_weights.h5')
print('Saved Model.')



# X, num = audioToPrediction('audios/' + key_audio + '.wav')
# y = model.predict(X, batch_size=n_batch)
# y = y.reshape(y.shape[0]*y.shape[1], y.shape[2]) 
# print('y:', y[0:num].shape)



