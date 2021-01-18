from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

def train_detector(X_train, X_test, Y_train, Y_test, nb_filters = 32, batch_size=128, epochs=5, nb_classes=2, 
	do_augment=False, save_file='./detector_model.h5y', random_state=0):
	"""
	"""
	# Fix random number generator for reproducibility
	np.random.seed(random_state)

	# input image dimensions
	img_rows, img_cols = X_train.shape[1], X_train.shape[2]

	# size of pooling area for max pooling
	pool_size = (2, 2)
	# convolution kernel size
	kernel_size = (3, 3) 
	input_shape = (img_rows, img_cols, 3)


	model = Sequential()
	model.add(Conv2D(filters=nb_filters, kernel_size=(kernel_size[0], kernel_size[1]),
	                        padding='valid',
	                        input_shape=input_shape))
	model.add(Activation('relu'))
	model.add(Conv2D(filters=nb_filters, kernel_size=(kernel_size[0], kernel_size[1])))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=pool_size))
	# (16, 8, 32)

	model.add(Conv2D(filters=nb_filters*2, kernel_size=(kernel_size[0], kernel_size[1])))
	model.add(Activation('relu'))
	model.add(Conv2D(filters=nb_filters*2, kernel_size=(kernel_size[0], kernel_size[1])))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=pool_size))
	# (8, 4, 64) = (2048)

	model.add(Flatten())
	model.add(Dense(4096))
	model.add(Activation('relu'))
	
	model.add(Dense(4096))
	model.add(Activation('relu'))
	# model.add(Dropout(0.5))
	model.add(Dense(nb_classes))
	model.add(Activation('softmax'))

	model.compile(loss='categorical_crossentropy',
	              optimizer='adam',
	              metrics=['accuracy'])

	callback = EarlyStopping(monitor='val_loss', patience=3)

	if do_augment:
	    datagen = ImageDataGenerator(
	        rotation_range=20,
	        width_shift_range=0.2,
	        height_shift_range=0.2,
	        shear_range=0.2,
	        zoom_range=0.2)
	    datagen.fit(X_train)
	    history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
	                        steps_per_epoch=np.int(len(X_train)/batch_size), epochs=epochs,
	                        validation_data=(X_test, Y_test))
	else:
	    history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs,
	          verbose=1, callbacks=[callback], validation_data=(X_test, Y_test))
    
    # Save model
	model.save(save_file)


	score = model.evaluate(X_test, Y_test, verbose=0)
	print('Test score:', score[0])
	print('Test accuracy:', score[1])

	try:
		pred = model.predict(X_test)
		print(classification_report(Y_test.argmax(axis=1), pred.argmax(axis=1), 
			target_names=['bg', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']))
	except:
		pass
	

	# Plot
	try:
		loss = history.history['loss']
		val_loss = history.history['val_loss']
		acc = history.history['accuracy']
		val_acc = history.history['val_accuracy']

		plt.plot(np.arange(len(loss)), loss, label='loss')
		plt.plot(np.arange(len(loss)), val_loss, label='val_loss')
		plt.plot(np.arange(len(loss)), acc, label='acc')
		plt.plot(np.arange(len(loss)), acc, label='val_acc')
		plt.legend(loc='lower left')
		plt.show()
	except:
		pass