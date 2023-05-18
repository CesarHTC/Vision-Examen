from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.utils import np_utils
import matplotlib.pyplot as plt
import numpy as np


#cargo mis variables con las imagenes del data set cifar
(x_train, y_train), (x_test, y_test) = cifar10.load_data()


X_train = np.reshape( x_train, (x_train.shape[0],x_train.shape[1]*x_train.shape[2]) )
X_test = np.reshape( x_test, (x_test.shape[0],x_test.shape[1]*x_test.shape[2]) )



#Normalizamos las imagenes
X_train = X_train/255.0
X_test = X_test/255.0
#convertimos y_train y y_test en one-hot
nclasses = 10
Y_train = np_utils.to_categorical(y_train,nclasses)
Y_test = np_utils.to_categorical(y_test,nclasses)

#####################################       Creacion del modelo         ##############################################

np.random.seed(1)

input_dim = X_train.shape[1] #defino el tamaño de la capa de entrada, su dimencion sera 768 la imagen aplanada
output_dim = Y_train.shape[1] #defino la capa dee salida, el numero de categorias


modelo = Sequential()

modelo.add( Dense(15, input_dim=input_dim, activation='relu')) #defino la funcion de activacion con 15 neuronas
modelo.add( Dense(output_dim, activation='softmax'))
sgd = SGD(lr=0.2)#learnig reate
modelo.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])


puntaje = modelo.evaluate(X_test,Y_test,verbose=0)
print('Precisión en el set de validación: {:.1f}%'.format(100*puntaje[1]))


Y_prob = modelo.predict(X_test)
Y_pred = np.argmax(Y_prob, axis=1)

ids_imgs = np.random.randint(0,X_test.shape[0],9)
for i in range(len(ids_imgs)):
	idx = ids_imgs[i]
	img = X_test[idx,:].reshape(32,32)
	cat_original = np.argmax(Y_test[idx,:])
	cat_prediccion = Y_pred[idx]

	plt.subplot(3,3,i+1)
	plt.imshow(img, cmap='gray')
	plt.axis('off')
	plt.title('"{}" clasificado como "{}"'.format(cat_original,cat_prediccion))
plt.suptitle('Ejemplos de clasificación en el set de validación')
plt.show()