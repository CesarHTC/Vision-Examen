from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.utils import np_utils
import matplotlib.pyplot as plt
import numpy as np

#Cifar 10 en un dataset de de fiferente tipo  de objetos la siferencia con mnist es que este data set es de imagenes a color por lo que es necesario agregar una dimension
#cargo mis variables con las imagenes del data set cifar
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

#######################      Pre-procesamiento del set de datos      #############################################
#Keras no interpreta matrices por lo que debo trasformar las imagenes(matris de 28x28)
#a un vectre de 784x1


X_train = np.reshape(x_train, (x_train.shape[0], np.prod(x_train.shape[1:])))
X_test = np.reshape(x_test, (x_test.shape[0], np.prod(x_test.shape[1:])))


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

#Crea un modelo secuencial de Keras, que es una pila lineal de capas de redes neuronales.
modelo = Sequential()
#Agrega una capa densa con 2900 neuronas, que tiene una función 
#de activación ReLU. La capa recibe una entrada de tamaño input_dim, que es la dimensión de la imagen aplanada.
modelo.add( Dense(290, input_dim=input_dim, activation='relu'))

modelo.add( Dense(output_dim, activation='softmax'))
sgd = SGD(lr=0.15)#learnig reate
modelo.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

historia = modelo.fit(X_train, Y_train, epochs=20, batch_size=700, verbose=2)

puntaje = modelo.evaluate(X_test,Y_test,verbose=0)
print('Precisión en el set de validación: {:.1f}%'.format(100*puntaje[1]))

Y_prob = modelo.predict(X_test)
Y_pred = np.argmax(Y_prob, axis=1)
label= ["Avion","Auto","Ave","Gato","Venado","Perro","Rana","Caballo","Barco","Camion"]
clasificacion = np.random.randint(0,X_test.shape[0],9)
for i in range(len(clasificacion)):
    idx = clasificacion[i]
    img = X_test[idx,:].reshape(32,32,3)
    cat_original = np.argmax(Y_test[idx,:])
    cat_prediccion = Y_pred[idx]

    plt.subplot(4,3,i+1)
    plt.imshow(img)
    plt.axis('off')
    plt.title('"{}" clasificado como "{}"'.format(label[cat_original], label[cat_prediccion]))

plt.suptitle('Ejemplos de clasificación en el set de validación')
plt.rcParams['figure.figsize'] = [10, 10]
plt.show()
