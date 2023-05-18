import numpy as np

from tensorflow import keras
# Datos de entrenamiento
celsius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

# Definir el modelo de la red neuronal

model = keras.Sequential([ keras.layers.Dense(units=1, input_shape=[1]), keras.layers.Dense(units=1, input_shape=[1])])

# Compilar el modelo
model.compile(optimizer=keras.optimizers.Adam(0.1), loss='mean_squared_error')

# Entrenar el modelo
model.fit(celsius, fahrenheit, epochs=600,verbose=False)

# Predecir los valores de Fahrenheit para una nueva entrada de Celsius

resultado = model.predict([90])
print("115 grados celcuis a fahrenheit son: ",resultado)
