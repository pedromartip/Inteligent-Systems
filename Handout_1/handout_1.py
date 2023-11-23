import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from keras import layers

#Cargamos los datos del txt
data = np.loadtxt('C:/Users/ca01770/Documents/GitHub/Inteligent-Systems/Handout_1/ic_lab1.txt')
X = data[:,:-1]
y = data[:,-1]

# Asignamos las clases
M = len(np.unique(y))
print(f'Numero de clases = {M}')

# Proceso de normalización
scaler = StandardScaler()
X_ = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_, y, test_size=M, random_state=100)

# Definición de red neuronal de un solo hidden layer

inputs = keras.Input(shape=(2,))
x = layers.Dense(2, activation="relu")(inputs)
outputs = layers.Dense(1, activation="softmax")(x)
model1 = keras.Model(inputs, outputs)
model1.summary()



