# red_neuronal.py

```Phyton
import tensorflow as tf 
from tensorflow.keras.datasets import mnist 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping 
from tensorflow.keras.utils import to_categorical import matplotlib.pyplot as plt


(x_entrenamiento, y_entrenamiento), (x_prueba, y_prueba) = mnist.load_data()

x_entrenamiento = x_entrenamiento/ 255.0 x_prueba = x_prueba / 255.0

y_entrenamiento =
to_categorical(y_entrenamiento, 10) 
y_prueba = to_categorical(y_prueba, 10)

#Crear el modelo secuencial con capas adicionales

modelo = Sequential([ 
Flatten(input_shape=(28, 28)),  
Dense(256, activation='relu'),     Dropout(0.3),                    
Dense(128, activation='relu'),     
Dense(10, activation='softmax') 
])

modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

detener_temprano = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

historial = model.fit( 
x_entrenamiento, y_entrenamiento,
epochs=15, 
batch_size=32, 
validation_split=0.1, 
callbacks=[detener_temprano], 
verbose=2 
)

#Evaluar el modelo en el conjunto de prueba

pérdida, precisión = modelo.evaluate(x_prueba, y_prueba) print(f"\nPérdida en test: {pérdida:.4f}") print(f"Precisión en test: {precisión:.4f}")

#Graficar la precisión y la pérdida

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1) plt.plot(history.history['precisión'], label='Entrenamiento') plt.plot(history.history['val_precisión'], label='Validación') plt.title('Precisión por época') plt.xlabel('Época') plt.ylabel('Precisión') plt.legend()

plt.subplot(1, 2, 2) plt.plot(history.history['pérdida'], label='Entrenamiento') plt.plot(history.history['val_pérdida'], label='Validación') plt.title('Pérdida por época') plt.xlabel('Época') plt.ylabel('Pérdida') plt.legend()

plt.tight_layout()
plt.show()
```
###Explicación 
Este proyecto utiliza el conjunto de datos MNIST, que contiene imágenes de dígitos manuscritos del 0 al 9, para construir y entrenar un modelo de red neuronal utilizando TensorFlow y Keras. A continuación, se describen los pasos realizados en el código.

1. Cargar el Dataset MNIST

Se carga el conjunto de datos MNIST utilizando la función load_data() de Keras. Este conjunto de datos se divide en dos partes: un conjunto de entrenamiento y un conjunto de prueba.

2. Normalizar los Datos

Los valores de los píxeles de las imágenes se normalizan a un rango de 0 a 1 dividiendo por 255.0. Esto ayuda a mejorar la convergencia del modelo durante el entrenamiento.

3. One-Hot Encoding de las Etiquetas

Las etiquetas de clase se convierten a un formato de codificación one-hot utilizando la función to_categorical(). Esto es necesario para que el modelo pueda aprender a clasificar correctamente las imágenes.

4. Crear el Modelo Secuencial

Se define un modelo secuencial que consiste en varias capas:

• Capa de entrada: Flatten, que convierte las imágenes de 28x28 píxeles en vectores de 784 elementos.

• Capa oculta 1: Dense con 256 neuronas y activación ReLU.

• Dropout: Se aplica una capa de Dropout con una tasa del 30% para evitar el sobreajuste.

• Capa oculta 2: Otra capa Dense con 128 neuronas y activación ReLU.

• Capa de salida: Dense con 10 neuronas y activación softmax para clasificar los dígitos.
    
5. Compilar el Modelo

El modelo se compila utilizando el optimizador Adam y la función de pérdida categorical_crossentropy, además de establecer la métrica de precisión para evaluar el rendimiento.

6. Early Stopping para Evitar Sobreentrenamiento

Se implementa el mecanismo de EarlyStopping para detener el entrenamiento si la pérdida de validación no mejora después de 3 épocas, lo que ayuda a prevenir el sobreentrenamiento.

7. Entrenar el Modelo

El modelo se entrena durante 15 épocas con un tamaño de lote de 32, utilizando un 10% de los datos de entrenamiento para validación.

8. Evaluar el Modelo en el Conjunto de Prueba

Se evalúa el rendimiento del modelo en el conjunto de prueba y se imprime la pérdida y precisión obtenidas.

9. Graficar la Precisión y la Pérdida

Finalmente, se grafican las métricas de precisión y pérdida tanto para el conjunto de entrenamiento como para el de validación a lo largo de las épocas.
