import pandas as pd
import warnings
warnings.filterwarnings('ignore')

#Importar los modelos lineales (el ranscregressor y el hubber)
from sklearn.linear_model import (
    RANSACRegressor, HuberRegressor
)

#comparamos el resultado con un modelo regresor de SVM
from sklearn.svm import SVR

#Herramientas para cargar nuestros datos
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error #metrica para medir el error

#leo mi dataset
if __name__ == "__main__":
    dataset = pd.read_csv('./data/felicidad.csv', sep=";")
    print(dataset.head(5))


    X = dataset.drop(['country', 'score'], axis=1)  #el axis=1 es columnas el 0 es filas
    y = dataset[['score']] # target variable lo que queremos predecir

    #Partir la data en entrenamiento y prueba.
    #pasamos las variables y el tamaño. ramdom_sta para la replicabilidad puede ser cualquier numero, como semilla
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=7) 

    estimadores = {
        'SVR' : SVR(gamma= 'auto', C=1.0, epsilon=0.1),
        'RANSAC' : RANSACRegressor(), #Ransac es meta-estimador se puede trabajar con diferentes estomadores, si no le pones parametros trabaja con regresion lineal
        'HUBER' : HuberRegressor(epsilon=1.35) #el parametro es configurable si es pequeño muchos datos no van a ser considerados actipicos, si es mayor mas datos atipicos.

    }
#las funciones de python pueden retonar mas de un valor

    for name, estimador in estimadores.items():
        estimador.fit(X_train, y_train)
        predictions = estimador.predict(X_test)
        print("="*64)
        print(name)
        print('MSE: ', mean_squared_error(y_test, predictions))


#En resumen, SVR parece tener un MSE razonablemente bajo, lo que sugiere un buen rendimiento en la predicción de tus datos. 
# Por otro lado, tanto RANSAC como HUBER tienen MSE muy cercanos a cero, lo que podría indicar sobreajuste. 



import matplotlib.pyplot as plt

# Crear una figura con tres subparcelas
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))  # 1 fila, 3 columnas

for idx, (name, estimador) in enumerate(estimadores.items()):
    estimador.fit(X_train, y_train)
    predictions = estimador.predict(X_test)
    
    # Configurar el título y etiquetas
    axes[idx].set_title(name)
    axes[idx].set_ylabel('Predicted Score')
    axes[idx].set_xlabel('Real Score')
    
    # Dibujar el gráfico de dispersión y la línea de referencia
    axes[idx].scatter(y_test, predictions)
    axes[idx].plot(predictions, predictions, 'r--')

# Título general para todas las gráficas
plt.suptitle('Predicted vs. Real Score for Different Models', fontsize=16)

# Ajustar los espacios entre subparcelas
plt.tight_layout()

# Mostrar las gráficas
plt.show()

