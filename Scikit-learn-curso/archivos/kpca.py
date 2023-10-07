# Kernesl : Es una funcíon matemática que toma mediciones que se comportan de manera no lineal y las proyecta en un espacio
# dimensional más grande donde sean linealmente separables. ej: si tenemos un punto en 2 dimensiones se aplica kernel para que este 
# en 3 o 4 dimensiones se le sube la dimension a los datos. Sirven cuando los datos por medio de trazar una linea matematicamente no se 
# pueden clasificar. Se aumenta la dimensionalidad a otro plano de tal manera q con una funcion lineal pudieramos lograr esa clasificación.
#Algunos funciones kernel mas conocidas:  LINEALES - POLINOMIALES - GAUSSIANOS(RBF)

import pandas as pd
import sklearn
import matplotlib.pyplot as plt


from sklearn.decomposition import KernelPCA


   
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#Ejercicio con dataset pacientes con riesgo enfermeda cardiaca.
if __name__ == "__main__":
    dt_heart = pd.read_csv('./data/heart.csv')

    print(dt_heart.head(5))

    #partir las entradas por lo q vamos a entrenar  
    dt_features = dt_heart.drop(['target'], axis=1)
    dt_target = dt_heart['target']

    # Siempre en PCA se debe normalizar los datos con una función (usamos aca el standcarlar)
    dt_features = StandardScaler().fit_transform(dt_features)

    #Partir los datos en entrenamiento y testeo "utilizamos el trains_test_split".
    #para añadir replicabilidad se pone el ramdon_state q cuando sea mismo valor dara lo mismo.

    X_train, X_test, Y_train, Y_test = train_test_split(dt_features, dt_target, test_size=0.3, random_state=42)

    kpca = KernelPCA(n_components=4, kernel='linear')

    #ajustamos los datos
    kpca.fit(X_train)

    dt_train = kpca.transform(X_train)
    dt_test = kpca.transform(X_test)

    Logistic = LogisticRegression(solver='lbfgs')

    #Entrenamos los datos 
    Logistic.fit(dt_train, Y_train)
    print("SCRORE KPCA: ", Logistic.score(dt_test, Y_test))

# Regulazición: Consiste en disminuir la complejidad del modelo a través de 
# una penalización aplicada a sus variables mas irrelevantes.

#Pérdida o penalización: nos dice que tan lejos estamos de nuestros datos reales.
#Entre menor sea la pérdida mejor es el modelo.

# 3 tipos de Regularización:
# L1 LASSO: Reduccir la complejidad a través de la eliminación de feactures que no aportan demasiado al modelo.
#L2 Ridge: Reducir la complejidad disminuyendo el impacto de ciertes feactures a nuestro modelo.
#ElasticNet: Es una combinación de las dos anteriores.

