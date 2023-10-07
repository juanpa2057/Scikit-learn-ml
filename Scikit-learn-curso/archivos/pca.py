import pandas as pd
import sklearn
import matplotlib.pyplot as plt


from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
   
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

    print(X_train.shape)
    print(Y_train.shape)

    #configurar algoritmo PCA

    #n_components = min(n_muestras, n_features)
    pca = PCA(n_components=3)
    pca.fit(X_train)

    #el incrementa ipca es si no tenemos buen pc o rendimiento y el no manda todo los bloques si no q lo hace por partes
    ipca = IncrementalPCA(n_components=3 , batch_size=10)
    ipca.fit(X_train)

    #Medimos la varianza y comprobamos si el algoritmo esta aportando informacion valiosa
    plt.plot(range(len(pca.explained_variance_)), pca.explained_variance_ratio_)
    plt.show()

    #Corremos la regresión logistica y comparamos los dos algoritmos (pac & ipca)
    Logistic= LogisticRegression(solver='lbfgs')

    dt_train = pca.transform(X_train)
    dt_test = pca.transform(X_test)
    Logistic.fit(dt_train, Y_train)

    #Calcular la metricas de presicion
    print("SCORE PCA: ", Logistic.score(dt_test, Y_test))


    dt_train = ipca.transform(X_train)
    dt_test = ipca.transform(X_test)
    Logistic.fit(dt_train, Y_train)
    print("SCORE IPCA: ", Logistic.score(dt_test, Y_test))



