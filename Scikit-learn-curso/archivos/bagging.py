#Basados en los datos 1 presencia 0 ausencia de afectacion cardiaca

import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Importar el clasificador KNeighborsClassifier y BaggingClassifier de scikit-learn.
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier

# Importar herramientas para medir y evaluar modelos de aprendizaje automático.
# 'train_test_split' se utiliza para dividir el conjunto de datos en datos de entrenamiento y prueba.
# 'accuracy_score' se utiliza para calcular la precisión de las predicciones de los modelos.
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


if __name__ == "__main__":
    dt_heart = pd.read_csv('./data/heart.csv')
    print(dt_heart['target'].describe())

    X = dt_heart.drop(['target'], axis=1)  #copio el dataset pero sin la variable a predecir "feactures"
    y = dt_heart['target'] #Genero mi variable objetivo "target"

    #divido mis daros en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split (X ,y, test_size=0.35)

    knn_class = KNeighborsClassifier().fit(X_train, y_train)
    knn_pred = knn_class.predict(X_test)
    print("="*64)
    print(accuracy_score(knn_pred, y_test))

    bag_class = BaggingClassifier(base_estimator=KNeighborsClassifier(), n_estimators=50).fit(X_train, y_train)
    bag_pred = bag_class.predict(X_test)
    print("="*64)
    print(accuracy_score(bag_pred, y_test))
