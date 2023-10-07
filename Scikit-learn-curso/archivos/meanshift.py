#Si no se cuantas categorias tiene mi grupo decido que el algoritmo las clasifique

import pandas as pd
import matplotlib.pyplot as plt
#Importo el algoritmo meanshift
from sklearn.cluster import MeanShift


if __name__=="__main__":
    dataset = pd.read_csv("./data/candy.csv")
    print(dataset.head(5))

    X = dataset.drop('competitorname', axis=1) #Elimino la variable categorica ya que no se puede entrenar con el algotimo.

    meanshift = MeanShift().fit(X) # Guardo mi modelo, el tiene un parametro ancho banda "bandwidt", el por defecto lo puede poner
    print(max(meanshift.labels_))
    print("="*64)
    print(meanshift.cluster_centers_) #imprimir la clasificacion de los centros

    dataset['meanshift'] = meanshift.labels_
    print("=")
    print(dataset)


    # Gráfico de dispersión para visualizar los clusters
    plt.scatter(dataset['sugarpercent'], dataset['winpercent'], c=meanshift.labels_, cmap='viridis')
    plt.xlabel('Sugar Percent')
    plt.ylabel('Win Percent')
    plt.title('Mean Shift Clustering')
    plt.show()