import pandas as pd
import warnings 
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

from sklearn.cluster import MiniBatchKMeans

if __name__=="__main__":
    dataset = pd.read_csv("./data/candy.csv")
    print(dataset.head(5))

    X = dataset.drop('competitorname', axis=1) #Elimino la variable categorica ya que no se puede entrenar

    kmeans = MiniBatchKMeans(n_clusters=4, batch_size=8).fit(X)
    print("Total de centros: ", len(kmeans.cluster_centers_))
    print("="*64)
    labels = kmeans.predict(X)

    dataset['group'] = labels

    print(dataset)

    # Gráfico de dispersión para visualizar los grupos
    scatter = plt.scatter(dataset['sugarpercent'], dataset['winpercent'], c=labels, cmap='viridis')
    plt.xlabel('Sugar Percent')
    plt.ylabel('Win Percent')
    plt.title('MiniBatchKMeans Clustering')

    # Crear una leyenda con etiquetas de grupo
    legend1 = plt.legend(*scatter.legend_elements(), title="Grupos")
    plt.gca().add_artist(legend1)

    plt.show()
