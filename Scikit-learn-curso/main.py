from utils import Utils
from models import Models

if __name__ == "__main__":

    Utils = Utils()
    models = Models()
    data = Utils.load_from_csv('./in/felicidad_ok.csv')
    X, y = Utils.features_target(data, ['score', 'rank', 'country'], ['score'])

    models.grid_training(X,y)

    print(data)


