#todos los metodos que se suelen utilizar
import pandas as pd
import joblib


#creamos nuestra clase
class Utils:
    
    def load_from_csv(self, path):
        return pd.read_csv(path, sep=';')

    def load_from_mysql(self):  #debo poner self para que se llame a ella misma, puede tener varias funciones dentro de una misma clase
        pass  # puede estar vacia si no la utiliza


    def features_target(self, dataset, drop_cols, y):
        X = dataset.drop(drop_cols, axis=1)
        y = dataset[y]
        return X,y

    def model_export(self, clf, score):
        print(score)
        joblib.dump(clf, './models/best_model.pkl')