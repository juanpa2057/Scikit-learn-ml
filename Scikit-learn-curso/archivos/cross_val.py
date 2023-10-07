import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import (
    cross_val_score, KFold
)

from sklearn.metrics import mean_squared_error

if __name__ == "__main__":

    dataset = pd.read_csv('./data/felicidad_ok.csv', sep=';')

    X = dataset.drop(['country', 'score'], axis=1) #feactures
    y = dataset['score']  #variable a predecir

    model = DecisionTreeRegressor()
    #para hacer una validación rapida , utilizando la conf por defecto. cada valor darar el error cuadratico medio
    score = cross_val_score(model, X,y, cv=3, scoring='neg_mean_squared_error') # con el cv definimos cuantos plieges de la data queremos
    print(np.abs(np.mean(score)))

    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    losses = []
    for train, test in kf.split(dataset):
  
        X_train = X.iloc[train]
        X_test = X.iloc[test]
        y_train = y[train]
        y_test = y[test]
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        loss = mean_squared_error(y_test, predictions)
        losses.append(loss)

    print('Error para cada partición: ', losses)
    print('Promedio de los errores: ', np.mean(losses))