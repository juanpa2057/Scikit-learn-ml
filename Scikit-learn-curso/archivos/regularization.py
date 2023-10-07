import pandas as pd
import sklearn


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


#Metrecias para cargar datos y medirlos
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

if __name__ == '__main__':
    dataset = pd.read_csv('./data/felicidad.csv')
    print(dataset.describe())

    X = dataset[['gdp', 'family', 'lifexp', 'freedom', 'corruption', 'generosity', 'dystopia']]
    Y = dataset[['score']]

    #ahora miro si la particicion de feactures "entradas" y target "variable objectivo" tiene el mismo tamaño
    print(X.shape)
    print(Y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.25)

    modelLinear = LinearRegression().fit(X_train, y_train)
    y_predict_linear = modelLinear.predict(X_test)

    #entre mas grande sea el alpha , mas grande sera la penalizacion
    modelLasso = Lasso(alpha=0.02).fit(X_train, y_train)
    y_predict_lasso = modelLasso.predict(X_test)

    modelRidge = Ridge(alpha=1).fit(X_train, y_train)
    y_predict_ridge = modelRidge.predict(X_test)

    #Predeccir nuestra pérdida
    linear_loss = mean_squared_error(y_test, y_predict_linear)
    print("Linear Loss: ", linear_loss)

    lasso_loss = mean_squared_error(y_test, y_predict_lasso)
    print("Lasso loss: ", lasso_loss)

    ridge_loss = mean_squared_error(y_test, y_predict_ridge)
    print("Ridge loss:", ridge_loss)


    print("="*32)
    print("Coef Linear")
    print(modelLinear.coef_)

    print("="*32)
    print("Coef LASSO")
    print(modelLasso.coef_)

    print("="*32)
    print("Coef RIDGE")
    print(modelRidge.coef_)


# Un valor más bajo de la métrica de pérdida indica un mejor rendimiento del modelo.
#  En este caso, el modelo con la menor pérdida es el modelo de regresión lineal
#Linear Loss:  9.875049620340361e-08
#Lasso loss:  0.05188912583862035
#Ridge loss: 0.005502323291684915