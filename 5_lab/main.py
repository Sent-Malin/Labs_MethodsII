# Енилов ИСТбд-42 5 лаб.
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB

def quadrDiscrAnalys(X_train, y_train, X_test, y_test):
    xtr=Normalizer().fit_transform(X_train)
    xtst=Normalizer().fit_transform(X_test)
    model=QuadraticDiscriminantAnalysis()
    model.fit(xtr, y_train)
    predict_model=model.predict(xtst)
    print('Основные метрики классификации')
    print(classification_report(y_test, predict_model, zero_division=0))

    print('Матрица ошибок для оценки точности')
    print(confusion_matrix(y_test, predict_model))

    print('Счет X-train с Y-train : ', model.score(xtr, y_train))
    print('Счет X-test  с Y-test  : ', model.score(xtst, y_test))
    print('Точность метода квадратичного дискриминантного анализа ', accuracy_score(y_test, predict_model))

    point=300
    # Графическое отображение признаков
    fig, ax = plt.subplots(2, 1, figsize=(12, 12))
    ax[0].scatter(X_test[:point, 0]+X_test[:point, 2]+X_test[:point, 4]+X_test[:point, 6]+
    X_test[:point, 8], X_test[:point, 1]+X_test[:point, 3]+X_test[:point, 5]+X_test[:point, 7]+
    X_test[:point, 9], c=predict_model[:point])
    ax[0].set_title('Зависимость ранга карт от их мастей(прогноз)')

    ax[1].scatter(X_test[:point, 0]+X_test[:point, 2]+X_test[:point, 4]+X_test[:point, 6]+
    X_test[:point, 8], X_test[:point, 1]+X_test[:point, 3]+X_test[:point, 5]+X_test[:point, 7]+
    X_test[:point, 9], c=y_test[:point])
    ax[1].set_title('Зависимость ранга карт от их мастей(по факту)')
    plt.show()

def logisticRegr(X_train, y_train, X_test, y_test):
    xtr=Normalizer().fit_transform(X_train)
    xtst=Normalizer().fit_transform(X_test)
    model=LogisticRegression()
    model.fit(xtr, y_train)
    predict_model=model.predict(xtst)
    print('Основные метрики классификации')
    print(classification_report(y_test, predict_model, zero_division=0))

    print('Матрица ошибок для оценки точности')
    print(confusion_matrix(y_test, predict_model))

    print('Счет X-train с Y-train : ', model.score(xtr, y_train))
    print('Счет X-test  с Y-test  : ', model.score(xtst, y_test))
    print('Точность метода логистической регрессии ', accuracy_score(y_test, predict_model))

    point=300
    # Графическое отображение признаков
    fig, ax = plt.subplots(2, 1, figsize=(12, 12))
    ax[0].scatter(X_test[:point, 0]+X_test[:point, 2]+X_test[:point, 4]+X_test[:point, 6]+
    X_test[:point, 8], X_test[:point, 1]+X_test[:point, 3]+X_test[:point, 5]+X_test[:point, 7]+
    X_test[:point, 9], c=predict_model[:point])
    ax[0].set_title('Зависимость ранга карт от их мастей(прогноз)')

    ax[1].scatter(X_test[:point, 0]+X_test[:point, 2]+X_test[:point, 4]+X_test[:point, 6]+
    X_test[:point, 8], X_test[:point, 1]+X_test[:point, 3]+X_test[:point, 5]+X_test[:point, 7]+
    X_test[:point, 9], c=y_test[:point])
    ax[1].set_title('Зависимость ранга карт от их мастей(по факту)')
    plt.show()

def KNN(X_train, y_train, X_test, y_test):
    model=KNeighborsClassifier(n_neighbors=7)
    model.fit(X_train, y_train)
    predict_model=model.predict(X_test)
    print('Основные метрики классификации')
    print(classification_report(y_test, predict_model, zero_division=0))

    print('Матрица ошибок для оценки точности')
    print(confusion_matrix(y_test, predict_model))

    print('Счет X-train с Y-train : ', model.score(X_train, y_train))
    print('Счет X-test  с Y-test  : ', model.score(X_test, y_test))
    print('Точность метода k ближайших соседей ', accuracy_score(y_test, predict_model))

    point=300
    # Графическое отображение признаков
    fig, ax = plt.subplots(2, 1, figsize=(12, 12))
    ax[0].scatter(X_test[:point, 0]+X_test[:point, 2]+X_test[:point, 4]+X_test[:point, 6]+
    X_test[:point, 8], X_test[:point, 1]+X_test[:point, 3]+X_test[:point, 5]+X_test[:point, 7]+
    X_test[:point, 9], c=predict_model[:point])
    ax[0].set_title('Зависимость ранга карт от их мастей(прогноз)')

    ax[1].scatter(X_test[:point, 0]+X_test[:point, 2]+X_test[:point, 4]+X_test[:point, 6]+
    X_test[:point, 8], X_test[:point, 1]+X_test[:point, 3]+X_test[:point, 5]+X_test[:point, 7]+
    X_test[:point, 9], c=y_test[:point])
    ax[1].set_title('Зависимость ранга карт от их мастей(по факту)')
    plt.show()

# Читаем данные
dataset_train=pd.read_csv('poker-hand-training-true.data')
dataset_test=pd.read_csv('poker-hand-testing.data')

print('\nHead dataset:')
print(dataset_train.head())
print('\nРазмерность тренировочной выборки (строки, столбцы)',dataset_train.shape)
print('\nРазмерность тестовой выборки (строки, столбцы)',dataset_test.shape)

X_train = dataset_train.drop(columns='CLASS').values
y_train=dataset_train['CLASS'].values
X_test=dataset_test.drop(columns='CLASS').values
y_test=dataset_test['CLASS'].values

print('\nРазмеры тренировочной и тестовой выборок')
print('X train : ', X_train.shape, 'Y train : ', y_train.shape)
print('X test  : ', X_test.shape, 'Y test  : ', y_test.shape)

print('\nКвадратичный дискриминантный анализ: \n')
quadrDiscrAnalys(X_train, y_train,  X_test, y_test)

print('\nЛогистическая регрессия: \n')
logisticRegr(X_train, y_train,  X_test, y_test)

print('\nK ближайших соседей: \n')
KNN(X_train, y_train,  X_test, y_test)