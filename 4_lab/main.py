#Енилов ИСТбд-42 4 лаб.
import csv
import math
import pylab as pl
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap

#Создаем файл
titles=["продукт","сладость","хруст","класс"]
product=["Яблоко", "салат","бекон","банан","орехи","рыба","сыр","виноград","морковь","апельсин"]
sweetness=[7,2,1,9,1,1,1,8,2,6]
crunch=[7,5,2,1,5,1,1,1,8,1]
classes=["Фрукт","Овощь","Протеин","Фрукт","Протеин","Протеин","Протеин","Фрукт","Овощь","Фрукт"]

item=[10]

with open("data.csv", mode="w") as wr_file:
    writer=csv.writer(wr_file, delimiter=";", lineterminator="\n")
    writer.writerow(titles)
    for i in range(0,10):
        item=[product[i],sweetness[i],crunch[i],classes[i]]
        writer.writerow(item)

#Читаем данные
with open("data.csv", mode="r") as r_file:
    data=[]
    reader=csv.reader(r_file, delimiter=";",lineterminator="\n")
    count=0
    #Переводим словесные описания классов в численные
    for row in reader:
        if count != 0:
            if row[3]=='Фрукт':
                cl=0
            if row[3]=='Овощь':
                cl=1
            if row[3]=='Протеин':
                cl=2
            data.append([row[0],int(row[1]),int(row[2]),cl])
        else: count+=1

#Функция отображения данных синий-протеин, красный - фрукты, зеленый - овощи
def showData (trainData):
    classColormap  = ListedColormap(['#FF0000', '#00FF00','#FFA500', '#0000FF'])
    #Принимает 2 набора точек и набор значений, определяющих класс, ну и цвет
    pl.scatter([trainData[i][1] for i in range(len(trainData))],
               [trainData[i][2] for i in range(len(trainData))],
               c=[trainData[i][3] for i in range(len(trainData))],
               cmap=classColormap)
    pl.show()

#Программируем метрический классификатор
def KNN(trainData, testData, k, numberOfClasses):
    def dist(a,b):
        return math.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)
    prognozCLasses=[]
    for testProduct in testData:
        testDist=[]
        #Считаем расстояния между тестовой точкой и всеми тренировочными точками
        #Создаем массив расстояний от точки testProduct до каждой тренировочной точки
        for i in range(len(trainData)):
            #Записываем в distance расстояние тест. точки до трен. и класс трен. точки
            distance=[dist([testProduct[1],testProduct[2]], [trainData[i][1],trainData[i][2]]), trainData[i][3]]
            testDist.append(distance)
        stat=[0 for i in range(numberOfClasses)]
        #Считаем количество точек каждого класса среди ближайших K соседей
        #Алгоритм сортирует расстояния до др. точек по убыванию, а потом
        #прибавляет ближайшие К соседей
        for d in sorted(testDist)[0:k]:
            stat[d[1]]+=1
        print("Распределение",k,"соседей элемента-",testProduct[0], "сладость=", testProduct[1]
        ," хруст=",testProduct[2]," классов фрукт, овощь, протеин, выпечка:", stat)
        prognozCLasses.append(sorted(zip(stat, range(numberOfClasses)),reverse=True)[0][1])
    return prognozCLasses

#Функция применения sklearn
def knn_sklearn(points, y, k):
    point_train, point_test, class_train, class_test = train_test_split(points,
    y,test_size=0.2, shuffle=False, stratify=None)
    scaler=StandardScaler()
    scaler.fit(point_train)

    point_train=scaler.transform(point_train)
    point_test=scaler.transform(point_test)

    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(point_train, class_train)
    prognoz=model.predict(point_test)
    return point_train, point_test, class_train, class_test, prognoz

#Применение ручного классификатора
#7 элементов в качестве тренировочной выборки и 3 в качестве тестовой
barier=6
trainVib=data[:barier]
testVib=data[barier+1:]
k=2
numClass=3
prognozes=KNN(trainVib,testVib,k,numClass)
print("Algorithm knn")
print("Тестовые продукты относятся к классам(Фрукт=0, Овощь=1, Протеин=2)", prognozes)
for i in range(0, len(prognozes)):
    if prognozes[i]==0:
        cl='Фрукт'
    if prognozes[i]==1:
        cl='Овощь'
    if prognozes[i]==2:
        cl='Протеин'
    print("Элемент ", testVib[i][0], 
    " по версии классификатора принадлежит к классу ", cl)
#Применение sklearn
print("sklearn")
points=[]
classes=[]
for i in range(len(data)):
    points.append([data[i][1],data[i][2]])
    classes.append(data[i][3])
point_train, point_test, classes_train, classes_test, prognoz=knn_sklearn(points, classes, 2)
print("По версии sklearn те же продукты относятся к классам(Фрукт=0, Овощь=1, Протеин=2): ", prognoz)
print("Правильный ответ: ", classes_test)
showData (data)

#Добавляем новый класс - выпечка
#Создаем новый файл
titles=["продукт","сладость","хруст","класс"]
product=["Яблоко", "салат","бекон","банан","орехи","рыба","сыр","виноград","морковь",
"апельсин", "хлебцы", "грейпфрут","багет","печенье","свекла","инжир","гранат","редис","чак-чак"]
sweetness=[7,2,1,9,1,1,1,8,2,6,1,9,4,8,4,10,8,1,5]
crunch=[7,5,2,1,5,1,1,1,8,1,11,6,9,10,6,2,6,6,9]
classes=["Фрукт","Овощь","Протеин","Фрукт","Протеин","Протеин","Протеин","Фрукт",
"Овощь","Фрукт","Выпечка","Фрукт","Выпечка","Выпечка","Овощь","Фрукт","Фрукт","Овощь","Выпечка"]

item=[19]

with open("dataup.csv", mode="w") as wr_file:
    writer=csv.writer(wr_file, delimiter=";", lineterminator="\n")
    writer.writerow(titles)
    for i in range(0,19):
        item=[product[i],sweetness[i],crunch[i],classes[i]]
        writer.writerow(item)

#Читаем данные
with open("dataup.csv", mode="r") as r_file:
    data=[]
    reader=csv.reader(r_file, delimiter=";",lineterminator="\n")
    count=0
    for row in reader:
        if count != 0:
            if row[3]=='Фрукт':
                cl=0
            if row[3]=='Овощь':
                cl=1
            if row[3]=='Протеин':
                cl=2
            if row[3]=='Выпечка':
                cl=3
            data.append([row[0],int(row[1]),int(row[2]),cl])
        else: count+=1

#Применение ручного классификатора
#15 элементов в качестве тренировочной выборки и 4 в качестве тестовой
barier=14
trainVib=data[:14]
testVib=data[barier+1:]
k=3
numClass=4
prognozes=KNN(trainVib,testVib,k,numClass)
print("Algorithm knn")
print("Тестовые продукты относятся к классам(Фрукт=0, Овощь=1, Протеин=2, Выпечка=3)", prognozes)
for i in range(0, len(prognozes)):
    if prognozes[i]==0:
        cl='Фрукт'
    if prognozes[i]==1:
        cl='Овощь'
    if prognozes[i]==2:
        cl='Протеин'
    if prognozes[i]==3:
        cl='Выпечка'
    print("Элемент ", testVib[i][0], 
    " по версии классификатора принадлежит к классу ", cl)

#Применение sklearn
print("sklearn")
points=[]
classes=[]
for i in range(len(data)):
    points.append([data[i][1],data[i][2]])
    classes.append(data[i][3])
point_train, point_test, classes_train, classes_test, prognoz=knn_sklearn(points, classes, 4)
print("По версии sklearn те же продукты относятся к классам(Фрукт=0, Овощь=1, Протеин=2, Выпечка=3): ", prognoz)
print("Правильный ответ: ", classes_test)

showData (data)