#Енилов ИСТбд-42 3 лаб.
import csv
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as mpl
#Генерируем файл
titles=["Табельный номер", "Фамилия И.О.", "Пол", "Год рождения","Год начала работы в компании",
"Подразделение", "Должность", "Оклад", "Количество выполненных проектов"]
families=["Сурков", "Попов", "Ленин", "Краев", "Мельников", "Караваев", "Пустынников", "Сидоров",
"Москвичев", "Мельников", "Кузнецов", "Гончаров", "Рыбаков", "Смердяков", "Синицын", "Орлов", "Журавлев",
"Мясников", "Степанов", "Лобанов", "Быков", "Романов", "Рюриков", "Шпагин", "Зыков", "Шолохов","Лермонтов",
"Киров", "Галкин", "Медведев", "Козлов", "Щукин", "Андреев", "Авасев", "Коршунов"]
names=["А","Б","В","Г","Д","Е","Ж","З","И","К","Л","М","Н","О","П","Р","С","Т","У","Ф","Э","Ю","Я"]
divisions=["Секретариат","Служба HR","Бухгалтерия","Маркетинг","Служба безопасности","Юрототдел","Конструкторское бюро",
"Отдел автоматизации","Отдел контроля качества","Транспортный цех" ]
position=["Юрист","Младший специалист","Программист","Менеджер","Диспетчер","Специалист","Старший специалист","Бухгалтер","Экономист","Младший инженер","Инженер"]

data=list()

count=random.randint(1000,2000)
worker=[9]

with open("workers.csv", mode="w", encoding="utf-16") as wr_file:
    writer=csv.writer(wr_file, delimiter=";", lineterminator="\n")
    writer.writerow(titles)
    for i in range(count):
        #Создаем рабочего
        if random.randint(1,2)==1:
            sex="M"
        else:
            sex="W"
        tabNum=i
        if sex=="M":
            fio=(random.choice(families)+" "+random.choice(names)+"."+random.choice(names)+".")
        else:
            fio=(random.choice(families)+"а"+" "+random.choice(names)+"."+random.choice(names)+".")
        yearBirth=random.randint(1970,2004)
        yearStartWork=random.randint(2000,2022)
        division=random.choice(divisions)
        pos=random.choice(position)
        salary=random.randint(13000, 80000)
        salary -= salary % -10000
        countProj=random.randint(0,10)
        worker=[tabNum, fio, sex, yearBirth, yearStartWork, division, pos, salary, countProj]
        writer.writerow(worker)

#Читаем набор
with open("workers.csv", mode="r", encoding="utf-16") as r_file:
    yearStart=[]
    tubNum=[]
    pos=[]

    reader=csv.reader(r_file, delimiter=";",lineterminator="\n")
    count=0
    for row in reader:
        if count != 0:
            yearStart.append(int(row[4]))
            tubNum.append(int(row[0]))
            pos.append(row[6])
        count+=1

#Год начала работы
print("Статистика лет начала работы")
print("Максимальный год: ", np.max(yearStart))
print("Минимальный год: ", np.min(yearStart))
print("Средний год поступления на работу: ", np.mean(yearStart))
print("Дисперсия: ", np.var(yearStart))
print("Стандартное отклонение: ", np.std(yearStart))
print("Медиана: ", np.median(yearStart))

#Должность
#Считаем количество работников разных специальностей и виды специальностей
arrPos, countPos=np.unique(pos,return_counts=True)
i=0
maximum=0
minimum=10000
#Определяем индексы max и min значения
for value in countPos:
    if value>maximum:
        maximum=value
        indexMax=i
    if value<minimum:
        minimum=value
        indexMin=i
    i+=1
print("Статистические данные пункта - должность:")
print("Наибольшее количество специалистов в должности: ", arrPos[indexMax])
print("Наименьшее количество специалистов в должности: ", arrPos[indexMin])
print("Среднее количество работников каждой специальности: ", np.mean(countPos))
print("Дисперсия: ", np.var(countPos))
print("Стандартное отклонение: ", np.std(countPos))
print("Медиана: ", np.median(countPos))

#Табельный номер
print("Статистика табельных номеров")
print("Максимальный номер: ", np.max(tubNum))
print("Минимальный номер: ", np.min(tubNum))
print("Среднее табельных номеров: ", np.mean(tubNum))
print("Дисперсия: ", np.var(tubNum))
print("Стандартное отклонение: ", np.std(tubNum))
print("Медиана: ", np.median(tubNum))

#Pandas
dataframe=pd.read_csv("workers.csv", delimiter=";", lineterminator="\n", encoding="utf-16")

#Стаж
yearsStartWorkEmployee=dataframe["Год начала работы в компании"]
print("Статистика стажа в pandas")
print("Максимальный стаж работы: ", yearsStartWorkEmployee.min())
print("Минимальный стаж работы: ", yearsStartWorkEmployee.max())
print("Средний стаж работы: ", yearsStartWorkEmployee.mean())
print("Дисперсия: ", yearsStartWorkEmployee.var())
print("Стандартное отклонение: ", yearsStartWorkEmployee.std())
print("Медиана: ", yearsStartWorkEmployee.median())

#Считаем количество работников разных специальностей и виды специальностей
specializations=dataframe["Должность"]
count=specializations.value_counts()
print("Статистические данные пункта - должность в pandas:")
print("Наибольшее количество специалистов в должности: ", count.index.tolist()[0])
print("Наименьшее количество специалистов в должности: ", count.index.tolist()[len(count)-1])
print("Среднее количество работников каждой специальности: ", count.mean())
print("Дисперсия: ", count.var())
print("Стандартное отклонение: ", count.std())
print("Медиана: ", count.median)

#Табельный номер
tubNums=dataframe["Табельный номер"]
print("Статистика табельных номеров в pandas:")
print("Максимальный номер: ", tubNums.max())
print("Минимальный номер: ", tubNums.min())
print("Среднее табельных номеров: ", tubNums.mean())
print("Дисперсия: ", tubNums.var())
print("Стандартное отклонение: ", tubNums.std())
print("Медиана: ", tubNums.median())

#Графики
print("Гистограмма распределения лет начала работы")
x = np.linspace(2000,2022, 23)
mpl.hist(yearsStartWorkEmployee, x, rwidth=0.9)
mpl.title("Гистограмма распределения лет начала работы")
mpl.show()

print("Диаграмма распределения специальностей")
mpl.pie(count,labels=count.index.tolist())
mpl.title("Диаграмма распределения специальностей")
mpl.show()

dictSalYear=dataframe[["Оклад", "Год начала работы в компании"]]
#Высчитываем среднюю зарплату в зависимости от стажа
#Подсчитываем сумму всех зарплат в зависимости от стажа
arrSumSal=np.empty(23)
countWorkExp=np.zeros(23)
for row in dictSalYear.values:
    arrSumSal[2022-int(row[1])]+=int(row[0])
    countWorkExp[2022-int(row[1])]+=1
#Усредняем значения
for i in range(23):
    arrSumSal[i]=arrSumSal[i]/countWorkExp[i]
print("График распределения зарплаты в зависимости от стажа")
mpl.plot(np.linspace(0,22, 23),arrSumSal)
mpl.title("График распределения зарплаты в зависимости от стажа")
mpl.show()