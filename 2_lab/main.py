#Енилов ИСТбд-42 2 лаб. 8 вариант
import numpy as np
import matplotlib.pyplot as mpl

K=int(input("Введите K"))
N=int(input("Введите четное N, будет создана матрица N*N"))

matrB=np.random.randint(-10,10,(N//2,N//2))
matrC=np.random.randint(-10,10,(N//2,N//2))
matrD=np.random.randint(-10,10,(N//2,N//2))
matrE=np.random.randint(-10,10,(N//2,N//2))

matrA=np.array([[matrE, matrB],[matrD, matrC]])
matrF=matrA

arrNat=np.array([-7,-5,-3,-2,2,3,5,7])

print("Матрица B")
print(matrB)
print("Матрица C")
print(matrC)
print("Матрица D")
print(matrD)
print("Матрица E")
print(matrE)
print("Матрица A")
print(matrA)

countNatC=0
flag=0
#Считаем количество натуральных чисел в нечетных столбцах matrС3
for row in range(0, N//2):
    for col in range(1, N//2, 2):
        for nat in range(0, 7):
            if matrC[row][col] == arrNat[nat]:
                flag=1
        if flag == 1:
            countNatC+=1
            flag=0
print("Количество простых чисел в нечетных столбцах: ", countNatC)

countZeroC=0
#Считаем количество нулевых элементов в четных строках matrС3
for row in range(0, N//2, 2):
    for col in range(0, N//2):
        if matrC[row][col] == 0:
            countZeroC+=1
print("Количество нулевых элементов в четных строках: ", countZeroC)

cont=0
r=N//2-1
c=N//2-1
print("Матрица F перед преобразованием:")
print(matrF)

if countNatC > countZeroC:
    print("Меняем E и C симметрично")
    for row in range(0, N//2):
        for col in range(0, N//2):
            cont = matrF[0][0][row][col]
            matrF[0][0][row][col] = matrF[1][1][r][c]
            matrF[1][1][r][c] = cont
            r -= 1
        c -= 1
        r = N//2-1
else:
    print("Меняем B и C несимметрично")
    for row in range(0, N//2):
        for col in range(0, N//2):
            cont=matrF[0][1][row][col]
            matrF[0][1][row][col]=matrF[1][1][row][col]
            matrF[1][1][row][col]=cont

print("Матрица F после преобразования:")
print(matrF)

#Объединение подматриц A в одну
matrixA1=np.concatenate((matrE,matrB), axis=1)
matrixA2=np.concatenate((matrD,matrC), axis=1)
matrixA=np.concatenate((matrixA1, matrixA2), axis=0)

#Объединение подматриц F в одну
matrixF1=np.concatenate((matrF[0][0],matrF[0][1]), axis=1)
matrixF2=np.concatenate((matrF[1][0],matrF[1][1]), axis=1)
matrixF=np.concatenate((matrixF1, matrixF2), axis=0)

detA=np.linalg.det(matrixA)
print("Определитель матрицы А:", detA)
sumDiagF=np.trace(matrF[0][0])+np.trace(matrF[1][1])
print("Сумма диагонали матрицы F:", sumDiagF)
matrAT = np.transpose(matrixA)
print("Транспонированная матрица А:", matrAT)

if detA>sumDiagF:
    #Вычисляем A^(-1)*A^T-K*F
    matrAInv=np.linalg.inv(matrixA)
    out=matrAInv*matrAT-K*matrixF
    print("Определитель больше суммы, результат выражения A^(-1)*A^T-K*F: ", out)
else:
    #Вычисляем (A^T+G^(-1)-F^(-1))*K
    matrG=np.tril(matrixA)
    matrGInv=np.linalg.inv(matrG)
    matrFInv=np.linalg.inv(matrixF)
    out=(matrAT+matrGInv-matrFInv)*K
    print("Определитель меньше или равен сумме, результат выражения (A^T+G^(-1)-F^(-1))*K: ", out)

print("График значений строк матрицы F")
x = np.linspace(0, N, N)
y = np.asarray(matrixF)
mpl.plot(x, y)
mpl.title("График значений строк матрицы F")
mpl.show()

print("Гистограмма распределения значений матрицы F")
x = np.linspace(-10, 10, 21)
y = np.asarray(matrixF).reshape(-1)
mpl.hist(y, x, rwidth=0.9)
mpl.title("Гистограмма распределения значений матрицы F")
mpl.show()

print("Диаграмма распределения значений подматрицы B матрицы F")
y = np.asarray(matrF[0][1]).reshape(-1)
#Считаем количество значений
arrCount=np.zeros(21)
for i in range(0,len(y)):
    arrCount[y[i]+10]+=1
names=[-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10]
#Отсеиваем нулевые значения
namesB = [value for value in names if arrCount[value+10]!=0]
valuesB = [value for value in arrCount if value!=0]
mpl.pie(valuesB,labels=namesB)
mpl.title("Диаграмма распределения в подматрице B")
mpl.show()