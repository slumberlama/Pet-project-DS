from google.colab import files
uploaded = files.upload()
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set()
# корреляционная матрица
def plotCorrelationMatrix(df, graphWidth):
    filename = df.dataframeName
    df = df.dropna('columns') # drop columns with NaN
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    if df.shape[1] < 2:
        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')
        return
    corr = df.corr()
    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')
    corrMat = plt.matshow(corr, fignum = 1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.title(f'Корреляционная матрица для {filename}', fontsize=15)
    plt.show()
# Графики рассеивания и плотности распределения
def plotScatterMatrix(df, plotSize, textSize):
    df = df.select_dtypes(include =[np.number]) # оставить только числовые столбцы
    # Удаляем строки и столбцы, из-за которых df будет единственным
    df = df.dropna('columns')
    df = df[[col for col in df if df[col].nunique() > 1]] # сохранить столбцы, в которых есть более 1 уникального значения
    columnNames = list(df)
    if len(columnNames) > 10: # уменьшить количество столбцов для матричной инверсии графиков плотности ядра
        columnNames = columnNames[:10]
    df = df[columnNames]
    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')
    corrs = df.corr().values
    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):
        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)
    plt.suptitle('График рассеивания и плотности распределения')
    plt.show()
toplivo = pd.read_excel(open('топливо.xlsx', 'rb'))
nRow, nCol = toplivo.shape
print(f'в файле {nRow} строк и {nCol} столбцов')
toplivo.head()
toplivo.describe()
toplivo.info()
toplivo.dataframeName = 'топливо.xlsx'
plotCorrelationMatrix(toplivo, 10)
def clean_numeric(x):
    try:
        return bool(x)
    except:
        return np.nan
def clean_numeric1(x):
    try:
        return int(x)
    except:
        return np.nan
toplivo['двигатель 0, кг'] = toplivo['двигатель 0, кг'].apply(clean_numeric1).astype('int')
toplivo['земля, кг'] = toplivo['земля, кг'].apply(clean_numeric1).astype('int')
toplivo['топливо, кг(промежуточное'] = toplivo['топливо, кг(промежуточное'].apply(clean_numeric1).astype('int')
toplivo['с грузом, кг'] = toplivo['с грузом, кг'].apply(clean_numeric1).astype('int')
toplivo['с ПОС, кг'] = toplivo['с ПОС, кг'].apply(clean_numeric1).astype('int')
toplivo['итог топливо, кг'] = toplivo['итог топливо, кг'].apply(clean_numeric1).astype('int')
toplivo['МСА '] = toplivo['МСА '].apply(clean_numeric).astype('int')
toplivo['Двигатель'] = toplivo['Двигатель'].apply(clean_numeric).astype('int')
toplivo['Включение ПОС'] = toplivo['Включение ПОС'].apply(clean_numeric).astype('int')
toplivo['груз'] = toplivo['груз'].apply(clean_numeric).astype('int')
sns.countplot(data=toplivo, x="дальность полета, м", hue='Ветер ')
toplivo['высота, м'].hist(figsize=(9, 6), color='purple', alpha=0.9);
from sklearn.model_selection import train_test_split
X = toplivo.drop('итог топливо, кг', axis=1)
y = toplivo['итог топливо, кг']
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=65, random_state=53)
X1 = toplivo[['высота, м','топливо, кг(промежуточное']]
fig, axs = plt.subplots(1, 2, figsize=(20, 5))
for i, col in enumerate(X1.columns):
    axs[i].scatter(X1[col], y)
    axs[i].set_xlabel(col)
    axs[i].set_ylabel('топливо при ветре, кг ')
plt.show()
# Применение линейной регрессии
# Обучение модели на обучающих данных
from sklearn import linear_model
from sklearn.metrics import r2_score
lin_reg = linear_model.LinearRegression()
lin_reg.fit(X_train, y_train)
 
# Оценка эффективности модели на тестовых данных
y_pred = lin_reg.predict(X_test)
r2 = r2_score(y_test, y_pred)
print('R2 score:', r2)
degree = 1
max_r2 = 0
best_degree = 1
from sklearn.preprocessing import PolynomialFeatures
while True:
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
 
    model = linear_model.LinearRegression()
    model.fit(X_train_poly, y_train)
 
    y_pred = model.predict(X_test_poly)
    r2 = r2_score(y_test, y_pred)
 
    print(degree, r2)
 
    if r2 > max_r2:
        max_r2 = r2
        best_degree = degree
    elif r2 < 0.8:
        break
 
    degree += 1
