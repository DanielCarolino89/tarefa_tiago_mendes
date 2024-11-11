
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

#import treinar e testar arquivos CSV
movie = pd.read_csv("ml-latest/movies.csv")
ratings = pd.read_csv("ml-latest/ratings.csv")
links = pd.read_csv("ml-latest/links.csv")
tags = pd.read_csv("ml-latest/tags.csv")

#dê uma olhada nos dados de treinamento imprimir 
print(movie.shape)
print(ratings.shape)
print(links.shape)
print(tags.shape,end="\n\n")
m=pd.read_csv("ml-latest/movies.csv")
movie.head(3)
links.head(3)
ratings.head(3)
tags.head(3)

#obter uma lista dos recursos do conjunto de dados
print("Movie : ", movie.columns,end="\n\n")
print("Rating : ", ratings.columns,end="\n\n")
print("Links : ", links.columns,end="\n\n")
print("Tags : ", tags.columns,end="\n\n")

movie.info()
ratings.info()
tags.info()


# Removendo a coluna de carimbo de data/hora do arquivo de classificações e tags
ratings.drop(columns='timestamp',inplace=True)
tags.drop(columns='timestamp',inplace=True)

#Extraindo o ano do título
movie['Year'] = movie['title'].str.extract('.*\((.*)\).*',expand = False)

#Plotando um gráfico com o número de filmes de cada ano correspondente ao seu ano
plt.plot(movie.groupby('Year').title.count())
plt.show()
a=movie.groupby('Year').title.count()
print('Max No.of Movies Relesed =',a.max())
for i in a.index:
    if a[i] == a.max():
        print('Year =',i)
a.describe()

# Separe a coluna Gêneros e codifique-os com o método One-Hot-Encoding.
genres=[]
for i in range(len(movie.genres)):
    for x in movie.genres[i].split('|'):
        if x not in genres:
            genres.append(x)  

len(genres)
for x in genres:
    movie[x] = 0
for i in range(len(movie.genres)):
    for x in movie.genres[i].split('|'):
        movie[x][i]=1
movie

movie.drop(columns='genres',inplace=True)
movie.sort_index(inplace=True)

x={}
for i in movie.columns[4:23]:
    x[i]=movie[i].value_counts()[1]
    print("{}    \t\t\t\t{}".format(i,x[i]))

plt.bar(height=x.values(),x=x.keys())
plt.show()

#Adicione uma coluna rating no DF do filme e atribua a ela a Classificação Média do Filme para esse Filme.
x=ratings.groupby('movieId').rating.mean()
movie = pd.merge(movie,x,how='outer',on='movieId')
movie['rating'].fillna('0',inplace=True)

# Agora vamos agrupar todas as classificações com relação ao movieId e contar o número de usuários
x = ratings.groupby('movieId',as_index=False).userId.count()
x.sort_values('userId',ascending=False,inplace=True)
y = pd.merge(movie,x,how='outer',on='movieId')

y.drop(columns=[i for i in movie.columns[2:23]],inplace=True)

y.sort_values(['userId','rating'],ascending=False)

#encontre o usuário com o maior número de classificações de filmes e a classificação média desse usuário.
x = ratings.groupby('userId',as_index=False).movieId.count()
y = ratings.groupby('userId',as_index=False).rating.mean()
x = pd.merge(x,y,how='outer',on='userId')
x.describe()

for i in movie.columns[3:]:
    movie[i] = movie[i].astype(int)
    
#dividir os dados em recursos e resultados
X = movie[movie.columns[3:23]]
y = movie[movie.columns[-1]]

# Divisão dos dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1, shuffle=True)

# Criação e treinamento do modelo
model = RandomForestRegressor(n_estimators=560, random_state=42)
model.fit(X_train, y_train)

# Cálculo do erro absoluto médio no conjunto de treino
train_mae = mean_absolute_error(model.predict(X_train), y_train)
print("Mean Absolute Error (Train):", train_mae)

# Previsões no conjunto de teste
preds = model.predict(X_test)
print(preds)

# Cálculo do erro absoluto médio no conjunto de teste
test_mae = mean_absolute_error(y_test, preds)
print("Erro médio absoluto (teste):", test_mae)