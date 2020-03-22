import numpy as np
import pandas as pd
from numpy import random as rd
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics.pairwise import euclidean_distances

#Preparo la simulacion
num_personas = 1000
distancia_contagiable = 0.05
step_vel = 0.01

personas = [rd.random(size=3) for j in range(num_personas)]
personas = pd.DataFrame(personas,columns = ['x','y','enfermo'])
personas.loc[:,'enfermo'] = personas.loc[:,'enfermo'].apply(lambda x:1 if x > 0.995  else 0 )


plt.ion()
plt.axis([0, 1, 0, 1])

red_patch = mpatches.Patch(color='red', label='enfermo')
blue_patch = mpatches.Patch(color='blue', label='Infectado')

for j in range(100):
    velocidades = [[rd.random(), rd.random(), 0] for j in range(num_personas)]
    velocidades = pd.DataFrame(velocidades, columns=['x', 'y', 'enfermo'])
    velocidades.loc[:,['x','y']] = velocidades.loc[:,['x','y']] - 0.5

    personas = personas + step_vel*velocidades

    personas[["x", 'y']] = personas[["x", 'y']] % 1

    contagiables_distancia = euclidean_distances(personas[['x', 'y']]) - np.identity(num_personas)
    contagiables_distancia = (contagiables_distancia < distancia_contagiable) & (contagiables_distancia != -1)

    a_contagiar = contagiables_distancia @ personas["enfermo"]
    a_contagiar = np.argwhere(a_contagiar > 0).flatten()

    personas.loc[a_contagiar,"enfermo"] = 1

    plt.axis([0, 1, 0, 1])
    plt.scatter(personas['x'], personas['y'], c=personas['enfermo'].apply(lambda x: 'red' if x==1 else 'blue'))

    plt.legend(handles=[red_patch, blue_patch], loc='upper left')
    plt.pause(0.2)
    plt.cla()

    print(personas['enfermo'].sum())