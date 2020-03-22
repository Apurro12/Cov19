import numpy as np
import pandas as pd
from numpy import random as rd
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics.pairwise import euclidean_distances

#Preparo la simulacion
num_personas = 60
distancia_contagiable = 0.1
step_vel = 0.02

personas = [rd.random(size=3) for j in range(num_personas)]
personas = pd.DataFrame(personas,columns = ['x','y','sano'])
personas.loc[:,'sano'] = personas.loc[:,'sano'].apply(lambda x:1 if x < 0.8  else 0 )


plt.ion()
plt.axis([0, 1, 0, 1])

red_patch = mpatches.Patch(color='red', label='Sano')
blue_patch = mpatches.Patch(color='blue', label='Infectado')

#keepgoing = True
#while keepgoing:
t_lapse=[]
tot_contagiados=[]
for t in range(100):
    velocidades = [[rd.random(), rd.random(), 0] for t in range(num_personas)]
    velocidades = pd.DataFrame(velocidades, columns=['x', 'y', 'sano'])
    velocidades.loc[:,['x','y']] = velocidades.loc[:,['x','y']] - 0.5

    personas = personas + step_vel*velocidades

    personas[["x", 'y']] = personas[["x", 'y']] % 1
    
    tot_contagiados.append((num_personas - personas['sano'].sum())/num_personas)
    
        
    # --Check threshold distance between people.
    contagiable_dist = euclidean_distances(personas[['x', 'y']]) - np.identity(num_personas)
    contagiable_dist = (contagiable_dist < distancia_contagiable) & (contagiable_dist != -1)

    posibilidad_contagio = np.tensordot(personas["sano"], personas["sano"], axes=0)
    posibilidad_contagio = posibilidad_contagio == 0

    a_contagiar = posibilidad_contagio & contagiable_dist
    a_contagiar = np.argwhere(a_contagiar == True)
    a_contagiar = set(a_contagiar.flatten())

    personas.loc[a_contagiar, 'sano'] = 0

    plt.subplot(1,2,1)
    plt.axis([0, 1, 0, 1])
    plt.scatter(personas['x'], personas['y'], c=personas['sano'])

    plt.legend(handles=[red_patch, blue_patch], loc='upper left')
    plt.pause(0.2)

    t_lapse.append(t)

    plt.subplot(1, 2, 2)
    plt.plot(t_lapse,tot_contagiados,'r+')
    plt.ylabel(f'frecuencia contagiados')
    plt.xlabel(f'tiempo [pasos MC]')
    plt.pause(0.2)
    #plt.cla()
