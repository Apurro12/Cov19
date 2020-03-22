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


#keepgoing = True
#while keepgoing:
t_lapse=[]
tot_contagiados=[]
for t in range(100):
    velocidades = [[rd.random(), rd.random(), 0] for t in range(num_personas)]
    velocidades = pd.DataFrame(velocidades, columns=['x', 'y', 'enfermo'])

    velocidades.loc[:,['x','y']] = velocidades.loc[:,['x','y']] - 0.5

    personas = personas + step_vel*velocidades

    personas[["x", 'y']] = personas[["x", 'y']] % 1
    
    tot_contagiados.append(personas['enfermo'].sum()/num_personas)
    
        
    # --Check threshold distance between people.
    contagiable_dist = euclidean_distances(personas[['x', 'y']]) - np.identity(num_personas)
    contagiable_dist = (contagiable_dist < distancia_contagiable) & (contagiable_dist != -1)

    a_contagiar = contagiable_dist @ personas["enfermo"]
    a_contagiar = np.argwhere(a_contagiar > 0).flatten()


    personas.loc[a_contagiar,"enfermo"] = 1
    print(personas['enfermo'].sum())
    t_lapse.append(t)


    """"""""""""""""""""""""""""""""""""""""""""""""
    """""""""""""Aca empiezan los plots """""""""""""
    """""""""""""No actualizar variables"""""""""""""
    """""""""""""De aca en adelante solo plotear  """
    """"""""""""""""""""""""""""""""""""""""""""""""
    plt.subplot(1,2,1)
    plt.tight_layout()
    plt.axis([0, 1, 0, 1])
    plt.scatter(personas['x'], personas['y'], c=personas['enfermo'].apply(lambda x: 'red' if x==1 else 'blue'))

    plt.legend(handles=[red_patch, blue_patch], loc='upper left')

    plt.subplot(1, 2, 2)
    plt.tight_layout()
    plt.plot(t_lapse,tot_contagiados,'r+')
    plt.ylabel(f'frecuencia contagiados')
    plt.xlabel(f'tiempo [pasos MC]')
    plt.pause(0.2)

