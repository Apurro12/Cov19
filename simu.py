import sys, argparse
import numpy as np
import pandas as pd
from numpy import random as rd
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics.pairwise import euclidean_distances


def Config_Parse():
    """
    Set all the configuration to your parser object.
    
    # Args::None

    # Returns::parser object.
    """
    parser = argparse.ArgumentParser('Covid19')
    parser.add_argument('-nP', '--NbrPpl', required=True, help='Number of people in simulation' )
    parser.add_argument('-rS', '--R_Spread', required=True, help='Radious of spread' )
    parser.add_argument('-lT', '--LastTime', required=True, help='Total lenght of simulation' )
    parser.add_argument('-I', '--Input', required=False, help='<Input folder or file/s>' )
    parser.add_argument('-O', '--Output', required=False, help='<Output folder or files/s>')
    parser.add_argument('-D','--Debug', required=False, help='Debug flag', action='store_true')
    parser.add_argument('-M','--MaxEvents', required=False, help='Set maximum of events. Default -1 == all', type=int, default=-1)
    return parser


def main(argv):
    parser = Config_Parse()
    args = parser.parse_args()

    
    #Preparo la simulacion
    num_ppl = int(args.NbrPpl)
    distancia_contagiable = float(args.R_Spread)
    step_vel = 0.02
    lastTime= int(args.LastTime)

    ppl = [rd.random(size=3) for j in range(num_ppl)]
    ppl = pd.DataFrame(ppl,columns = ['x','y','enfermo'])
    ppl.loc[:,'enfermo'] = ppl.loc[:,'enfermo'].apply(lambda x:1 if x > 0.995  else 0 )


    plt.ion()
    plt.axis([0, 1, 0, 1])

    red_patch = mpatches.Patch(color='red', label='enfermo')
    blue_patch = mpatches.Patch(color='blue', label='Infectado')

    #keepgoing = True
    #while keepgoing:
    t_lapse=[]
    tot_contagiados=[]
    for t in range(lastTime):
        velocidades = [[rd.random(), rd.random(), 0] for t in range(num_ppl)]
        velocidades = pd.DataFrame(velocidades, columns=['x', 'y', 'enfermo'])
        velocidades.loc[:,['x','y']] = velocidades.loc[:,['x','y']] - 0.5

        ## -- THIS 2 COULD BE A "WALK" FUNCTION
        # --Update people position
        ppl = ppl + step_vel*velocidades
        
        # --Set periodic boundary conditions
        ppl[["x", 'y']] = ppl[["x", 'y']] % 1
        
        tot_contagiados.append(ppl['enfermo'].sum()/num_ppl)
        
            
        # --Check threshold distance between people.
        contagiable_dist = euclidean_distances(ppl[['x', 'y']]) - np.identity(num_ppl)
        contagiable_dist = (contagiable_dist < distancia_contagiable) & (contagiable_dist != -1)

        #posibilidad_contagio = np.tensordot(ppl["sano"], ppl["sano"], axes=0)
        #posibilidad_contagio = posibilidad_contagio == 0

        a_contagiar = contagiable_dist @ ppl["enfermo"]
        a_contagiar = np.argwhere(a_contagiar > 0).flatten()


        ppl.loc[a_contagiar,"enfermo"] = 1
        print(ppl['enfermo'].sum())
        t_lapse.append(t)
    
        if ppl['enfermo'].sum()==ppl.shape[0]: break

        """
                     Aca empiezan los plots              
                     No actualizar variables             
                     De aca en adelante solo
                     plotear
        """
        
        plt.subplot(1,2,1)
        plt.tight_layout()
        plt.axis([0, 1, 0, 1])
        plt.scatter(
                ppl['x'],
                ppl['y'],
                c=ppl['enfermo'].apply(lambda x: 'red' if x==1 else 'blue')
                )

        plt.legend(handles=[red_patch, blue_patch], loc='upper left')

        plt.subplot(1,2,2)
        plt.tight_layout()
        plt.plot(t_lapse,tot_contagiados,'r+')
        plt.ylabel(f'frecuencia contagiados')
        plt.xlabel(f'tiempo [pasos MC]')
        plt.pause(0.2)

        plt.subplot(1,2,1)
        plt.cla()
        plt.subplot(1,2,2)
        plt.cla()

    return

if __name__ == '__main__':
    main(sys.argv[1:])
