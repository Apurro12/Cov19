#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
from numpy import random as rd
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances

def init_ppl(Tot_nbr_ppl,Init_prob_healthy=0.995):
    """
    Initializes the df of ppl in the simulation.
    if 'enfermo' == 1 -> means true -> it's a sick person

    #Args:: Tot_nbr_ppl, Init_prob_healthy (default = 0.995 == 99,5% not sick)

    #Returns:: DataFrame of people
    """
    ppl = [rd.random(size=4) for j in range(Tot_nbr_ppl)]
    ppl = pd.DataFrame(ppl,columns = ['x','y','enfermo','prob_contagio'])
    ppl.loc[:,'enfermo'] = ppl.loc[:,'enfermo'].apply(
            lambda x:1 if x > Init_prob_healthy  else 0
            )
     
    return ppl

def update_vel(Tot_nbr_ppl):
    """
    Compute the velocity of each person in the simulation at each time and returns a generator
    Note: velocity df has to have the SAME dimensions as ppl df!

    #Args::Tot_nbr_ppl

    #Returns:: a velocity generator object.
    """
    vel = [[rd.random(), rd.random(), 0, 0] for t in range(Tot_nbr_ppl)]
    vel = pd.DataFrame(vel, columns=['x', 'y', 'enfermo','prob_contagio'])
    vel.loc[:,['x','y']] = vel.loc[:,['x','y']] - 0.5
    
    yield vel

def walk(Ppl,Tot_nbr_ppl,Step_vel):
    """
    Function to update the position of each person acordingly to a step velocity in a Random Walk

    #Args::
        Ppl: the DataFrame that holds positions
        Tot_nbr_ppl: total amount of people in simulation
        Step_vel: step to update the velocity of each person

    #Returns:: Ppl generator object
    """
    Vel = next(update_vel(Tot_nbr_ppl))
    # --Update people position
    Ppl = Ppl + Step_vel*Vel
    # --Set periodic boundary conditions
    Ppl[["x", 'y']] = Ppl[["x", 'y']] % 1

    # --Update people chance to get sick
    Ppl.loc[:,'prob_contagio']=[rd.random() for x in range(Tot_nbr_ppl)]


    yield Ppl
    
def spread_sickness(Ppl,Tot_nbr_ppl,R_spread,Rate_spread):
    # --Check threshold distance between people.
    dist_to_sick = euclidean_distances(Ppl[['x', 'y']]) - np.identity(Tot_nbr_ppl)
    dist_to_sick = (dist_to_sick < R_spread) & (dist_to_sick != -1)
    # --People in the range of sick people will have a 1.0 for close_to_sick
    close_to_sick = dist_to_sick @ Ppl["enfermo"]
    # --Get all indexes where people can get sick
    close_to_sick = np.argwhere(close_to_sick > 0).flatten()
    candid_sick = Ppl.loc[close_to_sick] 
    # --Change the status of those close to sick people that fullfill the chance to get sick
    new_sick = candid_sick[(candid_sick['prob_contagio'] >= Rate_spread) == True]
    Ppl.loc[new_sick.index,"enfermo"] = 1

    yield Ppl

def heal(ppl,Tot_nbr_ppl,prob_healing):
    """
    Function to heal a infected person

    #Args::
        Ppl: the DataFrame that holds positions
        Tot_nbr_ppl: total amount of people in simulation
        P_Healing: probability of heal of each person (varies timestep to timestep)

    #Returns:: None the update is inplace
    """
    ppl_to_heal = np.random.random(Tot_nbr_ppl) < prob_healing
    ppl_to_heal = ppl_to_heal & (ppl['enfermo'] == 1)
    ppl.loc[ppl_to_heal,'enfermo'] = 0

