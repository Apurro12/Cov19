#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
from numpy import random as rd
import pandas as pd

def init_ppl(Tot_nbr_ppl,Init_prob_sick=0.995):
    """
    Initializes the df of ppl in the simulation.
    if 'enfermo' == 1 -> means true -> it's a sick person

    #Args:: Tot_nbr_ppl, Init_prob_sick (default = 0.995 == 99,5% not sick)

    #Returns:: DataFrame of people
    """
    ppl = [rd.random(size=3) for j in range(num_ppl)]
    ppl = pd.DataFrame(ppl,columns = ['x','y','enfermo'])
    ppl.loc[:,'enfermo'] = ppl.loc[:,'enfermo'].apply(
            lambda x:1 if x > Init_prob_sick  else 0
            )
    
    return ppl

def update_vel(Tot_nbr_ppl):
    """
    Compute the velocity of each person in the simulation at each time and returns a generator

    #Args::Tot_nbr_ppl

    #Returns:: a velocity generator object.
    """
    vel = [[rd.random(), rd.random(), 0] for t in range(Tot_nbr_ppl)]
    vel = pd.DataFrame(vel, columns=['x', 'y', 'enfermo'])
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

    yield Ppl
