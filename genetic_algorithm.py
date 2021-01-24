from numpy.random import randint,uniform,normal
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
import multiprocessing as mp
from functools import partial
import numpy as np
from neural_network import *

def initialise_population(pop_size):
    # create a Nx5 array, holding all the models in the population, as well as their parameters, species and results
    population = np.empty([pop_size,5])
    
    for i in range(pop_size):
        nodes_per_layer = randint(2,20)
        layers = randint(2,20)
        learning_rate = uniform(0.0001,0.02) 

        population[i,0] = nodes_per_layer
        population[i,1] = layers
        population[i,2] = learning_rate
        population[i,3] = None # this will store the val loss once the model is trained
        population[i,4] = None # this will store the species
        
    return population

def speciation(population, num_species):
    kmeans = KMeans(n_clusters = num_species)
    input_data = population[:,0:3]
    scaler = MinMaxScaler()
    input_data = scaler.fit_transform(input_data)
    species = kmeans.fit_predict(input_data)
    population[:,4] = species
    #print(input_data)
    return population



# this function takes the population of networks, fits them to the data and records the minimum val loss
def test_population(population, x,y,pool):

    nodes = population[:,0]
    layers = population[:,1]
    learning = population[:,2]
    
    train = partial(train_network,x=x,y=y)
    #pool = mp.Pool(mp.cpu_count())
    #population[:,3] = pool.map(train, list(zip(nodes,layers,learning)))
    population[:,3] = pool.starmap(train, population,1)
    #population[:,3] = [pool.apply(train, (population[i,:])) for i in range(len(population))]
    #pool.close()
    #pool.join()
    return population

def breed(model1,model2):
    child_1 = model1
    child_2 = model2
    
    # 80% chance of crossover
    if randint(1,11) <= 8:
        crossover_point = randint(0,2)
        child_1[crossover_point] = model2[crossover_point]
        child_2[crossover_point] = model1[crossover_point]
    
    # 2% chance of one child undergoing massive catastrophic mutation
    if randint(1,101) <= 1:
        child_1[0] = randint(2,20) # mutate nodes gene
        child_1[1] = randint(2,20) # mutate layers gene
        child_1[2] = uniform(0.0001,0.02) # mutate learning gene 
    elif randint(1,101) <= 1:
        child_2[0] = randint(2,20) # mutate nodes gene
        child_2[1] = randint(2,20) # mutate layers gene
        child_2[2] = uniform(0.0001,0.02) # mutate learning gene
    
    # else each gene has 20 % chance of Gaussian mutation (centred on prev value)
    else:
        if randint(1,11) <= 2:
            child_1[0] = round(normal(child_1[0], 1.5)) 
        if randint(1,11) <= 2:
            child_1[1] = round(normal(child_1[1], 1.5)) 
        if randint(1,11) <= 2:
            child_1[2] = normal(child_1[2], 0.002) 
        if randint(1,11) <= 2:
            child_2[0] = round(normal(child_1[0], 1.5)) 
        if randint(1,11) <= 2:
            child_2[1] = round(normal(child_1[1], 1.5)) 
        if randint(1,11) <= 2:
            child_2[2] = normal(child_1[2], 0.002) 
    
    return child_1, child_2
        
def generate_next_gen(population, num_species):
    # sort population by species
    population = population[population[:,4].argsort()]
    # split population into arrays for different species
    species = np.array_split(population, np.where(np.diff(population[:,4]))[0]+1) 
    # create array for next generation
    next_gen = np.zeros((len(population),5))
    
    position_in_pop = 0
    for i in range(num_species):
        # create temporary array to store next gen of the species
        new_species = np.zeros((len(species[i]),5))
        # sort species by loss
        species[i] = species[i][species[i][:,3].argsort()]
        # if the number of individuals in the species is odd, the best individual is allowed to clone themself into the next gen
        if len(species[i]) % 2 !=0:
            new_species[len(species[i])-1,:] = species[i][0,:]
        # if only one member of species, breed asexually but only allow one child
        if len(species[i]) == 1:
            new_species[0,:], new_species[0,:] = breed(species[i][0,:],species[i][0,:])

        # else breed random samples from best half of species until we have filled the new
        else:
            for j in range(0,len(species[i])-1,2):
                new_species[j,:], new_species[j+1,:] = breed(species[i][randint(0,np.ceil(len(species[i])/2)),:],species[i][randint(0,np.ceil(len(species[i])/2)),:])
        
        next_gen[position_in_pop:position_in_pop+len(new_species),:] = new_species
        position_in_pop += len(species[i])
        
        next_gen[:,3] = None
        next_gen[:,4] = None
        
    return(next_gen)