import matplotlib.pyplot as plt
import numpy as np
from genetic_algorithm import *
from neural_network import *

def mscatter(x,y,z, ax=None, m=None, **kw):
    import matplotlib.markers as mmarkers
    ax = ax or plt.gca()
    sc = ax.scatter(x,y,z,**kw)
    if (m is not None) and (len(m)==len(x)):
        paths = []
        for marker in m:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(
                        marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
    return sc

# generates list of markers corresponding to each species, currently supports up to 5 species
def generate_markers(population):
    markers = []
    for i in range(len(population)):
        if population[i,4] == 0:
            markers.append("o")
        elif population[i,4] == 1:
            markers.append("s")
        elif population[i,4] == 2:
            markers.append("P")
        elif population[i,4] == 3:
            markers.append("d")
        elif population[i,4] == 4:
            markers.append("*")
        else:
            markers.append("v")
    return markers

def plot_population(population):
    fig = plt.figure(figsize=(13,10))
    ax = fig.add_subplot(111, projection='3d')

    xs = population[:,0]
    ys = population[:,1]
    zs = population[:,2]
    col = population[:,3]

    markers = generate_markers(population)
            
    sc=  mscatter(xs,ys,zs,ax=ax,m=markers,c=col,cmap="Spectral",s=50)    
    #sc = ax.scatter(xs, ys, zs, c=col,cmap="Spectral", marker=markers)
    ax.set_xlabel("Nodes Per Layer")
    ax.set_ylabel("Number of Layers")
    ax.set_zlabel("Learning Rate")
    
    ax.set_xlim3d(0, 20)
    ax.set_ylim3d(0,20)
    ax.set_zlim3d(0.0001,0.02)
    plt.colorbar(sc)
    plt.savefig("gen9.png",dpi=500)
    
def display_generation(gen_number):
    population = np.genfromtxt("generation " + str(gen_number))
    population = population[population[:,3].argsort()]
    
    for i in range(len(population)):
        if population[i,3] > 1:
            population[i,3] = None
    print(population[0:5,:])
    plot_population(population)
    
def display_all_generations(num_gens, num_species):
    
    population = np.genfromtxt("generation 0")
    if num_gens > 1:
        for i in range(1,num_gens):
            gen = np.genfromtxt("generation " + str(i))
            population = np.concatenate((population,gen))
    
    population = speciation(population, num_species)
    population = population[population[:,3].argsort()]
    
    #remove outliers which might ruin the color bar
    for i in range(len(population)):
        if population[i,3] > 1:
            population[i,3] = None
    np.savetxt("Test 1",population)
    print(population[0:5,:])
    plot_population(population)