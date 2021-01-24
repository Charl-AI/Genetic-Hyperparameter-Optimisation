# Genetic-Hyperparameter-Optimisation

## Executive Summary

When designing nerual networks, hyperparameter optimisation can be tedious and often relies on experience and guesswork from the data scientist. This project uses a genetic algorithm to optimise the hyperparameters for a simple neural network problem. It was developed to solve my machine learning coursework without me having to go through a tedious trial and error process and was optimised to run on 24 cores on a high-performance computing (HPC) cluster; the resulting architecture achieved an A+ grade in the coursework assignment.

This is simply a proof-of-concept hack and was built in less than 2 days - it is a solution to a specific problem and is not intended as a reusable framework. 

## Introduction

Neural network hyperparameter optimisation involves optimising a mixture of continuous and discrete parameters, this function is non-smooth and non-convex so gradient descent is usually a poor method. Genetic algorithms are typically useful for this kind of problem and use a biologically-inspired evolution process for optimisation:

1. Generate an initial population
2. Evaluate fitness of all individuals in population
3. Apply selection process to generate a breeding pool
4. Breed individuals from the pool and apply mutations to offspring
5. Repeat steps 2-5

This was applied to a task involving predicting whether a catapault would hit a target, based on 6 design parameters of the catapaults. 4000 rows of data were available for training and validation (the final model was tested on an unseen dataset as part of the coursework).

## Implementation

This solution optimises 3 hyperparameters or 'genes': learning rate of network, depth of network (number of hidden layers), width of network (nodes per layer). The population size is 100 models and the algorithm ran for 10 generations. Speciation was implemented, consisting of K-means clustering being used to split the population into 5 species at the end of each generation. The inclusion of speciation preserves genetic diversity so allows multiple different solutions to be found.

1. The population is initialised by creating 100 individuals with genes drawn from a uniform distribution across the hyperparameter space. The hyperparameter space was selected by hand to cover 2-20 hidden layers and nodes per layer and 0.0001-0.02 learning rate (in future this should be done automatically and not by hand).

2. Each model trains on CPU and training was parallelised using the python multiprocessing module. This provided a significant speed improvement, as 24 models were able to be trained simultaneously.

3. The selection algorithm involved creating a breeding pool from the best performing half of the species and randomly selecting individuals to breed from that pool. When the number of individuals in the species is odd, the best performing individual is immediately cloned into the next generation (a process known as elitism).

4. Breeding takes two parents and produces two children, where each child is a clone of one of the parents with an 80 % chance of inheriting an alternate allele from the other parent (a process known as crossover). In each child, each gene has a 20 % chance of mutating and the child has a 2 % chance of undergoing 'massive catastrophic mutation'. Standard mutation involves selecting the new value of each gene from a Gaussian distribution, centered on the previous value (note: the variance of these distributions has been hand-crafted, so in future should be automated). 'Massive catastrophic mutation' involves selecting new values for all genes from a uniform distribution covering the full hyperparameter space.

5. The algorithm was repeated for 10 generations, so 1000 models were trained in total.

## Results
![alt text](Plot_all individuals.png)
Genetic-Hyperparameter-Optimisation/Plot_all individuals.png

## Discussion 

This project succeded in automating some of my machine learning coursework and had the brilliant side effect of teaching me even more machine learning methods! I am happy I completed this project, however it has illuminated one of the key reasons why genetic algorithms are not often used for hyperparameter optimisation: computational cost. Finding the final solution involved training as many as 1000 neural networks - fortunately, this was possible in this case, due to the realtive simplicity of the models, short training times and accessibility of HPC facilities but it would simply not be possible in more difficult (and arguably more important) tasks such as convolutional neural networks.

## Future Work

Some of the hand-crafted and arbitrary decisions made are described in the implementation section, so these should ideally be automated. The project would need significant refactoring in order to be converted into a reusable framework and is of questionable use outside of a handful of specific cases. For this reason it is diffucult to recommend that the project be followed up on in the future
