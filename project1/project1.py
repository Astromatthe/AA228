import sys

import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def write_gph(dag, idx2names, filename):
    with open(filename, 'w') as f:
        for edge in dag.edges():
            f.write("{}, {}\n".format(idx2names[edge[0]], idx2names[edge[1]]))

def compute_bayes_score(G, data):
    score = 0
    # get number of nodes and assign pseudocounts of uniform prior
    n_node = data.shape[1]
    alpha = np.ones(n_node)

    # extract pseudocounts for each node
    alpha_ij = {}

    # iterate through each node in the graph
    for node in G.nodes():


    return score

def compute(infile, outfile):
    # read in csv file 
    data = pd.read_csv(infile)
    labels = data.head()

    ## K2 algorithm
    # initialize graph
    G = nx.DiGraph()
    G.add_nodes_from(labels)

    # initialize node ordering and maximum amount of parents
    order = labels
    max_parents = 3

    # construct example graph
    G.add_edge('parent1','child1')
    G.add_edge('parent2','child2') 
    G.add_edge('parent3','child3')
    G.add_edge('parent1','child2') 
    G.add_edge('parent3','child2') 
    nx.draw(G, with_labels=True)
    plt.show()

    # calculate Bayesian score
    score = compute_bayes_score(G, data)
    print("Bayesian score: ", score)
    pass


def main():    
    if len(sys.argv) == 3:
        # raise Exception("usage: python project1.py <infile>.csv <outfile>.gph")
        inputfilename = sys.argv[1]
        outputfilename = sys.argv[2]
    else:
        inputfilename = 'project1/example/example.csv'
        outputfilename = 'project1/example/example.gph'

    compute(inputfilename, outputfilename)


if __name__ == '__main__':
    main()