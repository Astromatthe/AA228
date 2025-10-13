using Graphs
using GraphPlot
using Printf
using DataFrames
using CSV
using LinearAlgebra
using SpecialFunctions

struct Variable
    name::String
    r::Int
end

struct BayesianNetwork
    vars::Vector{Variable}
    G::SimpleDiGraph{Int64}
end

struct K2Search
    ordering::Vector{Int64}
end

struct LocalDirectedGraphSearch
    G::SimpleDiGraph{Int64}
    k_max::Int64
end

"""
    write_gph(G::DiGraph, idx2names, filename)

Takes a DiGraph, a Dict of index to names and a output filename to write the graph in `gph` format.
"""
function write_gph(G::DiGraph, idx2names, filename)
    open(filename, "w") do io
        for edge in edges(G)
            @printf(io, "%s,%s\n", idx2names[src(edge)], idx2names[dst(edge)])
        end
    end
end

function sub2ind(siz, x)
    k = vcat(1, cumprod(siz[1:end-1]))
    return dot(k, x .- 1) + 1
end

function statistics(vars, G, D)
    n = size(D,1)   # get number of nodes
    r = [vars[i].r for i in 1:n]    # get number of possible instatiations for each node
    q = [prod([r[j] for j in inneighbors(G, i)]) for i in 1:n] # get number of instantiations of parents
    M = [zeros(q[i], r[i]) for i in 1:n]
    for o in eachcol(D)     # each column is a dataset
        for i in 1:n
            k = o[i]
            parents = inneighbors(G, i)
            j = 1
            if !isempty(parents)
                j = sub2ind(r[parents], o[parents])
            end
            M[i][j,k] += 1.0
        end
    end
    return M
end

function prior(vars, G)
    n = length(vars)
    r = [vars[i].r for i in 1:n]
    q = [prod([r[j] for j in inneighbors(G, i)]) for i in 1:n]
    return [ones(q[i], r[i]) for i in 1:n]
end

function bayesian_score_component(M,alpha)
    p = sum(loggamma.(alpha + M))
    p -= sum(loggamma.(alpha))
    p += sum(loggamma.(sum(alpha, dims = 2)))
    p -= sum(loggamma.(sum(alpha, dims = 2) + sum(M, dims = 2)))
    return p
end

function bayesian_score(vars,G,D)
    n = length(vars)
    M = statistics(vars, G, D) # create psuedocount matrices for all nodes
    alpha = prior(vars, G)  # create uniform prior
    return sum(bayesian_score_component(M[i], alpha[i]) for i in 1:n)
end

function K2_fit(method::K2Search, vars, D)
    G = SimpleDiGraph(length(vars))
    for (k,i) in enumerate(method.ordering[2:end])
        y = bayesian_score(vars, G, D)
        while true
            y_best, j_best = -Inf, 0
            for j in method.ordering[1:k]
                if !has_edge(G, j, i)
                    add_edge!(G, j, i)
                    y_new = bayesian_score(vars, G, D)
                    if y_new > y_best
                        y_best, j_best = y_new, j
                    end
                    rem_edge!(G, j, i)
                end
            end
            if y_best > y
                y = y_best
                add_edge!(G, j_best, i)
            else 
                break
            end
        end
    end
    return G
end

function rand_graph_neighbor(G)
    n = nv(G)
    i = rand(1:n)
    j = mod1(i + rand(2:n)-1,n)
    G_new = copy(G)
    has_edge(G, i, j) ? rem_edge!(G_new, i, j) : add_edge!(G_new, i, j)
    return G_new
end

function LDGS_fit(method::LocalDirectedGraphSearch, vars, D)
    G = method.G
    y = bayesian_score(vars, G, D)
    for k in 1:method.k_max
        G_new = rand_graph_neighbor(G)
        y_new = is_cyclic(G_new) ? -Inf : bayesian_score(vars, G_new, D)
        if y_new > y
            G, y = G_new, y_new
        end
    end
    return G
end

function compute(infile, outfile)
    # WRITE YOUR CODE HERE
    # FEEL FREE TO CHANGE ANYTHING ANYWHERE IN THE CODE
    # THIS INCLUDES CHANGING THE FUNCTION NAMES, MAKING THE CODE MODULAR, BASICALLY ANYTHING

    # read in csv file
    data_raw = CSV.read(infile, DataFrame)
    # create variables based on names in csv files and maximum value of each node
    vars = [Variable(names(data_raw)[i], maximum(unique(data_raw[:,i]))) for i in 1:size(data_raw,2)]
    D = transpose(Matrix(data_raw)) # data matrix has column for each set, one line per node (same as in book)

    # initialize example graph
    G = SimpleDiGraph(length(vars))
    add_edge!(G, 1, 2)
    add_edge!(G, 3, 4)
    add_edge!(G, 5, 6)
    add_edge!(G, 1, 4)
    add_edge!(G, 5, 4)
    score = bayesian_score(vars, G, D)

    # start with empty graph
    G = SimpleDiGraph(length(vars))
    # test K2 search
    order = vars; # use order in csv file for testing
    K2 = K2Search([findfirst(x -> x == v, vars) for v in order]) # create ordering based on order in csv file
    G_K2 = K2_fit(K2, vars, D) # fit graph to data
    bayesian_score(vars, G_K2, D) |> show # show score of fitted graph
    # test local directed graph search
    #LDGS = LocalDirectedGraphSearch(G, 100) 
    #G_LDGS = LDGS_fit(LDGS, vars, D) # fit graph
    p = gplot(G_K2, nodelabel=vars .|> x -> x.name) # plot graph

    ## output
    # write_gph(G_LDGS, Dict(i => vars[i].name for i in 1:length(vars)), outfile) # write to gph file
end

if length(ARGS) == 2
    inputfilename = ARGS[1]
    outputfilename = ARGS[2]
else
    inputfilename = "project1/example/example.csv"
    outputfilename = "project1/example/out.gph"
end

compute(inputfilename, outputfilename)
