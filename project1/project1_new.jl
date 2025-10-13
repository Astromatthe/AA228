using Graphs
using GraphPlot
using Printf
using DataFrames
using CSV
using LinearAlgebra
using SpecialFunctions

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

function bayesian_score(r,G,D)
    score = 0.0
    n = nv(G)
    alpha, M = create_pseudocount_matrices(r, G, D)
    q = [size(alpha[i],1) for i in 1:n]
    for i in 1:nv(G)
        alpha_ij0 = sum(alpha[i], dims=2)
        m_ij0 = sum(M[i], dims=2)
        for j in 1:q[i]
            for k in 1:r[i]
                score += sum(loggamma.(alpha[i][j,k] + M[i][j,k]))  # nominator in r sum
                score -= sum(loggamma.(alpha[i][j,k]))  # denominator in r sum
            end
        end
        score += sum(loggamma.(alpha_ij0))  # nominator in q sum
        score -= sum(loggamma.(alpha_ij0 + m_ij0)) # denominator in q sum
    end
    return score
end

function create_pseudocount_matrices(r, G, D)
    n = nv(G)   # get number of nodes
    q = zeros(1,n)
    # number of instantiations of parents
    for i in 1:n
        par = inneighbors(G,i)
        if isempty(par)
            q[i] = 1
        else
            q[i] = prod(r[par])
        end
    end
    # create prior
    alpha = Vector{Matrix{Int64}}(undef,n)
    for i in 1:n
        alpha[i] = ones(Int64(q[i]), r[i])
    end

    ## create pseudocount matrix
    M = [zeros(Int64(q[i]), r[i]) for i in 1:n]
    for set in eachcol(D)
        for i in 1:n
            k = set[i] # recorded instantion of node i in data set
            par = inneighbors(G,i)
            if isempty(par)
                j = 1
            else
                j = 1   # instantiation table starts at 1
                inc_par = zeros(1, length(par)) # increment of index per parent
                for p in 1:length(par)
                    inc_par[p] = prod(r[par[1:p-1]])    # ordering here: equal instantations of last parent first, then second to last, ...
                    j += (set[par[p]]-1) * inc_par[p]
                end
            end
            M[i][Int64(j),k] += 1.0
        end
    end
    return alpha, M
end

function K2_search(order, var_ids, r, D)
    G = SimpleDiGraph(length(var_ids))
    score_global = -Inf
    k = 0
    for i in order[2:end]
        k += 1
        score_global = bayesian_score(r, G, D)
        while true
            score_best_iter = -Inf
            j_best = 0
            for j in order[1:k]
                if !has_edge(G, j, i)
                    add_edge!(G, j, i)
                    score_iter = bayesian_score(r, G, D)
                    if score_iter > score_best_iter
                        j_best = j
                        score_best_iter = score_iter
                    end
                    rem_edge!(G, j, i)
                end
            end
            if score_best_iter > score_global
                score_global = score_best_iter
                add_edge!(G, j_best, i)
            else 
                break
            end
        end
    end
    return G, score_global
end

function compute(infile, outfile)

    # read in csv file
    data_raw = CSV.read(infile, DataFrame)
    # create variables based on names in csv files and maximum value of each node
    var_names = names(data_raw)
    var_ids = 1:length(var_names)
    r = [maximum(unique(data_raw[:,i])) for i in 1:size(data_raw,2)] # get number of possible instantiations for each node
    D = transpose(Matrix(data_raw)) # data matrix has column for each set, one line per node (same as in book)

    # initialize example graph

    order = var_ids
    score = 0
    G, score = K2_search(order,var_ids,r,D)
    show(score)
    p = gplot(G, nodelabel=var_names) # plot graph
    
    ## output
    write_gph(G, Dict(i => var_names for i in 1:length(var_names)), outfile)
end

if length(ARGS) == 2
    inputfilename = ARGS[1]
    outputfilename = ARGS[2]
else
    dataset = "small";
    inputfilename = "project1/data/" * dataset * ".csv"
    outputfilename = "project1/output/" * dataset * ".gph"
end

@time compute(inputfilename, outputfilename)