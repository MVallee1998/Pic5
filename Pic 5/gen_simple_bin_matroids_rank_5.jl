using Oscar
using IterTools
using ProgressBars
using LinearAlgebra
using ProgressMeter

function all_nonzero_binary_vectors(n)
    k = (2^n) - 1
    mat = Matrix{Int}(undef, k, n)
    for i in 1:k
        v = digits(i, base=2, pad=n)
        mat[i,:]= v
    end
    return mat
end

function find_lower_dim_matroids(list_bin_mat)
    local S_rep = Set{Matroid}()
    @showprogress for M in list_bin_mat
        for v in matroid_groundset(M)
            if v in coloops(M)
                print("is in coloop")
                continue
            end
            M1 = deletion(M,v)
            if rank(M)!=rank(M1)
                continue
            end
            is_isom=false
            for M2 in S_rep
                if is_isomorphic(M1,M2)
                    is_isom = true
                    break
                end
            end
            if is_isom==false
                push!(S_rep,M1)
            end
        end
    end
    return S_rep
end

global pic = 5

global A = matrix(GF(2), all_nonzero_binary_vectors(pic))
global m=size(A)[1]
global M0 = matroid_from_matrix_rows(A)
global S = Set{Matroid}()
global simple_bin_matroids = Dict{Int,Vector{Vector{Vector{UInt8}}}}()
global simple_bin_matroids_bin = Dict{Int,Vector{Vector{UInt32}}}()

function make_binary(L::Vector{Vector{UInt8}})
    result = Vector{UInt32}()
    for elt in L
        push!(result,reduce(|,[UInt32(UInt32(1)<<(i-1)) for i in elt]))
    end
    return sort(result)
end

push!(S,M0)
while m > pic+1
    print("number of elements ",m," number of matroids ",length(S),"\n")
    simple_bin_matroids[m]=[bases(M) for M in S]
    simple_bin_matroids_bin[m]=[make_binary(Vector{Vector{UInt8}}(bases_M)) for bases_M in simple_bin_matroids[m]]
    global S = find_lower_dim_matroids(S)
    global m-=1
end




using Serialization

# save
open("rank_5_simple_bin_mat_DB_bin.jls", "w") do io
    serialize(io, simple_bin_matroids_bin)
end







