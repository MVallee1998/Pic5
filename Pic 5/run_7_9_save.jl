include("algo_parallel_v.jl")


mat_DB_bin = open("rank_5_mat_DB_bin.jls", "r") do io
    deserialize(io)
end

iso_DB = open("rank_5_iso_DB_bin_all_v.jls", "r") do io
    deserialize(io)
end

pseudo_manifolds_DB = Dict{Int,Vector{Set{BitVector}}}()

mmax=9


build_finalDB_single_v!(pseudo_manifolds_DB,mat_DB_bin,iso_DB,mmax)

open("pseudo_manifolds_DB_7-9.jls", "w") do io
    serialize(io, pseudo_manifolds_DB)
end