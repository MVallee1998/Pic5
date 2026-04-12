include("algo_parallel_gray_low_mem.jl")


mat_DB_bin = open("rank_5_mat_DB_bin.jls", "r") do io
    deserialize(io)
end

iso_DB = open("rank_5_iso_DB_bin_all_v.jls", "r") do io
    deserialize(io)
end



pseudo_manifolds_DB = open("pseudo_manifolds_DB_7-9.jls", "r") do io
    deserialize(io)
end

pseudo_manifolds_DB[10] = Vector{Set{BitVector}}[]

@showprogress desc="compiling all separate files" for k=1:46
    pseudo_manifolds_DB_k = open("new_pseudo_manifolds_DB_10_$(k).jls", "r") do io
        deserialize(io)
    end
    selected_pseudomanifolds = Set{BitVector}()
    V = reduce(|, mat_DB_bin[10][k])
    compl_basis_vecs = [b ⊻ V for b in mat_DB_bin[10][k]]   # ← renommé
    for facets_bit in pseudo_manifolds_DB_k
        facets_bin = compl_basis_vecs[findall(facets_bit)]
        nv_K = count_ones(reduce(|, facets_bin))
        nv_K == 10 && push!(selected_pseudomanifolds, facets_bit)
    end

    push!(pseudo_manifolds_DB[10], selected_pseudomanifolds)
end

open("pseudo_manifolds_DB_7-10_new.jls", "w") do io
    serialize(io, pseudo_manifolds_DB)
end


# database_before_iso = Dict{Tuple{Int,Int}, Set{Vector{UInt32}}}()


# for m=7:10
#     for (l,bases) in enumerate(mat_DB_bin[m])
#         # display(bases)
#         V = reduce(|,bases)
#         compl_bases = [base⊻V for base in bases]
#         @showprogress desc="for m=$(m) " for facets_bit in pseudo_manifolds_DB[m][l]
#             facets_bin = compl_bases[findall(facets_bit)]
#             nv_K = count_ones(reduce(|,facets_bin))
#             d_K = count_ones(facets_bin[1])-1
#             if !((d_K,nv_K) in keys(database_before_iso))
#                 database_before_iso[(d_K,nv_K)] = Set{Vector{UInt32}}()
#             end
#             push!(database_before_iso[(d_K,nv_K)],copy(sort(facets_bin)))
#         end
#     end
# end


# open("rank_5_db_before_iso_7-10.jls", "w") do io
#     serialize(io, database_before_iso)
# end