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


# database_reduce_autom = open("database_reduce_autom_7-9.jls", "r") do io
#     deserialize(io)
# end

# build_finalDB_single_v!(pseudo_manifolds_DB,database_reduce_autom,mat_DB_bin,iso_DB,mmax,mstart=10)
build_finalDB_single_v_one_m!(pseudo_manifolds_DB,mat_DB_bin,iso_DB,10)

# open("pseudo_manifolds_DB_7-9.jls", "w") do io
#     serialize(io, pseudo_manifolds_DB)
# end

# open("database_reduce_autom_7-9.jls", "w") do io
#     serialize(io, database_reduce_autom)
# end

database_before_iso = Dict{Tuple{Int,Int}, Set{Vector{UInt32}}}()


for m=7:mmax
    for (l,bases) in enumerate(mat_DB_bin[m])
        # display(bases)
        V = reduce(|,bases)
        compl_bases = [base⊻V for base in bases]
        @showprogress desc="for m=$(m) " for facets_bit in pseudo_manifolds_DB[m][l]
            facets_bin = compl_bases[findall(facets_bit)]
            nv_K = count_ones(reduce(|,facets_bin))
            d_K = count_ones(facets_bin[1])-1
            if !((d_K,nv_K) in keys(database_before_iso))
                database_before_iso[(d_K,nv_K)] = Set{Vector{UInt32}}()
            end
            push!(database_before_iso[(d_K,nv_K)],copy(sort(facets_bin)))
        end
    end
end
# for facets_M in list_link
#     for facets_bin in database_before_iso[(3,8)]
#         facets_L = [[i for i=1:(8*sizeof(facet_bin)) if (facet_bin>>(i-1))&1==1] for facet_bin in facets_bin]
#         L = simplicial_complex(facets_L)
#         if is_isomorphic(L,simplicial_complex(facets_M))
#             println("hello",facets_M)
#             break
#         end
#     end
# end
            
open("rank_5_db_before_iso_7-10.jls", "w") do io
    serialize(io, database_before_iso)
end