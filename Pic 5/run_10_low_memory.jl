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