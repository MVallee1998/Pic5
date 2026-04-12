include("Pic_5_reduce_iso_and_PL_sphere.jl")

function convert_dict_uint16_to_uint32(
    d::Dict{Tuple{Int,Int}, Set{Tuple{Vararg{UInt16}}}}
)::Dict{Tuple{Int,Int}, Set{Tuple{Vararg{UInt32}}}}
    Dict(
        key => Set(Tuple(UInt32.(t)) for t in val)
        for (key, val) in d
    )
end

mat_DB_bin = open("rank_5_mat_DB_bin.jls", "r") do io
    deserialize(io)
end

iso_DB = open("rank_5_iso_DB_bin_all_v.jls", "r") do io
    deserialize(io)
end


pseudo_manifolds_DB = open("pseudo_manifolds_DB_7-9.jls", "r") do io
    deserialize(io)
end

database_before_iso = Dict{Tuple{Int,Int}, Set{Vector{UInt32}}}()


for m=7:9
    for (l,bases) in enumerate(mat_DB_bin[m])
        # display(bases)
        V = reduce(|,bases)
        compl_bases = [base⊻V for base in bases]
        @showprogress desc="for m=$(m) " for facets_bit in pseudo_manifolds_DB[m][l]
            facets_bin = compl_bases[findall(facets_bit)]
            nv_K = count_ones(reduce(|,facets_bin))
            d_K = count_ones(facets_bin[1])-1
            nv_K == m || continue
            if !((d_K,nv_K) in keys(database_before_iso))
                database_before_iso[(d_K,nv_K)] = Set{Vector{UInt32}}()
            end
            push!(database_before_iso[(d_K,nv_K)],copy(sort(facets_bin)))
        end
    end
end


            
open("rank_5_db_before_iso_7-10_new.jls", "w") do io
    serialize(io, database_before_iso)
end

# database_before_iso = open("rank_5_db_before_iso_7-10.jls", "r") do io
#     deserialize(io)
# end


# Keep original database structure - only bin format
# database_tc_PLS = Dict{Tuple{Int,Int},Set{Tuple{Vararg{UInt32}}}}()
# database_tc_seed_PLS = open("Pic_5_tc_PLS_7-9.jls", "r") do io
#     deserialize(io)
# end

# database_tc_seed_PLS = Dict{Tuple{Int,Int},Set{Tuple{Vararg{UInt32}}}}()

database_tc_seed_PLS_16 = open("Pic_4_tc_PLS_6-13.jls", "r") do io
    deserialize(io)
end

database_tc_seed_PLS = convert_dict_uint16_to_uint32(database_tc_seed_PLS_16)


# for m in 2:9
#     for Pic in 1:5
#         key = (m - Pic - 1, m)
#         haskey(database_tc_seed_PLS, key) || continue
#         database_tc_seed_PLS[key] = Set{Tuple{Vararg{UInt32}}}(
#             Tuple(UInt32(f) for f in facets_bin)
#             for facets_bin in database_tc_seed_PLS[key]
#         )
#     end
# end

# # Initialize
# # database_tc_PLS[(0, 2)] = Set([(UInt32(1), UInt32(2))])
# database_tc_seed_PLS[(0, 2)] = Set([(UInt32(1), UInt32(2))])

# cube_facets = index_to_bin(vec([[x...] for x in Iterators.product(1:2, 3:4, 5:6, 7:8)]))
# # database_tc_PLS[(3, 8)] = Set([cube_facets])
# database_tc_seed_PLS[(3, 8)] = Set([cube_facets])

# oct_facets = index_to_bin(vec([[x...] for x in Iterators.product(1:2, 3:4, 5:6)]))
# # database_tc_PLS[(2, 6)] = Set([oct_facets])
# database_tc_seed_PLS[(2, 6)] = Set([oct_facets])

for m in 7:9
    for Pic in 5:5
        key_in = (m - Pic - 1, m)
        haskey(database_before_iso, key_in) || continue
        items   = collect(database_before_iso[key_in])
        db_seed = get!(database_tc_seed_PLS, key_in, Set{Tuple{Vararg{UInt32}}}())

        # ── Phase 1: parallel — pure Julia filters only ───────────────────────
        prog = Progress(length(items); desc="Filters (m=$m, Pic=$Pic): ")

        candidates = let
            local_cands = [Vector{Tuple{Vararg{UInt32}}}() for _ in 1:length(items)]
            @threads :dynamic for i in eachindex(items)
                facets_bin = items[i]
                next!(prog)
                is_seed_bit(facets_bin)    || continue
                is_mod2_sphere(facets_bin) || continue
                push!(local_cands[i], Tuple(facets_bin))
            end
            reduce(vcat, local_cands)
        end

        # ── Phase 2: sequential Oscar checks, index-accelerated ──────────────
        db_index = build_index(db_seed)
        prog2 = Progress(length(candidates); desc="Iso checks (m=$m, Pic=$Pic): ")

        for facets_bin in candidates
            next!(prog2; showvalues = [(:candidates, length(candidates)),
                                       (:seeds, length(db_seed)),
                                       (:buckets, length(db_index))])

            verts = vertices_from_mask(vertex_mask(facets_bin))

            all_links_ok = all(verts) do v
                Lk    = find_seed_bit(link_facets(facets_bin, v))
                isempty(Lk) && return false
                key_L = (facet_dim(Lk[1]), count_ones(vertex_mask(Lk)))
                haskey(database_tc_seed_PLS, key_L) &&
                    is_isomorphic_to_any(Lk, database_tc_seed_PLS[key_L])
            end

            if all_links_ok && !is_isomorphic_to_any_indexed(facets_bin, db_index)
                push_indexed!(db_seed, db_index, facets_bin)
            end
        end

        Pic == 5 && println("Seed count Pic=$Pic m=$m: ", length(db_seed))
    end
end


# open("Pic_4_tc_PLS.jls", "w") do io
#     serialize(io, database_tc_PLS)
# end

open("Pic_5_tc_seed_PLS_5-9.jls", "w") do io
    serialize(io, database_tc_seed_PLS)
end
