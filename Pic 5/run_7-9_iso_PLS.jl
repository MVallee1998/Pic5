include("Pic_5_reduce_iso_and_PL_sphere.jl")


database_before_iso = open("rank_5_db_before_iso_7-9.jls", "r") do io
    deserialize(io)
end


# Keep original database structure - only bin format
# database_tc_PLS = Dict{Tuple{Int,Int},Set{Tuple{Vararg{UInt32}}}}()
# database_tc_seed_PLS = open("Pic_5_tc_PLS_7-9.jls", "r") do io
#     deserialize(io)
# end

database_tc_seed_PLS = Dict{Tuple{Int,Int},Set{Tuple{Vararg{UInt32}}}}()

# database_tc_seed_PLS = open("Pic_4_tc_PLS_7-9.jls", "r") do io
#     deserialize(io)
# end

for m in 2:13
    for Pic in 1:5
        key = (m - Pic - 1, m)
        haskey(database_tc_seed_PLS, key) || continue
        database_tc_seed_PLS[key] = Set{Tuple{Vararg{UInt32}}}(
            Tuple(UInt32(f) for f in facets_bin)
            for facets_bin in database_tc_seed_PLS[key]
        )
    end
end

# Initialize
# database_tc_PLS[(0, 2)] = Set([(UInt32(1), UInt32(2))])
database_tc_seed_PLS[(0, 2)] = Set([(UInt32(1), UInt32(2))])

cube_facets = index_to_bin(vec([[x...] for x in Iterators.product(1:2, 3:4, 5:6, 7:8)]))
# database_tc_PLS[(3, 8)] = Set([cube_facets])
database_tc_seed_PLS[(3, 8)] = Set([cube_facets])

oct_facets = index_to_bin(vec([[x...] for x in Iterators.product(1:2, 3:4, 5:6)]))
# database_tc_PLS[(2, 6)] = Set([oct_facets])
database_tc_seed_PLS[(2, 6)] = Set([oct_facets])

for m in 3:9
    for Pic in 1:5
        key_in = (m - Pic - 1, m)
        haskey(database_before_iso, key_in) || continue

        items = collect(database_before_iso[key_in])
        prog  = Progress(length(items))

        db_seed = get!(database_tc_seed_PLS, key_in, Set{Tuple{Vararg{UInt32}}}())

        # ── Phase 1: parallel — pure Julia filters only ───────────────────────
        # No Oscar calls here whatsoever.
        candidates = Vector{Tuple{Vararg{UInt32}}}()
        cand_lock  = ReentrantLock()

        @threads :dynamic for facets_bin in items
            next!(prog; showvalues = [(:seeds, length(db_seed)), (:Pic, Pic), (:m, m)])

            is_seed_bit(facets_bin)    || continue
            is_mod2_sphere(facets_bin) || continue

            # Structural link check (no Oscar): verify that each link's (d, nv)
            # key is already present in the database — cheap early-out.
            Vmask = vertex_mask(facets_bin)
            verts = vertices_from_mask(Vmask)
            structurally_ok = true

            for v in verts
                Lk = find_seed_bit(link_facets(facets_bin, v))
                if isempty(Lk)
                    structurally_ok = false; break
                end
                key_L = (facet_dim(Lk[1]), count_ones(vertex_mask(Lk)))
                if !haskey(database_tc_seed_PLS, key_L)
                    structurally_ok = false; break
                end
            end
            structurally_ok || continue

            lock(cand_lock) do
                push!(candidates, Tuple(facets_bin))
            end
        end

        # ── Phase 2: sequential Oscar checks, now index-accelerated ──────────
        # Build index once from whatever is already in db_seed.
        db_index = build_index(db_seed)
        prog2 = Progress(length(candidates); desc="Iso checks (m=$m, Pic=$Pic): ")

        for facets_bin in candidates
            Vmask = vertex_mask(facets_bin)
            verts = vertices_from_mask(Vmask)
            all_links_ok = true

            for v in verts
                Lk = find_seed_bit(link_facets(facets_bin, v))
                isempty(Lk) && (all_links_ok = false; break)

                key_L = (facet_dim(Lk[1]), count_ones(vertex_mask(Lk)))
                if !haskey(database_tc_seed_PLS, key_L) ||
                   !is_isomorphic_to_any(Lk, database_tc_seed_PLS[key_L])  # link db stays flat (small)
                    all_links_ok = false; break
                end
            end

            if all_links_ok && !is_isomorphic_to_any_indexed(facets_bin, db_index)
                push_indexed!(db_seed, db_index, facets_bin)
            end

            next!(prog2; showvalues = [(:candidates, length(candidates)),
                                       (:seeds, length(db_seed)),
                                       (:buckets, length(db_index))])
        end

        if Pic == 5
            println("Seed count Pic=$Pic m=$m: ", length(db_seed))
        end
    end
end


# open("Pic_4_tc_PLS.jls", "w") do io
#     serialize(io, database_tc_PLS)
# end

open("Pic_5_tc_seed_PLS.jls", "w") do io
    serialize(io, database_tc_seed_PLS)
end
