using Serialization
using Oscar
using ProgressMeter
using SparseArrays
using Serialization
using Base.Threads
using LinearAlgebra
using Nemo

const F2 = GF(2)


function boundary_incidence_facets_to_ridges(facets::Vector{UInt32})
    # collect ridges (each facet contributes its (d-1)-subfaces by deleting one vertex)
    ridge_dict = Dict{UInt32,Int}()  # ridge -> row index
    ridges = Vector{UInt32}()
    for f in facets
        for i = 0:((8*sizeof(f))-1)
            if (f >> i) & 1 == 1
                r = f ⊻ (UInt32(1) << i)
                if !haskey(ridge_dict, r)
                    push!(ridges, r)
                    ridge_dict[r] = length(ridges)
                end
            end
        end
    end

    m = length(ridges)
    n = length(facets)

    # build sparse 0/1 matrix (m x n): ridges x facets
    I = Int[]   # row indices
    J = Int[]   # col indices

    for (j, f) in pairs(facets)
        for i = 0:(8*sizeof(f)-1)
            if (f >> i) & 1 == 1
                r = f ⊻ (UInt32(1) << i)
                row_idx = ridge_dict[r]
                push!(I, row_idx)
                push!(J, j)
            end
        end
    end

    A = sparse(I, J, true, m, n)  # SparseMatrixCSC{Bool} 

    return ridges, A
end

function mod2_rank_nemo(A::SparseMatrixCSC)
    m, n = size(A)
    if m == 0 || n == 0
        return 0
    end

    M = zero_matrix(F2, m, n)

    # Fill matrix directly from sparse structure
    for col in 1:n
        for idx in A.colptr[col]:(A.colptr[col+1]-1)
            row = A.rowval[idx]
            M[row, col] = F2(1)
        end
    end

    return rank(M)
end


function is_mod2_sphere(top_facets::Vector{UInt32})

    isempty(top_facets) && return true

    d = count_ones(top_facets[1]) - 1

    current_faces = top_facets

    # ---- Step 1: Top homology β_d ----
    # β_d = (#d-faces - rank ∂_d)
    ridges, B = boundary_incidence_facets_to_ridges(current_faces)
    rank_prev = mod2_rank_nemo(B)

    n_d = length(current_faces)
    β_d = n_d - rank_prev
    β_d == 1 || return false

    current_faces = ridges

    # ---- Step 2: Middle dimensions ----
    # For i = d-1 down to 1:
    for dim = d-1:-1:1

        ridges, B = boundary_incidence_facets_to_ridges(current_faces)
        rank_curr = mod2_rank_nemo(B)

        n_i = length(current_faces)

        # β_i = (#i-faces - rank ∂_i) - rank ∂_{i+1}
        β_i = (n_i - rank_curr) - rank_prev
        β_i == 0 || return false

        rank_prev = rank_curr
        current_faces = ridges
    end

    # ---- Step 3: β_0 ----
    # β_0 = (#vertices) - rank ∂_1
    n_0 = length(current_faces)
    β_0 = n_0 - rank_prev
    β_0 == 1 || return false

    return true
end


function is_seed(MNF::Vector{Set{Int}}, m::Int)
    MNF_sets = Set(MNF)

    # iterate over all pairs of vertices
    for v in 1:m-1
        for w in v+1:m

            pair = Set((v, w))
            is_pair = true

            for F in MNF_sets
                # exactly one present → break condition
                if (v in F) ⊻ (w in F)
                    is_pair = false
                    break
                end
            end

            # if always together and pair not itself MNF → not seed
            if is_pair && !(pair in MNF_sets)
                return false
            end
        end
    end

    return true
end

function find_wedge_pair(K::SimplicialComplex)
    m = nv(K)
    MNF_sets = Set(minimal_nonfaces(K))

    # iterate over all pairs of vertices
    for v in 1:m-1
        for w in v+1:m

            pair = Set((v, w))
            is_pair = true

            for F in MNF_sets
                # exactly one present → break condition
                if (v in F) ⊻ (w in F)
                    is_pair = false
                    break
                end
            end

            # if always together and pair not itself MNF → not seed
            if is_pair && !(pair in MNF_sets)
                return v
            end
        end
    end
    return -1
end

function find_seed(K::SimplicialComplex)
    v = find_wedge_pair(K)
    seed_K = K
    while v != -1
        seed_K = link_subcomplex(seed_K, [v])
        v = find_wedge_pair(seed_K)
    end
    return seed_K
end



# Make all functions work with both Vector and Tuple
const Facets = Union{Vector{UInt32},Tuple{Vararg{UInt32}}}

@inline function link_facets(facets::Facets, v::UInt32)
    mask = UInt32(1) << v
    out = UInt32[]
    for f in facets
        if (f & mask) != 0
            push!(out, f ⊻ mask)
        end
    end
    return out
end

@inline function link_without_vertex(facets::Facets, vmask::UInt32)
    lk = UInt32[]
    for f in facets
        if (f & vmask) != 0
            push!(lk, f & ~vmask)
        end
    end
    sort!(lk)
    return lk
end

@inline function vertex_mask(facets::Facets)
    m = UInt32(0)
    for f in facets
        m |= f
    end
    return m
end

@inline function vertices_from_mask(mask::UInt32)
    out = UInt32[]
    for i in 0:31
        if (mask >> i) & 1 == 1
            push!(out, UInt32(i))
        end
    end
    return out
end

@inline facet_dim(f::UInt32) = count_ones(f) - 1

function is_seed_bit(facets::Facets)
    verts_mask = vertex_mask(facets)
    verts = vertices_from_mask(verts_mask)

    # Compute all links once
    links = Dict{UInt32,Vector{UInt32}}()
    for v in verts
        links[v] = link_facets(facets, v)
    end

    # Check all vertex pairs
    for i in 1:length(verts)-1
        v = verts[i]
        lk_v = links[v]
        mask_v = UInt32(1) << v

        for j in i+1:length(verts)
            w = verts[j]
            mask_w = UInt32(1) << w

            # Check if w appears in lk(v) - means (v,w) is a face
            appears = any(f -> (f & mask_w) != 0, lk_v)
            appears || continue

            # Compute lk(w) with v and w swapped
            lk_w_relabel = UInt32[]
            for f in facets
                if (f & mask_w) != 0
                    g = f & ~mask_w  # remove w
                    if (g & mask_v) != 0  # if has v
                        g = (g & ~mask_v) | mask_w  # swap v↔w
                    end
                    push!(lk_w_relabel, g)
                end
            end
            sort!(lk_w_relabel)

            if lk_v == lk_w_relabel
                return false
            end
        end
    end

    return true
end

function find_wedge_vertex(facets::Facets)
    verts_mask = vertex_mask(facets)
    verts = vertices_from_mask(verts_mask)

    for i in 1:length(verts)-1
        v = verts[i]
        vmask = UInt32(1) << v
        lk_v = link_without_vertex(facets, vmask)

        for j in i+1:length(verts)
            w = verts[j]
            wmask = UInt32(1) << w

            # Check if (v,w) is a face
            is_face = any(f -> (f & vmask != 0) && (f & wmask != 0), facets)
            is_face || continue

            lk_w = link_without_vertex(facets, wmask)

            # Remove the other vertex from both links
            lk_v2 = sort!([f & ~wmask for f in lk_v])
            lk_w2 = sort!([f & ~vmask for f in lk_w])

            if lk_v2 == lk_w2
                return v
            end
        end
    end

    return UInt32(0xffffffff)
end

function find_seed_bit(facets::Facets)
    current = collect(facets)  # Convert to Vector for mutation

    while true
        v = find_wedge_vertex(current)
        v == UInt32(0xffffffff) && return current
        current = link_without_vertex(current, UInt32(1) << v)
    end
end

# Oscar conversion function
function to_oscar_complex(facets::Facets)
    Vmask = vertex_mask(facets)
    verts = vertices_from_mask(Vmask)

    # Convert facets from bitmasks to vertex sets (1-indexed for Oscar)
    facets_as_sets = Vector{Int}[]
    for f in facets
        facet_verts = Int[]
        for v in verts
            if (f >> v) & 1 == 1
                push!(facet_verts, Int(v) + 1)  # Oscar uses 1-indexed vertices
            end
        end
        push!(facets_as_sets, facet_verts)
    end

    return simplicial_complex(facets_as_sets)
end

# Check if complex is isomorphic to any in the database
function is_isomorphic_to_any(facets_bin::Facets, db)
    K_oscar = to_oscar_complex(facets_bin)

    for existing_bin in db
        existing_oscar = to_oscar_complex(existing_bin)
        if Oscar.is_isomorphic(K_oscar, existing_oscar)
            return true
        end
    end

    return false
end


function index_to_bin(facets::Vector{Vector{Int}})
    @assert max([max(f...) for f in facets]...) <= 32
    return Tuple(sort([reduce(|, [UInt32(1) << (i - 1) for i in facet]) for facet in facets]))
end


database_before_iso = open("rank5_db_before_iso.jls", "r") do io
    deserialize(io)
end


# Keep original database structure - only bin format
# database_tc_PLS = Dict{Tuple{Int,Int},Set{Tuple{Vararg{UInt32}}}}()
database_tc_seed_PLS = Dict{Tuple{Int,Int},Set{Tuple{Vararg{UInt32}}}}()

# Initialize
# database_tc_PLS[(0, 2)] = Set([(UInt32(1), UInt32(2))])
database_tc_seed_PLS[(0, 2)] = Set([(UInt32(1), UInt32(2))])

cube_facets = index_to_bin(vec([[x...] for x in Iterators.product(1:2, 3:4, 5:6, 7:8)]))
# database_tc_PLS[(3, 8)] = Set([cube_facets])
database_tc_seed_PLS[(3, 8)] = Set([cube_facets])

oct_facets = index_to_bin(vec([[x...] for x in Iterators.product(1:2, 3:4, 5:6)]))
# database_tc_PLS[(2, 6)] = Set([oct_facets])
database_tc_seed_PLS[(2, 6)] = Set([oct_facets])

for m in 2:12
    for Pic in 1:5
        key_in = (m - Pic - 1, m)
        haskey(database_before_iso, key_in) || continue

        @showprogress for facets_bin in database_before_iso[key_in]
            is_seed_bit(facets_bin) || continue
            is_mod2_sphere(facets_bin) || continue

            Vmask = vertex_mask(facets_bin)
            nv_K = count_ones(Vmask)
            d = facet_dim(facets_bin[1])
            key = (d, nv_K)

            db_seed = get!(database_tc_seed_PLS, key, Set{Tuple{Vararg{UInt32}}}())

            is_isomorphic_to_any(facets_bin, db_seed) && continue

            verts = vertices_from_mask(Vmask)
            all_links_ok = true

            for v in verts
                Lk = find_seed_bit(link_facets(facets_bin, v))
                isempty(Lk) && (all_links_ok = false; break)

                Lmask = vertex_mask(Lk)
                nv_Lk = count_ones(Lmask)
                d_Lk = facet_dim(Lk[1])
                key_L = (d_Lk, nv_Lk)

                # was haskey(database_tc_PLS, key_L) - but we no longer populate that
                haskey(database_tc_seed_PLS, key_L) || (all_links_ok = false; break)
                # (d_Lk>0 && is_mod2_sphere(Lk)) || (all_links_ok = false; break)
                is_isomorphic_to_any(Lk, database_tc_seed_PLS[key_L]) || (all_links_ok = false; break)
            end

            all_links_ok || continue

            push!(db_seed, Tuple(facets_bin))
        end

        key_out = (m - Pic - 1, m)
        if haskey(database_tc_seed_PLS, key_out) && Pic==5
            println("Seed count Pic=$Pic m=$m: ", length(database_tc_seed_PLS[key_out]))
        end
    end
end


# open("Pic_4_tc_PLS.jls", "w") do io
#     serialize(io, database_tc_PLS)
# end

open("Pic_5_tc_seed_PLS.jls", "w") do io
    serialize(io, database_tc_seed_PLS)
end
