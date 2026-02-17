using Serialization
using Oscar
using ProgressMeter
using SparseArrays
using Serialization
using Base.Threads
using LinearAlgebra
using Nemo
using Combinatorics

const F2 = GF(2)


function boundary_incidence_facets_to_ridges(facets::Vector{UInt16})
    # collect ridges (each facet contributes its (d-1)-subfaces by deleting one vertex)
    ridge_dict = Dict{UInt16, Int}()  # ridge -> row index
    ridges = Vector{UInt16}()
    for f in facets
        for i=0:((8*sizeof(f))-1)
            if (f>>i)&1==1
                r = f ⊻ (UInt16(1)<<i)
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
        for i=0:(8*sizeof(f)-1)
            if (f>>i)&1==1
                r = f ⊻ (UInt16(1)<<i)
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


function is_mod2_sphere(top_facets::Vector{UInt16})

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
    m=nv(K)
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
    v=find_wedge_pair(K)
    seed_K = K
    while v!=-1
        seed_K=link_subcomplex(seed_K,[v])
        v=find_wedge_pair(seed_K)
    end
    return seed_K
end






# mat_DB = open("rank_4_simple_bin_mat_DB_bin_test.jls", "r") do io
#     deserialize(io)
# end

# iso_DB = open("rank_4_iso_DB_7-15_bin.jls", "r") do io
#     deserialize(io)
# end

database_before_iso = open("rank4_db_before_iso_test13.jls", "r") do io
    deserialize(io)
end


function canonical_form_tuple(facets::Vector{UInt16})

    isempty(facets) && return ()

    # compute vertex mask
    verts_mask = reduce(|, facets)

    verts = Int[]
    for v in 0:15
        if (verts_mask >> v) & 1 == 1
            push!(verts, v)
        end
    end

    k = length(verts)

    # compress vertices to 0:k-1
    relabel0 = Dict{Int,Int}()
    for (i,v) in enumerate(verts)
        relabel0[v] = i-1
    end

    compressed = UInt16[]
    for f in facets
        g = UInt16(0)
        for v in verts
            if (f >> v) & 1 == 1
                g |= UInt16(1) << relabel0[v]
            end
        end
        push!(compressed, g)
    end

    sort!(compressed)

    best = nothing

    perm = collect(0:k-1)

    buffer = similar(compressed)

    function apply_perm!()
        for i in eachindex(compressed)
            f = compressed[i]
            g = UInt16(0)
            for v in 0:k-1
                if (f >> v) & 1 == 1
                    g |= UInt16(1) << perm[v+1]
                end
            end
            buffer[i] = g
        end
        sort!(buffer)
        return Tuple(buffer)
    end

    function backtrack(i)
        if i > k
            cf = apply_perm!()
            if best === nothing || cf < best
                best = cf
            end
            return
        end

        for j in i:k
            perm[i], perm[j] = perm[j], perm[i]
            backtrack(i+1)
            perm[i], perm[j] = perm[j], perm[i]
        end
    end

    backtrack(1)

    return best

end



@inline function link_facets(facets::Vector{UInt16}, v::UInt16)
    mask = UInt16(1) << v
    out = UInt16[]
    for f in facets
        if (f & mask) != 0
            push!(out, f ⊻ mask)
        end
    end
    return out
end

@inline function link_without_vertex(facets::Vector{UInt16}, vmask::UInt16)
    
    lk = UInt16[]
    
    for f in facets
        if (f & vmask) != 0
            push!(lk, f & ~vmask)
        end
    end
    
    sort!(lk)
    return lk
    
end


@inline function vertex_mask(facets::Vector{UInt16})
    m = UInt16(0)
    for f in facets
        m |= f
    end
    return m
end

@inline function vertices_from_mask(mask::UInt16)
    out = Int[]
    for i in 0:15
        if (mask >> i) & 1 == 1
            push!(out, i)
        end
    end
    return out
end

@inline facet_dim(f::UInt16) = count_ones(f) - 1

function is_seed_bit(facets::Vector{UInt16})

    verts_mask = reduce(|, facets)

    verts = UInt16[]
    for v in 0:15
        if (verts_mask >> v) & 1 == 1
            push!(verts, UInt16(v))
        end
    end

    # compute links
    links = Dict{UInt16,Vector{UInt16}}()

    for v in verts
        links[v] = link_facets(facets,v)
    end

    # check pairs
    for i in 1:length(verts)-1

        v = verts[i]
        lk_v = links[v]

        mask_v = UInt16(1) << v


        for j in i+1:length(verts)

            w = verts[j]

            # FAST FACE CHECK:
            # w ∈ lk(v) iff pair is a face
            appears = false
            mask_w = UInt16(1) << w

            for f in lk_v
                if (f & mask_w) != 0
                    appears = true
                    break
                end
            end

            appears || continue


            lk_w_relabel = UInt16[]

            for f in facets
                if (f & mask_w) != 0
                    
                    g = f & ~mask_w   # remove w
                    
                    # swap v and w inside the link
                    
                    has_v = (g & mask_v) != 0
                    
                    if has_v
                        g = (g & ~mask_v) | mask_w
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

function find_wedge_vertex(facets::Vector{UInt16})
    
    verts_mask = reduce(|, facets)
    
    verts = UInt16[]
    for v in 0:15
        if (verts_mask >> v) & 1 == 1
            push!(verts, UInt16(v))
        end
    end
    
    for i in 1:length(verts)-1
        
        v = verts[i]
        vmask = UInt16(1) << v
        
        lk_v = link_without_vertex(facets, vmask)
        
        for j in i+1:length(verts)
            
            w = verts[j]
            wmask = UInt16(1) << w
            
            # face condition: must appear together in some facet
            is_face = false
            for f in facets
                if (f & vmask != 0) && (f & wmask != 0)
                    is_face = true
                    break
                end
            end
            
            is_face || continue
            
            lk_w = link_without_vertex(facets, wmask)
            
            # now remove the other vertex from both links
            
            lk_v2 = UInt16[]
            for f in lk_v
                push!(lk_v2, f & ~wmask)
            end
            
            lk_w2 = UInt16[]
            for f in lk_w
                push!(lk_w2, f & ~vmask)
            end
            
            sort!(lk_v2)
            sort!(lk_w2)
            
            if lk_v2 == lk_w2
                return v
            end
            
        end
    end
    
    return UInt16(0xffff)
    
end


function find_seed_bit(facets::Vector{UInt16})
    
    current = copy(facets)
    
    while true
        
        v = find_wedge_vertex(current)
        
        if v == UInt16(0xffff)
            return current
        end
        
        current = link_without_vertex(current, UInt16(1)<<v)
        
    end
    
end



function index_to_bin(facets::Vector{Vector{Int}})
    @assert max([max(f...) for f in facets]...)<=16
    return canonical_form_tuple([reduce(|,[UInt16(1)<<(i-1) for i in facet]) for facet in facets])
end


database_tc_PLS = Dict{Tuple{Int,Int}, Set{Tuple{Vararg{UInt16}}}}()
database_tc_seed_PLS = Dict{Tuple{Int,Int}, Set{Tuple{Vararg{UInt16}}}}()

database_tc_PLS[(0,2)] = Set([(UInt16(1), UInt16(2))])
database_tc_seed_PLS[(0,2)] = Set([(UInt16(1), UInt16(2))])

database_tc_PLS[(3,8)] = Set{Tuple{Vararg{UInt16}}}([index_to_bin(vec([[x...] for x in Iterators.product(1:2, 3:4, 5:6, 7:8)]))])
database_tc_PLS[(3,8)] = Set{Tuple{Vararg{UInt16}}}([index_to_bin(vec([[x...] for x in Iterators.product(1:2, 3:4, 5:6, 7:8)]))])
database_tc_seed_PLS[(3,8)] = Set{Tuple{Vararg{UInt16}}}([index_to_bin(vec([[x...] for x in Iterators.product(1:2, 3:4, 5:6, 7:8)]))])
database_tc_PLS[(2,6)] = Set{Tuple{Vararg{UInt16}}}([index_to_bin(vec([[x...] for x in Iterators.product(1:2, 3:4, 5:6)]))])
database_tc_seed_PLS[(2,6)] = Set{Tuple{Vararg{UInt16}}}([index_to_bin(vec([[x...] for x in Iterators.product(1:2, 3:4, 5:6)]))])


# database_tc_PLS = Dict{Tuple{Int,Int}, Set{Tuple{Vararg{UInt16}}}}()
# database_tc_seed_PLS = Dict{Tuple{Int,Int}, Set{Tuple{Vararg{UInt16}}}}()

for m in 2:15
    for Pic in 1:4

        key_in = (m-Pic-1, m)
        haskey(database_before_iso, key_in) || continue

        @showprogress for facets_bin in database_before_iso[key_in]

            # compute basic invariants
            Vmask = vertex_mask(facets_bin)
            nv_K = count_ones(Vmask)
            d = facet_dim(facets_bin[1])

            key = (d, nv_K)

            db = get!(database_tc_PLS, key, Set{Tuple{Vararg{UInt16}}}())
            db_seed = get!(database_tc_seed_PLS, key, Set{Tuple{Vararg{UInt16}}}())

            # canonical form once
            cfK = canonical_form_tuple(facets_bin)

            cfK in db && continue

            # vertex list once
            verts = vertices_from_mask(Vmask)

            # check links
            all_links_ok = true

            for v in verts

                Lk = find_seed_bit(link_facets(facets_bin, UInt16(v)))
                isempty(Lk) && (all_links_ok = false; break)

                Lmask = vertex_mask(Lk)
                nv_Lk = count_ones(Lmask)
                d_Lk = facet_dim(Lk[1])

                key_L = (d_Lk, nv_Lk)

                haskey(database_tc_PLS, key_L) || (all_links_ok = false; break)

                cfL = canonical_form_tuple(Lk)

                if !(cfL in database_tc_seed_PLS[key_L])
                    all_links_ok = false
                    break
                end

            end

            all_links_ok || continue

            # sphere test
            is_mod2_sphere(facets_bin) || continue

            # insert once
            push!(db, cfK)
            if is_seed_bit(facets_bin)
                push!(db_seed, cfK)
            end

        end

        key_out = (m-Pic-1, m)

        if haskey(database_tc_PLS, key_out)
            println("PLS count Pic=$Pic m=$m: ",
                length(database_tc_PLS[key_out]))
        end

        if haskey(database_tc_seed_PLS, key_out)
            println("Seed count Pic=$Pic m=$m: ",
                length(database_tc_seed_PLS[key_out]))
        end

    end
end


# for m=2:15
#     for Pic=1:4
#         if !((m-Pic-1,m) in keys(database_before_iso))
#             continue
#         end
#         @showprogress for facets_bin in database_before_iso[(m-Pic-1,m)]
#             # facets_K = [[i for i=1:(8*sizeof(facet_bin)) if (facet_bin>>(i-1))&1==1] for facet_bin in facets_bin]
#             # K = simplicial_complex(facets_K)
#             V_bin = reduce(|,facets_bin)
#             nv_K = count_ones(V_bin)
#             d = count_ones(facets_bin[1])-1
#             V = [i for i=0:15 if (V_bin>>i)&UInt16(1)==UInt16(1)]
#             get!(database_tc_PLS, (d,nv_K), Set{Tuple{Vararg{UInt16}}}())
#             get!(database_tc_seed_PLS, (d,nv_K), Set{Tuple{Vararg{UInt16}}}())

#             is_isom=false
#             can_form_K = canonical_form_tuple(facets_bin)
#             if can_form_K in database_tc_PLS[(d,nv_K)]
#                 continue
#             else
#                 # test if all the link are in the DB
#                 all_link_isom=true

#                 for v in V
#                     Lk_v_bin = [facet_bin⊻(UInt16(1)>>v) for facet_bin in facets_bin if (facet_bin>>v)&UInt16(1)==UInt16(1)]
#                     V_Lk_bin = reduce(|,Lk_v_bin)
#                     nv_Lk = count_ones(V_Lk_bin)
#                     d_Lk = count_ones(Lk_v_bin[1])-1
#                     V_Lk = [i for i=0:15 if (V_Lk_bin>>i)&UInt16(1)==UInt16(1)]
#                     if !((d_Lk,nv_Lk) in keys(database_tc_PLS))
#                         all_link_isom=false
#                         break
#                     end
#                     if !(canonical_form_tuple(Lk_v_bin) in database_tc_PLS[(d_Lk,nv_Lk)])
#                         all_link_isom=false
#                         break
#                     end
#                 end
#                 if !all_link_isom 
#                     continue
#                 end
#                 if is_mod2_sphere(facets_bin)
#                     push!(database_tc_PLS[(d,nv_K)],can_form_K)
                    
#                     if is_seed(minimal_nonfaces(K),nv(K))
#                         push!(database_tc_seed_PLS[(d,nv_K)],can_form_K)
#                         # println("$(d),$(nv_K) ",minimal_nonfaces(K))
#                     end
#                 end
#             end
#         end
#         println("number of PLS up to isom for Pic=$(Pic) and m=$(m): ",length(database_tc_PLS[(m-Pic-1,m)]))

#         println("number of seeds up to isom for Pic=$(Pic) and m=$(m): ",length(database_tc_seed_PLS[(m-Pic-1,m)]))
#     end
# end

# for m=6:10
#     for (l,bases) in enumerate(mat_DB[m])
#         # display(bases)
#         V = reduce(|,bases)
#         compl_bases = [base⊻V for base in bases]
#         @showprogress for facets_bit in pseudo_manifolds_DB[m][l]
#             facets_bin = compl_bases[findall(facets_bit)]
#             facets_K = [[i for i=1:(8*sizeof(facet_bin)) if (facet_bin>>(i-1))&1==1] for facet_bin in facets_bin]
#             K = simplicial_complex(facets_K)
#             if !((d,nv_K) in keys(database_tc_PLS))
#                 database_tc_PLS[(d,nv_K)] = Vector{Vector{Vector{Int}}}()
#             end
#             if !((d,nv_K) in keys(database_tc_seed_PLS))
#                 database_tc_seed_PLS[(d,nv_K)] = Vector{Vector{Vector{Int}}}()
#             end
#             is_isom=false
#             for facets_L in database_tc_PLS[(d,nv_K)]
#                 if is_isomorphic(K,simplicial_complex(facets_L))
#                     is_isom=true
#                     break
#                 end
#             end
#             if !is_isom
#                 # test if all the link are in the DB
#                 all_isom=true
#                 for v=1:nv_K
#                     Lk_v = find_seed(link_subcomplex(K,Set{Int}([v])))
#                     if !((d_Lk,nv_Lk) in keys(database_tc_PLS))
#                         all_isom=false
#                         break
#                     end
#                     link_is_isom=false
#                     for facets_M in database_tc_PLS[(d_Lk,nv_Lk)]
#                         if is_isomorphic(Lk_v,simplicial_complex(facets_M))
#                             link_is_isom=true
#                             break
#                         end
#                     end
#                     all_isom &= link_is_isom
#                     if !all_isom
#                         break
#                     end
#                 end
#                 if !all_isom 
#                     continue
#                 end
#                 if is_mod2_sphere(facets_bin)
#                     push!(database_tc_PLS[(d,nv_K)],copy(facets_K))
#                     if is_seed(minimal_nonfaces(K),nv(K))
#                         push!(database_tc_seed_PLS[(d,nv_K)],copy(facets_K))
#                         # println("$(d),$(nv_K) ",minimal_nonfaces(K))
#                     end
#                 end
#             end
#         end
#     end
#     println("number of PLS up to isom for m=$(m): ",length(database_tc_PLS[(m-5,m)]))

#     println("number of seeds up to isom for m=$(m): ",length(database_tc_seed_PLS[(m-5,m)]))
# end

open("Pic_4_tc_PLS_test_13.jls", "w") do io
    serialize(io, database_tc_PLS)
end

open("Pic_4_tc_seed_PLS_test_13.jls", "w") do io
    serialize(io, database_tc_seed_PLS)
end
