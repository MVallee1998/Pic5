using Serialization
using Oscar
using ProgressMeter
using SparseArrays
using Serialization
using Base.Threads
using LinearAlgebra
using Nemo

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

database_before_iso = open("rank4_db_before_iso_test12.jls", "r") do io
    deserialize(io)
end




database_tc_PLS = Dict{Tuple{Int,Int}, Vector{Vector{Vector{Int}}}}()

database_tc_seed_PLS = Dict{Tuple{Int,Int}, Vector{Vector{Vector{Int}}}}()

database_tc_PLS[(0,2)] = Vector{Vector{Vector{Int}}}([[[1],[2]]])
database_tc_seed_PLS[(0,2)] = Vector{Vector{Vector{Int}}}([[[1],[2]]])

database_tc_PLS[(3,8)] = Vector{Vector{Vector{Int}}}([vec([[x...] for x in Iterators.product(1:2, 3:4, 5:6, 7:8)])])
database_tc_PLS[(3,8)] = Vector{Vector{Vector{Int}}}([vec([[x...] for x in Iterators.product(1:2, 3:4, 5:6, 7:8)])])
database_tc_seed_PLS[(3,8)] = Vector{Vector{Vector{Int}}}([vec([[x...] for x in Iterators.product(1:2, 3:4, 5:6, 7:8)])])
database_tc_PLS[(2,6)] = Vector{Vector{Vector{Int}}}([vec([[x...] for x in Iterators.product(1:2, 3:4, 5:6)])])
database_tc_seed_PLS[(2,6)] = Vector{Vector{Vector{Int}}}([vec([[x...] for x in Iterators.product(1:2, 3:4, 5:6)])])


for m=2:15
    for Pic=1:4
        if !((m-Pic-1,m) in keys(database_before_iso))
            continue
        end
        @showprogress for facets_bin in database_before_iso[(m-Pic-1,m)]
            facets_K = [[i for i=1:(8*sizeof(facet_bin)) if (facet_bin>>(i-1))&1==1] for facet_bin in facets_bin]
            K = simplicial_complex(facets_K)
            if !((dim(K),n_vertices(K)) in keys(database_tc_PLS))
                database_tc_PLS[(dim(K),n_vertices(K))] = Vector{Vector{Vector{Int}}}()
            end
            if !((dim(K),n_vertices(K)) in keys(database_tc_seed_PLS))
                database_tc_seed_PLS[(dim(K),n_vertices(K))] = Vector{Vector{Vector{Int}}}()
            end
            is_isom=false
            for facets_L in database_tc_PLS[(dim(K),n_vertices(K))]
                if is_isomorphic(K,simplicial_complex(facets_L))
                    is_isom=true
                    break
                end
            end
            if !is_isom
                # test if all the link are in the DB
                all_isom=true
                for v=1:n_vertices(K)
                    Lk_v = find_seed(link_subcomplex(K,Set{Int}([v])))
                    if !((dim(Lk_v),n_vertices(Lk_v)) in keys(database_tc_PLS))
                        all_isom=false
                        break
                    end
                    link_is_isom=false
                    for facets_M in database_tc_PLS[(dim(Lk_v),n_vertices(Lk_v))]
                        if is_isomorphic(Lk_v,simplicial_complex(facets_M))
                            link_is_isom=true
                            break
                        end
                    end
                    all_isom &= link_is_isom
                    if !all_isom
                        break
                    end
                end
                if !all_isom 
                    continue
                end
                if is_mod2_sphere(facets_bin)
                    push!(database_tc_PLS[(dim(K),n_vertices(K))],copy(facets_K))
                    
                    if is_seed(minimal_nonfaces(K),nv(K))
                        push!(database_tc_seed_PLS[(dim(K),n_vertices(K))],copy(facets_K))
                        # println("$(dim(K)),$(n_vertices(K)) ",minimal_nonfaces(K))
                    end
                end
            end
        end
        println("number of PLS up to isom for Pic=$(Pic) and m=$(m): ",length(database_tc_PLS[(m-Pic-1,m)]))

        println("number of seeds up to isom for Pic=$(Pic) and m=$(m): ",length(database_tc_seed_PLS[(m-Pic-1,m)]))
    end
end

# for m=6:10
#     for (l,bases) in enumerate(mat_DB[m])
#         # display(bases)
#         V = reduce(|,bases)
#         compl_bases = [base⊻V for base in bases]
#         @showprogress for facets_bit in pseudo_manifolds_DB[m][l]
#             facets_bin = compl_bases[findall(facets_bit)]
#             facets_K = [[i for i=1:(8*sizeof(facet_bin)) if (facet_bin>>(i-1))&1==1] for facet_bin in facets_bin]
#             K = simplicial_complex(facets_K)
#             if !((dim(K),n_vertices(K)) in keys(database_tc_PLS))
#                 database_tc_PLS[(dim(K),n_vertices(K))] = Vector{Vector{Vector{Int}}}()
#             end
#             if !((dim(K),n_vertices(K)) in keys(database_tc_seed_PLS))
#                 database_tc_seed_PLS[(dim(K),n_vertices(K))] = Vector{Vector{Vector{Int}}}()
#             end
#             is_isom=false
#             for facets_L in database_tc_PLS[(dim(K),n_vertices(K))]
#                 if is_isomorphic(K,simplicial_complex(facets_L))
#                     is_isom=true
#                     break
#                 end
#             end
#             if !is_isom
#                 # test if all the link are in the DB
#                 all_isom=true
#                 for v=1:n_vertices(K)
#                     Lk_v = find_seed(link_subcomplex(K,Set{Int}([v])))
#                     if !((dim(Lk_v),n_vertices(Lk_v)) in keys(database_tc_PLS))
#                         all_isom=false
#                         break
#                     end
#                     link_is_isom=false
#                     for facets_M in database_tc_PLS[(dim(Lk_v),n_vertices(Lk_v))]
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
#                     push!(database_tc_PLS[(dim(K),n_vertices(K))],copy(facets_K))
#                     if is_seed(minimal_nonfaces(K),nv(K))
#                         push!(database_tc_seed_PLS[(dim(K),n_vertices(K))],copy(facets_K))
#                         # println("$(dim(K)),$(n_vertices(K)) ",minimal_nonfaces(K))
#                     end
#                 end
#             end
#         end
#     end
#     println("number of PLS up to isom for m=$(m): ",length(database_tc_PLS[(m-5,m)]))

#     println("number of seeds up to isom for m=$(m): ",length(database_tc_seed_PLS[(m-5,m)]))
# end

open("Pic_4_tc_PLS_test.jls", "w") do io
    serialize(io, database_tc_PLS)
end

open("Pic_4_tc_seed_PLS_test.jls", "w") do io
    serialize(io, database_tc_seed_PLS)
end
