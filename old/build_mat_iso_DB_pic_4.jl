using Oscar
using ProgressBars
using SparseArrays
using Serialization

using Polymake
const topaz = Polymake.topaz
using ProgressMeter
using Nemo
F = GF(2)

# Build canonical vertex list present in a matroid (sorted)
function vertex_list_of_bases(bases::Vector{Vector{UInt8}})
    s = Set{UInt8}()
    for b in bases
        for v in b
            push!(s, v)
        end
    end
    return sort(collect(s))
end

# Construct a matrix from a binary matroid

function binary_matrix_representation(M; B=nothing)
    is_binary(M) || error("Matroid is not binary")

    # ground set
    E = collect(matroid_groundset(M))
    n = length(E)

    # choose basis
    if B === nothing
        B = first(bases(M))
    else
        length(B) == rank(M) || error("Provided set is not a basis")
    end

    r = length(B)

    # index maps
    row_of = Dict(b => i for (i,b) in enumerate(B))
    col_of = Dict(e => j for (j,e) in enumerate(E))

    # initialize matrix
    A = matrix(F,zeros(Int, (r, n)))

    # basis columns = identity
    for b in B
        A[row_of[b], col_of[b]] = F(1)
    end

    # non-basis columns via fundamental circuits
    for e in E
        e in B && continue
        C = fundamental_circuit(M, B, e)
        for b in C
            b in B || continue
            A[row_of[b], col_of[e]] = F(1)
        end
    end

    return A, B, E
end

# ---------------------------
# using Topaz for isom
# ---------------------------

function bases_to_topaz_complex(bases::Vector{Vector{UInt8}})
    # collect and sort vertices
    verts = sort!(unique(vcat(bases...)))

    # map vertex labels to 0-based indices
    vmap = Dict(v => i-1 for (i,v) in enumerate(verts))

    facets = [ [vmap[x] for x in B] for B in bases ]
    return topaz.SimplicialComplex(FACETS = facets), verts
end

function topaz_isomorphism(basesA, basesB)
    SC_A, vertsA = bases_to_topaz_complex(basesA)
    SC_B, vertsB = bases_to_topaz_complex(basesB)
    length(vertsA) == length(vertsB) || return nothing
    T = topaz.find_facet_vertex_permutations(SC_B, SC_A)
    if T===nothing
        return nothing
    end
    _,p=T
    perm = Dict(
        vertsA[i] => vertsB[p[i]+1]
        for i in eachindex(vertsA)
    )
    return perm
end






# ---------------------------
# Main builder
# ---------------------------
"""
build_iso_db!(Iso_DB, mat_DB; ms = sort(collect(keys(mat_DB))), verbose=false)

- Iso_DB will be mutated (create empty Dict before passing).
- mat_DB: Dict{Int, Vector{Vector{Vector{Int}}}} mapping m -> list of matroid-bases
- For each m in ms (skips if m-1 not present), and each k in mat_DB[m],
  finds first vertex v in 1:m such that corank(contract_at_vertex(M,v)) >= corank(M),
  computes contraction, then finds l and a permutation mapping contraction -> some mat_DB[m-1][l].
- Stores Iso_DB[m][k] = (l, mapping::Dict{Int,Int}) or (-1, nothing) if not found.
"""
function build_iso_db!(Iso_DB::Dict{Int,Dict{Int,Vector{Tuple{Int,Any}}}}, mat_DB::Dict{Int,Vector{Vector{UInt16}}}; ms=nothing, verbose=false)
    # ms === nothing && (ms = sort(collect(keys(mat_DB))))
    for m in ms
        println(m)
        if !haskey(mat_DB, m-1)
            verbose && @info("Skipping m=$m because mat_DB[$(m-1)] not present")
            continue
        end
        Iso_DB[m] = Dict{Int,Vector{Tuple{Int,Any}}}()
        for (k, Mbases_bin) in enumerate(mat_DB[m])
            Mbases= [[UInt8(i) for i=1:16 if ((base_bin>>(i-1))&UInt16(1))==1] for base_bin in Mbases_bin]
            V=vertex_list_of_bases(Mbases)
            M = matroid_from_bases(Mbases,V)
            Iso_DB[m][k] =Vector{Tuple{Int,Any}}()
            found_v=false
            dict_v = Dict{Int,Matroid}()
            v_chosen=-1
            for v in V
                if v in coloops(M)
                    continue
                end
                Mv = deletion(M,v)
                # dict_v[v] = Mv

                if length(coloops(Mv))>length(coloops(M))
                    dict_v[v] = Mv
                    continue
                end
                v_chosen=v
                found_v=true
            end
            if found_v
                dict_v = Dict{Int,Matroid}()
                dict_v[v_chosen] = deletion(M,v_chosen)
            end

            for v in keys(dict_v)
                deletion_bases = bases(dict_v[v])

                # search through mat_DB[m-1] for an isomorphism
                perm = nothing
                found_index = -1
                for (l, target_bases_bin) in enumerate(mat_DB[m-1])
                    target_bases = [[UInt8(i) for i=1:16 if ((base_bin>>(i-1))&UInt16(1))==1] for base_bin in target_bases_bin]
                    M2 = matroid_from_bases(target_bases,vertex_list_of_bases(target_bases))
                    if !is_isomorphic(dict_v[v],M2)
                        continue
                    end
                    perm = topaz_isomorphism(target_bases,deletion_bases)
                    if perm !== nothing
                        # Verify the permutation works correctly
                        relabeled_target = [sort([perm[vertex] for vertex in base]) for base in target_bases]
                        deletion_bases_sorted = [sort(collect(b)) for b in deletion_bases]
                        
                        relabeled_set = Set(relabeled_target)
                        deletion_set = Set(deletion_bases_sorted)
                        
                        if relabeled_set != deletion_set
                            @warn """Permutation verification failed at m=$m, k=$k, v=$v_chosen:
                                    Relabeled target_bases do not match deletion_bases.
                                    |relabeled| = $(length(relabeled_set))
                                    |deletion| = $(length(deletion_set))
                                    |intersection| = $(length(intersect(relabeled_set, deletion_set)))"""
                        end
                        
                        found_index = l
                        break
                    end
                end
                push!(Iso_DB[m][k],(found_index, perm))
            end
        end
    end
    return Iso_DB
end


mat_DB = open("rank_4_mat_DB_bin.jls", "r") do io
    deserialize(io)
end

iso_DB = Dict{Int,Dict{Int,Vector{Tuple{Int,Any}}}}()

build_iso_db!(iso_DB,mat_DB,ms=6:15,verbose=true)


open("rank_4_iso_DB_bin.jls", "w") do io
    serialize(io, iso_DB)
end