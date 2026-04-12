using Oscar
using ProgressMeter
using SparseArrays
using Serialization
using Base.Threads
using LinearAlgebra
using Nemo
using Polymake

const F2 = GF(2)


function boundary_incidence_facets_to_ridges(facets::Vector{UInt32})
    # collect ridges (each facet contributes its (d-1)-subfaces by deleting one vertex)
    ridge_dict = Dict{UInt32, Int}()  # ridge -> row index
    ridges = Vector{UInt32}()
    for f in facets
        for i=0:((8*sizeof(f))-1)
            if (f>>i)&1==1
                r = f ⊻ (UInt32(1)<<i)
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
                r = f ⊻ (UInt32(1)<<i)
                row_idx = ridge_dict[r]  
                push!(I, row_idx)
                push!(J, j)
            end
        end
    end

    A = sparse(I, J, true, m, n)  # SparseMatrixCSC{Bool} 

    return ridges, A
end

function euler_characteristic_sphere(top_facets::Vector{UInt32})
    isempty(top_facets) && return 0

    d = count_ones(top_facets[1]) - 1

    # faces_by_dim[i] will store i-faces
    faces_by_dim = Vector{Set{UInt32}}(undef, d+1)

    # Top dimension
    faces_by_dim[d+1] = Set(top_facets)

    # Generate all lower-dimensional faces
    for dim = d:-1:1
        current_faces = faces_by_dim[dim+1]
        lower_faces = Set{UInt32}()

        for f in current_faces
            x = f
            while x != 0
                i = trailing_zeros(x)
                push!(lower_faces, f ⊻ (UInt32(1) << i))
                x &= x - 1
            end
        end

        faces_by_dim[dim] = lower_faces
    end

    # Compute χ = Σ (-1)^i f_i
    χ = 0
    for i in 0:d
        χ += (-1)^i * length(faces_by_dim[i+1])
    end

    return χ
end

function euler_sphere_test(top_facets::Vector{UInt32})
    isempty(top_facets) && return false
    d = count_ones(top_facets[1]) - 1
    χ = euler_characteristic_sphere(top_facets)
    return χ == 1 + (-1)^d
end


function kernel_basis_mod2_sparse(A)
    m, n = size(A)

    # Store rows as sparse BitVectors (using sets of column indices)
    rows = [Set{Int}() for _ in 1:m]
    
    # Build rows from A mod 2
    @inbounds for j in 1:n
        for p in nzrange(A, j)
            i = rowvals(A)[p]
            if isodd(A.nzval[p])
                if j in rows[i]
                    delete!(rows[i], j)  # XOR: 1⊕1=0
                else
                    push!(rows[i], j)     # XOR: 0⊕1=1
                end
            end
        end
    end

    # RREF over GF(2), recording pivot columns and pivot rows
    pivcol = Int[]
    pivrow = Int[]
    row = 1
    
    @inbounds for col in 1:n
        row > m && break

        # Find pivot in this column at/under current row
        piv = 0
        for r in row:m
            if col in rows[r]
                piv = r
                break
            end
        end
        piv == 0 && continue

        # Swap rows
        if piv != row
            rows[row], rows[piv] = rows[piv], rows[row]
        end

        push!(pivcol, col)
        push!(pivrow, row)

        # Eliminate this column in all other rows (RREF)
        pivot_row = rows[row]
        for r in 1:m
            if r != row && col in rows[r]
                # XOR rows[r] with pivot_row
                symdiff!(rows[r], pivot_row)
            end
        end

        row += 1
    end

    # Free columns
    pivot_set = Set(pivcol)
    free_cols = [j for j in 1:n if !(j in pivot_set)]

    basis = BitVector[]
    isempty(free_cols) && return basis

    # Build one kernel vector per free column
    @inbounds for f in free_cols
        x_set = Set{Int}([f])  # Sparse representation of x

        # Back-substitute using RREF rows
        for t in length(pivcol):-1:1
            c = pivcol[t]
            r = pivrow[t]
            
            # Compute parity: count elements in rows[r] ∩ x_set, excluding c
            row_r = rows[r]
            parity = false
            for j in row_r
                if j != c && j in x_set
                    parity = !parity
                end
            end
            
            # Set x[c] based on parity
            if parity
                push!(x_set, c)
            else
                delete!(x_set, c)
            end
        end

        # Convert sparse set to BitVector
        x = falses(n)
        for j in x_set
            x[j] = true
        end
        push!(basis, x)
    end

    return basis
end

"""
Echelon form prioritizing: S columns (must be 1), then T columns (must be 0), then others.
"""
function kernel_basis_echelon_prioritize_with_constraints(B, S, T)
    isempty(B) && return (BitVector[], Int[])

    n = length(B[1])
    k = length(B)

    # Copy basis vectors
    B_ech = [copy(b) for b in B]
    pivots = Int[]

    # Build column order: S first, then T, then others
    cols_in_S = findall(S)
    cols_in_T = findall(T)
    cols_other = findall(.!(S .| T))
    col_order = vcat(cols_in_S, cols_in_T, cols_other)

    current_row = 1
    for col in col_order
        current_row > k && break

        # Find a vector with 1 at position col
        piv = 0
        for r in current_row:k
            if B_ech[r][col]
                piv = r
                break
            end
        end

        piv == 0 && continue

        # Swap
        if piv != current_row
            B_ech[current_row], B_ech[piv] = B_ech[piv], B_ech[current_row]
        end

        push!(pivots, col)

        # Eliminate other rows
        for r in 1:k
            if r != current_row && B_ech[r][col]
                B_ech[r] .⊻= B_ech[current_row]
            end
        end

        current_row += 1
    end

    # Trim B_ech to independent rows only
    rank = length(pivots)
    if rank == 0
        return (BitVector[], Int[])
    else
        return (B_ech[1:rank], pivots)
    end
end


function sparse_rows(A::SparseMatrixCSC{Bool,Int})
    m, n = size(A)
    rows = [Int[] for _ in 1:m]

    @inbounds for col in 1:n
        for ptr in A.colptr[col]:(A.colptr[col+1]-1)
            push!(rows[A.rowval[ptr]], col)
        end
    end
    return rows
end

@inline function check_Ay_is_02(rows, y::BitVector)
    @inbounds for cols in rows
        s = 0
        for j in cols
            s += y[j]
            if s > 2
                return false
            end
        end
        if !(s == 0 || s == 2)
            return false
        end
    end
    return true
end



# Part 1: Setup and determine num_free (cheap, just preprocessing)
function prepare_kernel_enumeration(A::SparseMatrixCSC{Bool,Int}, B::Vector{BitVector}, S::BitVector)
    m, n = size(A)
    rows = sparse_rows(A)

    if isempty(B)
        y = falses(n)
        if all(.!S) && check_Ay_is_02(rows, y)
            # Return special tuple indicating single solution with num_free=0
            return (nothing, nothing, Int[], y, rows, Vector{Vector{Tuple{Int,Vector{Int}}}}[], 0, true)  # added flag
        else
            return nothing  # infeasible
        end
    end

    S = copy(S)
    T = falses(n)

    # ── unit propagation ──────────────────────────────────────────────────
    changed = true
    while changed
        changed = false
        for i in 1:m
            s = 0
            free_cols = Int[]
            for j in rows[i]
                if S[j];      s += 1
                elseif !T[j]; push!(free_cols, j)
                end
            end
            s > 2 && return nothing
            if s == 2
                for j in free_cols
                    if !T[j]; T[j] = true; changed = true; end
                end
            elseif s == 1
                isempty(free_cols) && return nothing
                if length(free_cols) == 1
                    if !S[free_cols[1]]; S[free_cols[1]] = true; changed = true; end
                end
            elseif s == 0 && length(free_cols) == 1
                if !T[free_cols[1]]; T[free_cols[1]] = true; changed = true; end
            end
        end
    end
    # ─────────────────────────────────────────────────────────────────────

    B_ech, pivots = kernel_basis_echelon_prioritize_with_constraints(B, S, T)
    k = length(B_ech)
    @assert k <= 64
    @assert length(pivots) == k

    forced_one   = falses(k)
    free_indices = Int[]

    for i in 1:k
        piv = pivots[i]
        if S[piv]
            forced_one[i] = true
        elseif !T[piv]
            push!(free_indices, i)
        end
    end

    y = falses(n)
    for i in 1:k
        forced_one[i] && (y .⊻= B_ech[i])
    end

    for j in 1:n
        ((S[j] && !y[j]) || (T[j] && y[j])) && return nothing
    end

    num_free = length(free_indices)
    if num_free>=40
        @warn "Number of free variables is $num_free, which may lead to very long enumeration times."
    end

    # precompute per-free-variable row support
    free_row_support = Vector{Vector{Tuple{Int,Vector{Int}}}}(undef, num_free)
    for fi in 1:num_free
        bv = B_ech[free_indices[fi]]
        support = Tuple{Int,Vector{Int}}[]
        for r in 1:m
            cols = Int[]
            for j in rows[r]; bv[j] && push!(cols, j); end
            isempty(cols) || push!(support, (r, cols))
        end
        free_row_support[fi] = support
    end

    return (B_ech, pivots, free_indices, y, rows, free_row_support, num_free, false)  # added flag
end

# Part 2: Gray code enumeration (expensive)
function enumerate_from_prepared(prep_result)
    (B_ech, pivots, free_indices, y_forced, rows, free_row_support, num_free, is_empty_basis) = prep_result
    
    results = BitVector[]
    m = length(rows)
    n = length(y_forced)

    if is_empty_basis || num_free == 0
        rs = zeros(Int, m)
        for r in 1:m; for j in rows[r]; rs[r] += y_forced[j]; end; end
        all(s -> s == 0 || s == 2, rs) && push!(results, copy(y_forced))
        return results
    end

    y = copy(y_forced)
    row_sums = zeros(Int, m)
    for r in 1:m; for j in rows[r]; row_sums[r] += y[j]; end; end

    @inline function check_row_sums()
        @inbounds for s in row_sums
            (s == 0 || s == 2) || return false
        end
        return true
    end

    sizehint!(results, min(1 << num_free, 1000))
    check_row_sums() && push!(results, copy(y))

    total = UInt64(1) << num_free

    for i in UInt64(1):(total - UInt64(1))
        gray      = i ⊻ (i >> 1)
        gray_prev = (i - 1) ⊻ ((i - 1) >> 1)
        fi = trailing_zeros(gray ⊻ gray_prev) + 1
        idx = free_indices[fi]

        y .⊻= B_ech[idx]
        for (r, cols) in free_row_support[fi]
            for j in cols
                row_sums[r] += y[j] ? 1 : -1
            end
        end

        check_row_sums() && push!(results, copy(y))
    end

    return results
end

# Part 2: Gray code enumeration (expensive)
function enumerate_from_prepared_parallel(prep_result)
    (B_ech, pivots, free_indices, y_forced, rows, free_row_support, num_free, is_empty_basis) = prep_result

    m_rows = length(rows)
    n      = length(y_forced)

    # ── cas dégénérés ────────────────────────────────────────────────────────
    if is_empty_basis || num_free == 0
        rs = zeros(Int, m_rows)
        for r in 1:m_rows; for j in rows[r]; rs[r] += y_forced[j]; end; end
        result = BitVector[]
        all(s -> s == 0 || s == 2, rs) && push!(result, copy(y_forced))
        return result
    end

    # ── découpage en blocs ───────────────────────────────────────────────────
    num_threads = Threads.nthreads()
    prefix_bits = min(ceil(Int, log2(max(num_threads, 2))), num_free - 1)

    if prefix_bits == 0 || num_free <= prefix_bits
        return enumerate_from_prepared(prep_result)
    end

    num_blocks  = 1 << prefix_bits
    suffix_bits = num_free - prefix_bits
    block_size  = UInt64(1) << suffix_bits

    thread_results = [BitVector[] for _ in 1:num_blocks]

    Threads.@threads for block_idx in 0:(num_blocks - 1)
        # ── `let` isole toutes les variables du thread ───────────────────────
        let block_idx = block_idx,
            local_results = thread_results[block_idx + 1],
            # copies locales au thread des données mutables
            y        = copy(y_forced),
            row_sums = zeros(Int, m_rows)

            # ── état initial : code de Gray à l'indice `start` ──────────────
            start      = UInt64(block_idx) * block_size
            gray_start = start ⊻ (start >> 1)

            for fi in 1:num_free
                if (gray_start >> (fi - 1)) & UInt64(1) == UInt64(1)
                    y .⊻= B_ech[free_indices[fi]]
                end
            end

            for r in 1:m_rows
                @inbounds for j in rows[r]
                    row_sums[r] += y[j]
                end
            end

            # ── check inliné (pas de closure capturante) ─────────────────────
            valid = true
            @inbounds for s in row_sums
                if s != 0 && s != 2; valid = false; break; end
            end
            valid && push!(local_results, copy(y))

            # ── énumération par code de Gray dans le bloc ────────────────────
            stop = start + block_size - UInt64(1)
            for i in (start + UInt64(1)):stop
                gray      =  i      ⊻ ( i      >> 1)
                gray_prev = (i - 1) ⊻ ((i - 1) >> 1)
                fi  = trailing_zeros(gray ⊻ gray_prev) + 1
                idx = free_indices[fi]

                y .⊻= B_ech[idx]
                @inbounds for (r, cols) in free_row_support[fi]
                    for j in cols
                        row_sums[r] += y[j] ? 1 : -1
                    end
                end

                valid = true
                @inbounds for s in row_sums
                    if s != 0 && s != 2; valid = false; break; end
                end
                valid && push!(local_results, copy(y))
            end
        end # let
    end

    return reduce(vcat, thread_results)
end

# Wrapper that calls both (for backward compatibility)
function enumerate_kernel_with_constraints_bitvector(A::SparseMatrixCSC{Bool,Int}, B::Vector{BitVector}, S::BitVector)
    prep = prepare_kernel_enumeration(A, B, S)
    prep === nothing && return BitVector[]
    
    return enumerate_from_prepared_parallel(prep)
end

function relabel(facets_bin::Vector{UInt32},perm)
    rel_facets_bin = Vector{UInt32}()
    for facet_bin in facets_bin
        rel_facet_bin = UInt32(0)
        for (i,j) in perm
            if (facet_bin>>(i-1))&1==1
                rel_facet_bin |= UInt32(1)<<(j-1)
            end
        end
        push!(rel_facets_bin,copy(rel_facet_bin))
    end
    return rel_facets_bin
end

function relabel(facets_bin::Vector{UInt32},perm)
    rel_facets_bin = Vector{UInt32}()
    for facet_bin in facets_bin
        rel_facet_bin = UInt32(0)
        for (i,j) in perm
            if (facet_bin>>(i-1))&1==1
                rel_facet_bin |= UInt32(1)<<(j-1)
            end
        end
        push!(rel_facets_bin,copy(rel_facet_bin))
    end
    return rel_facets_bin
end



mat_DB_bin = open("rank_5_mat_DB_bin.jls", "r") do io
    deserialize(io)
end

iso_DB = open("rank_5_iso_DB_bin.jls", "r") do io
    deserialize(io)
end


function subset_bitvector(superset::Vector{UInt32}, subset::Vector{UInt32})
    S = Set(subset)
    return BitVector(x in S for x in superset)
end


function build_finalDB_single_v!(pseudo_manifolds_DB::Dict{Int,Vector{Set{BitVector}}},mat_DB::Dict{Int,Vector{Vector{UInt32}}},iso_DB::Dict{Int,Dict{Int,Vector{Tuple{Int,Any}}}},mmax;mstart=-1,list_links=[])
    mmin = minimum(collect(keys(mat_DB)))
    if mstart == -1
        mstart = mmin
    end
    for m=mstart:mmax
        println(m)
        pseudo_manifolds_DB[m] = Vector{Set{BitVector}}()
        for (l,bases_bin) in enumerate(mat_DB[m])
            # display(bases)
            V_bin = reduce(|,bases_bin)
            compl_bases_bin = [base⊻V_bin for base in bases_bin] # we need to complement to get the correct boundary matrix
            ridges, A = boundary_incidence_facets_to_ridges(compl_bases_bin)  
            kernel_basis = kernel_basis_mod2_sparse(A)
            push!(pseudo_manifolds_DB[m], Set{BitVector}())
            if m==mmin # Base case
                mandatory_facets_bit=falses(length(bases_bin))
                all_solutions_bit = enumerate_kernel_with_constraints_bitvector(A,kernel_basis,mandatory_facets_bit)
                for K_bit in all_solutions_bit
                    facets_bin = [facet_bin for facet_bin in compl_bases_bin[findall(K_bit)]]
                    if euler_sphere_test(facets_bin)
                        push!(pseudo_manifolds_DB[m][l],copy(K_bit))
                    end
                end
            else
                for (index_contraction, perm) in iso_DB[m][l] # find by computing the links
                    @showprogress desc="Number of links $(length(pseudo_manifolds_DB[m-1][index_contraction]))" for L_bit in pseudo_manifolds_DB[m-1][index_contraction]
                        mandatory_facets_bin = relabel(mat_DB[m-1][index_contraction][findall(L_bit)], perm) # for every potential link
                        mandatory_facets_bit = subset_bitvector(bases_bin, mandatory_facets_bin)
                        if count(mandatory_facets_bit) != length(mandatory_facets_bin)
                            @warn "Some mandatory facets not found in bases!" m l mandatory_facets_bin
                        end
                        all_solutions_bit = enumerate_kernel_with_constraints_bitvector(A,kernel_basis,mandatory_facets_bit)
                        for K_bit in all_solutions_bit
                            facets_bin = [facet_bin for facet_bin in compl_bases_bin[findall(K_bit)]]
                            if euler_sphere_test(facets_bin)
                                push!(pseudo_manifolds_DB[m][l],copy(K_bit))
                            end
                        end
                    end
                end
            end
        end
    end
end

pseudo_manifolds_DB = Dict{Int,Vector{Set{BitVector}}}()



mmax=10


build_finalDB_single_v!(pseudo_manifolds_DB,mat_DB_bin,iso_DB,mmax)



database_reduce_autom = Dict{Int,Vector{Set{BitVector}}}()


for m = 7:mmax
    database_reduce_autom[m] = Vector{Set{BitVector}}()

    for (l,bases) in enumerate(mat_DB_bin[m])
        push!(database_reduce_autom[m], Set{BitVector}())

        V_bin = reduce(|, bases)
        compl_bases = [base ⊻ V_bin for base in bases]

        # original facet lists (your input form)
        facets_M = [
            [i for i = 1:(8*sizeof(cobase)) if (cobase >> (i-1)) & 1 == 1]
            for cobase in compl_bases
        ]

        # build Oscar complex
        M = simplicial_complex(facets_M)

        faces_list = collect(facets(M))

        facets_internal = Vector{UInt32}(undef, length(faces_list))

        for j in eachindex(faces_list)
            mask = UInt32(0)
            for v in faces_list[j]      # v is already 1..n internally
                mask |= UInt32(1) << (v - 1)
            end
            facets_internal[j] = mask
        end

        index = Dict(facets_internal[i] => i for i in eachindex(facets_internal))

        # automorphism group: full list of elements
        G = automorphism_group(M)
        all_autos = collect(elements(G))

        # inline helper to permute a single facet-mask using a vertex permutation g
        @inline function permute_facet(mask::UInt32, g)
            h = UInt32(0)
            x = mask
            while x != 0
                v = trailing_zeros(x) + 1
                h |= UInt32(1) << (g(v) - 1)
                x &= x - 1
            end
            return h
        end

        # build all induced facet-permutations (full group)
        sigmas = Vector{Vector{Int}}(undef, length(all_autos))
        for i in eachindex(all_autos)
            g = all_autos[i]
            σ = Vector{Int}(undef, length(facets_internal))
            for j in eachindex(facets_internal)
                mask_img = permute_facet(facets_internal[j], g)
                if !haskey(index, mask_img)
                    # helpful debug print — shows the missing mask in hex and the corresponding vertex labels
                    lbls = Int[]
                    x = mask_img
                    while x != 0
                        v = trailing_zeros(x) + 1
                        push!(lbls, index_to_label[v])
                        x &= x - 1
                    end
                    error("permute_facet produced mask $(hex(mask_img)) not found in index; permuted facet labels = $lbls")
                end
                σ[j] = index[mask_img]
            end
            sigmas[i] = σ
        end

        # permutation application and canonical representative (lexicographic on BitVector)
        function apply_perm(χ::BitVector, σ::Vector{Int})
            χ2 = falses(length(χ))
            @inbounds for ii in eachindex(χ)
                if χ[ii]
                    χ2[σ[ii]] = true
                end
            end
            return χ2
        end

        function canonical_rep(χ::BitVector, sigmas::Vector{Vector{Int}})
            best = χ
            for σ in sigmas
                χ2 = apply_perm(χ, σ)
                if χ2 < best
                    best = χ2
                end
            end
            return best
        end

        # reduction
        @showprogress for χ in pseudo_manifolds_DB[m][l]
            χcanon = canonical_rep(χ, sigmas)
            push!(database_reduce_autom[m][l], χcanon)
        end
    end
end




database_before_iso = Dict{Tuple{Int,Int}, Set{Vector{UInt32}}}()

for m=7:mmax
    for (l,bases) in enumerate(mat_DB_bin[m])
        # display(bases)
        V = reduce(|,bases)
        compl_bases = [base⊻V for base in bases]
        @showprogress desc="for m=$(m) " for facets_bit in database_reduce_autom[m][l]
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


# open("Pic_4_DB_6-15_test7.jls", "w") do io
#     serialize(io, pseudo_manifolds_DB)
# end

