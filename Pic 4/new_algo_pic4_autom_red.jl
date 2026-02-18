using Oscar
using ProgressMeter
using SparseArrays
using Serialization
using Base.Threads
using LinearAlgebra
using Nemo
using Polymake

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

function euler_characteristic_sphere(top_facets::Vector{UInt16})
    isempty(top_facets) && return 0

    d = count_ones(top_facets[1]) - 1

    # faces_by_dim[i] will store i-faces
    faces_by_dim = Vector{Set{UInt16}}(undef, d+1)

    # Top dimension
    faces_by_dim[d+1] = Set(top_facets)

    # Generate all lower-dimensional faces
    for dim = d:-1:1
        current_faces = faces_by_dim[dim+1]
        lower_faces = Set{UInt16}()

        for f in current_faces
            x = f
            while x != 0
                i = trailing_zeros(x)
                push!(lower_faces, f ⊻ (UInt16(1) << i))
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

function euler_sphere_test(top_facets::Vector{UInt16})
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


function enumerate_kernel_with_constraints_bitvector(A::SparseMatrixCSC{Bool,Int}, B::Vector{BitVector}, S::BitVector)
    m, n = size(A)
    rows = sparse_rows(A)

    results = BitVector[]

    # Special case: no basis (only zero vector possible)
    if isempty(B)
        y = falses(n)
        if all(.!S) && check_Ay_is_02(rows, y)
            push!(results, copy(y))
        end
        return results
    end

    # 1) Propagate: any row with sum == 2 forbids all other columns that touch that row
    row_sums = zeros(Int, m)
    for i in 1:m
        s = 0
        for j in rows[i]
            s += S[j]
        end
        row_sums[i] = s
    end

    T_fixed = falses(n)
    for i in 1:m # THIS WORKS DO NOT CHANGE
        if row_sums[i] == 2
            for j in rows[i]
                # if column j is not already forced to 1, mark it forbidden
                if !S[j]
                    T_fixed[j] = true
                end
            end
        end
    end

    # 2) Compute echelon prioritizing S then T_fixed (final constraints for this single propagation)
    B_ech, pivots = kernel_basis_echelon_prioritize_with_constraints(B, S, T_fixed)
    k = length(B_ech)   # now k == rank
    @assert k <= 64     # assert on rank (or choose a different bound / strategy)
    @assert length(pivots) == k


    forced_one = falses(k)
    forced_zero = falses(k)
    free_indices = Int[]

    for i in 1:k
        piv = pivots[i]
        if S[piv]
            forced_one[i] = true
        elseif T_fixed[piv]
            forced_zero[i] = true
        else
            push!(free_indices, i)
        end
    end

    # x_bits encodes coefficients
    x_bits = UInt64(0)

    # y = Bx (mod 2)
    y = falses(n)

    # Apply forced ones to y
    for i in 1:k
        if forced_one[i]
            x_bits |= UInt64(1) << (i-1)
            y .⊻= B_ech[i]
        end
    end

    # Check immediate consistency with constraints S / T_fixed
    for j in 1:n
        if (S[j] && !y[j]) || (T_fixed[j] && y[j])
            return results
        end
    end

    num_free = length(free_indices)

    if num_free == 0
        if check_Ay_is_02(rows, y)
            push!(results, copy(y))
        end
        return results
    end

    sizehint!(results, min(1 << num_free, 1000))

    # first candidate
    if check_Ay_is_02(rows, y)
        push!(results, copy(y))
    end

    total = UInt64(1) << num_free

    # Gray code enumeration over free coefficients
    for i in UInt64(1):(total - UInt64(1))
        gray      = i ⊻ (i >> 1)
        gray_prev = (i - 1) ⊻ ((i - 1) >> 1)
        changed   = trailing_zeros(gray ⊻ gray_prev) + 1

        idx = free_indices[changed]

        # flip coefficient bit
        x_bits ⊻= UInt64(1) << (idx - 1)

        # update y incrementally
        y .⊻= B_ech[idx]

        if check_Ay_is_02(rows, y)
            push!(results, copy(y))
        end
    end
    return results
end

function relabel(facets_bin::Vector{UInt16},perm)
    rel_facets_bin = Vector{UInt16}()
    for facet_bin in facets_bin
        rel_facet_bin = UInt16(0)
        for (i,j) in perm
            if (facet_bin>>(i-1))&1==1
                rel_facet_bin |= UInt16(1)<<(j-1)
            end
        end
        push!(rel_facets_bin,copy(rel_facet_bin))
    end
    return rel_facets_bin
end



mat_DB_bin = open("rank_4_mat_DB_bin.jls", "r") do io
    deserialize(io)
end

iso_DB = open("rank_4_iso_DB_bin.jls", "r") do io
    deserialize(io)
end


function subset_bitvector(superset::Vector{UInt16}, subset::Vector{UInt16})
    S = Set(subset)
    return BitVector(x in S for x in superset)
end

function build_finalDB_single_v!(pseudo_manifolds_DB::Dict{Int,Vector{Set{BitVector}}},
                                  database_reduce_autom::Dict{Int,Vector{Set{BitVector}}},
                                  mat_DB::Dict{Int,Vector{Vector{UInt32}}},
                                  iso_DB::Dict{Int,Dict{Int,Vector{Tuple{Int,Any}}}},
                                  mmax; mstart=-1)
    mmin = minimum(collect(keys(mat_DB)))
    if mstart == -1
        mstart = mmin
    end

    for m = mstart:mmax
        println(m)
        pseudo_manifolds_DB[m] = Vector{Set{BitVector}}()
        database_reduce_autom[m] = Vector{Set{BitVector}}()

        for (l, bases_bin) in enumerate(mat_DB[m])
            V_bin = reduce(|, bases_bin)
            compl_bases_bin = [base ⊻ V_bin for base in bases_bin]
            ridges, A = boundary_incidence_facets_to_ridges(compl_bases_bin)
            kernel_basis = kernel_basis_mod2_sparse(A)

            # ── build automorphism group & facet permutations ──────────────────
            facets_M = [
                [i for i = 1:(8*sizeof(cobase)) if (cobase >> (i-1)) & 1 == 1]
                for cobase in compl_bases_bin
            ]
            M = simplicial_complex(facets_M)
            faces_list = collect(facets(M))

            facets_internal = Vector{UInt32}(undef, length(faces_list))
            for j in eachindex(faces_list)
                mask = UInt32(0)
                for v in faces_list[j]
                    mask |= UInt32(1) << (v - 1)
                end
                facets_internal[j] = mask
            end
            index = Dict(facets_internal[i] => i for i in eachindex(facets_internal))

            G = automorphism_group(M)
            all_autos = collect(elements(G))

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

            sigmas = Vector{Vector{Int}}(undef, length(all_autos))
            for i in eachindex(all_autos)
                g = all_autos[i]
                σ = Vector{Int}(undef, length(facets_internal))
                for j in eachindex(facets_internal)
                    mask_img = permute_facet(facets_internal[j], g)
                    if !haskey(index, mask_img)
                        lbls = Int[]
                        x = mask_img
                        while x != 0
                            v = trailing_zeros(x) + 1
                            push!(lbls, v)
                            x &= x - 1
                        end
                        error("permute_facet produced mask $(hex(mask_img)) not found in index; permuted facet labels = $lbls")
                    end
                    σ[j] = index[mask_img]
                end
                sigmas[i] = σ
            end

            function apply_perm(χ::BitVector, σ::Vector{Int})
                χ2 = falses(length(χ))
                @inbounds for ii in eachindex(χ)
                    if χ[ii]
                        χ2[σ[ii]] = true
                    end
                end
                return χ2
            end

            function canonical_rep(χ::BitVector)
                best = χ
                for σ in sigmas
                    χ2 = apply_perm(χ, σ)
                    if χ2 < best
                        best = χ2
                    end
                end
                return best
            end
            # ──────────────────────────────────────────────────────────────────

            push!(pseudo_manifolds_DB[m], Set{BitVector}())
            push!(database_reduce_autom[m], Set{BitVector}())

            # store unreduced for link iteration, reduced for output
            function try_insert!(K_bit::BitVector)
                facets_bin = compl_bases_bin[findall(K_bit)]
                if euler_sphere_test(facets_bin)
                    push!(pseudo_manifolds_DB[m][l], copy(K_bit))
                    push!(database_reduce_autom[m][l], canonical_rep(K_bit))
                end
            end

            if m == mmin
                mandatory_facets_bit = falses(length(bases_bin))
                all_solutions_bit = enumerate_kernel_with_constraints_bitvector(A, kernel_basis, mandatory_facets_bit)
                for K_bit in all_solutions_bit
                    try_insert!(K_bit)
                end
            else
                for (index_contraction, perm) in iso_DB[m][l]
                    @showprogress desc="Number of links $(length(pseudo_manifolds_DB[m-1][index_contraction]))" for L_bit in pseudo_manifolds_DB[m-1][index_contraction]
                        mandatory_facets_bin = relabel(mat_DB[m-1][index_contraction][findall(L_bit)], perm)
                        mandatory_facets_bit = subset_bitvector(bases_bin, mandatory_facets_bin)
                        if count(mandatory_facets_bit) != length(mandatory_facets_bin)
                            @warn "Some mandatory facets not found in bases!" m l mandatory_facets_bin
                        end
                        all_solutions_bit = enumerate_kernel_with_constraints_bitvector(A, kernel_basis, mandatory_facets_bit)
                        for K_bit in all_solutions_bit
                            try_insert!(K_bit)
                        end
                    end
                end
            end
        end
    end
end

pseudo_manifolds_DB = Dict{Int,Vector{Set{BitVector}}}()
database_reduce_autom = Dict{Int,Vector{Set{BitVector}}}()



mmax=15


build_finalDB_single_v!(pseudo_manifolds_DB,database_reduce_autom,mat_DB_bin,iso_DB,mmax)




database_before_iso = Dict{Tuple{Int,Int}, Set{Vector{UInt16}}}()

for m=6:mmax
    for (l,bases) in enumerate(mat_DB_bin[m])
        # display(bases)
        V = reduce(|,bases)
        compl_bases = [base⊻V for base in bases]
        @showprogress desc="for m=$(m) " for facets_bit in database_reduce_autom[m][l]
            facets_bin = compl_bases[findall(facets_bit)]
            nv_K = count_ones(reduce(|,facets_bin))
            d_K = count_ones(facets_bin[1])-1
            if !((d_K,nv_K) in keys(database_before_iso))
                database_before_iso[(d_K,nv_K)] = Set{Vector{UInt16}}()
            end
            push!(database_before_iso[(d_K,nv_K)],copy(sort(facets_bin)))
        end
    end
end

            
open("rank_4_db_before_iso_new.jls", "w") do io
    serialize(io, database_before_iso)
end


