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
                i = ridge_dict[r]
                push!(I, i)
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
        for ptr in A.colptr[col]:(A.colptr[col+1]-1)
            row = A.rowval[ptr]
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
Compute the fixpoint of constraints by alternating:
- S (must be 1) → find saturated rows → T (must be 0)
- T (must be 0) → find rows at 1 → U (must be 1 to reach 2)
- U → new T, etc.
"""
function compute_constraint_fixpoint(A, B, S_init)
    m, n = size(A)
    
    S = copy(S_init)
    T = falses(n)
    
    max_iterations = 1  # Prevent infinite loops
    
    for iter in 1:max_iterations
        S_old = copy(S)
        T_old = copy(T)
        
        # Compute what x would be if we only set S positions
        x_partial = compute_partial_solution(B, S, T)
        
        if isnothing(x_partial)
            # Constraints are inconsistent
            return (S, T)
        end
        
        # Compute A * x_partial
        Ax = zeros(Int, m)
        for j in 1:n
            if x_partial[j]
                for p in nzrange(A, j)
                    i = rowvals(A)[p]
                    Ax[i] += A.nzval[p]
                end
            end
        end
        
        # Better: iterate over columns
        for j in 1:n
            if S[j] || T[j]
                continue  # Already constrained
            end
            
            # Check if j must be 0 (appears in saturated row)
            must_be_zero = false
            for p in nzrange(A, j)
                i = rowvals(A)[p]
                if Ax[i] == 2 && A.nzval[p] != 0
                    must_be_zero = true
                    break
                end
            end
            
            if must_be_zero
                T[j] = true
                continue
            end
            
            # Check if j must be 1 (needed to saturate a row at 1)
            must_be_one = false
            for p in nzrange(A, j)
                i = rowvals(A)[p]
                if Ax[i] == 1 && A.nzval[p] != 0
                    # Check if j is the ONLY way to saturate row i
                    # Count how many unset columns could saturate row i
                    candidates = 0
                    for p2 in nzrange(A, i)  # Wrong - need CSR
                        # This is CSC, need to find all j' where A[i,j'] != 0
                    end
                    # For now, mark as must be 1 if it's in a row with value 1
                    must_be_one = true
                    break
                end
            end
            
            if must_be_one
                S[j] = true
            end
        end
        
        # Check convergence
        if S == S_old && T == T_old
            break
        end
    end
    
    return (S, T)
end

"""
Compute a partial solution satisfying S (must be 1) and T (must be 0) constraints.
Returns nothing if constraints are inconsistent with kernel.
"""
function compute_partial_solution(B, S, T)
    isempty(B) && return falses(length(S))
    
    n = length(B[1])
    
    # Use echelon form to find a solution
    B_ech, pivots = kernel_basis_echelon_prioritize_with_constraints(B, S, T)
    
    x = falses(n)
    
    # Set coefficients based on constraints
    for i in 1:length(pivots)
        piv = pivots[i]
        if S[piv]
            x .⊻= B_ech[i]
        elseif T[piv]
            # Coefficient is 0, don't add
        else
            # Free - leave at 0 for partial solution
        end
    end
    
    # Check if S constraints are satisfied
    for j in 1:n
        if S[j] && !x[j]
            return nothing
        end
        if T[j] && x[j]
            return nothing
        end
    end
    
    return x
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
    cols_in_S = [j for j in 1:n if S[j]]
    cols_in_T = [j for j in 1:n if !S[j] && T[j]]
    cols_other = [j for j in 1:n if !S[j] && !T[j]]
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
        
        # Eliminate
        for r in 1:k
            if r != current_row && B_ech[r][col]
                B_ech[r] .⊻= B_ech[current_row]
            end
        end
        
        current_row += 1
    end
    
    return (B_ech, pivots)
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

    # 1) Build an initial echelon prioritizing S only, to get the forced part of y
    B_ech_init, pivots_init = kernel_basis_echelon_prioritize_with_constraints(B, S, falses(n))

    k_init = length(B_ech_init)
    @assert k_init ≤ 64

    # Build initial forced y from pivots that correspond to S
    y_init = falses(n)
    for i in 1:k_init
        piv = pivots_init[i]
        if S[piv]
            y_init .⊻= B_ech_init[i]
        end
    end

    # 2) Propagate: any row with sum == 2 forbids all other columns that touch that row
    row_sums = zeros(Int, m)
    for i in 1:m
        s = 0
        for j in rows[i]
            s += S[j]
            if s > 2
                s = 3  # mark >2 (will break the constraint later)
                break
            end
        end
        row_sums[i] = s
    end

    T_fixed = falses(n)
    for i in 1:m
        if row_sums[i] == 2
            for j in rows[i]
                # if column j is not already forced to 1, mark it forbidden
                if !S[j]
                    T_fixed[j] = true
                end
            end
        end
    end

    # Conflict: a column both forced 1 and forbidden -> no solutions
    if any(S .& T_fixed)
        return results
    end

    # 3) Recompute echelon prioritizing S then T_fixed (final constraints for this single propagation)
    B_ech, pivots = kernel_basis_echelon_prioritize_with_constraints(B, S, T_fixed)

    k = length(B_ech)
    @assert k ≤ 64

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

    # # Check immediate consistency with constraints S / T_fixed
    # for j in 1:n
    #     if (S[j] && !y[j]) || (T_fixed[j] && y[j])
    #         return results
    #     end
    # end

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
    for i in UInt64(1):(total - 1)
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



global mat_DB_bin = open("rank_4_simple_bin_mat_DB_bin.jls", "r") do io
    deserialize(io)
end

global iso_DB = open("rank_4_iso_DB_7-15_bin.jls", "r") do io
    deserialize(io)
end

function subset_bitvector(superset::Vector{UInt16}, subset::Vector{UInt16})
    S = Set(subset)
    return BitVector(x in S for x in superset)
end


function build_finalDB_single_v!(pseudo_manifolds_DB::Dict{Int,Vector{Set{BitVector}}},mat_DB::Dict{Int,Vector{Vector{UInt16}}},iso_DB::Dict{Int,Dict{Int,Tuple{Int,Int,Any}}},mmax;mstart=-1)
    mmin = minimum(collect(keys(mat_DB)))
    if mstart == -1
        mstart = mmin
    end
    for m=mstart:mmax
        println(m)
        pseudo_manifolds_DB[m] = Vector{Set{BitVector}}()
        for (l,bases) in enumerate(mat_DB[m])
            # display(bases)
            V = reduce(|,bases)
            compl_bases = [base⊻V for base in bases] # we need to complement to get the correct boundary matrix
            ridges, A = boundary_incidence_facets_to_ridges(compl_bases)  
            basis = kernel_basis_mod2_sparse(A)
            push!(pseudo_manifolds_DB[m], Set{BitVector}())
            if m==mmin
                mandatory_facets_bit=falses(length(bases))
                all_solutions_bit = enumerate_kernel_with_constraints_bitvector(A,basis,mandatory_facets_bit)
                for K_bit in all_solutions_bit
                    facets_bin = [facet_bin for facet_bin in compl_bases[findall(K_bit)]]
                    if euler_sphere_test(facets_bin)
                        push!(pseudo_manifolds_DB[m][l],K_bit)
                    end
                    # if is_mod2_sphere(facets_bin)
                    #     push!(pseudo_manifolds_DB[m][l],K_bit)
                    # end
                end
            else
                index_contraction, v_contract, perm = iso_DB[m][l]
                @showprogress desc="Number of links $(length(pseudo_manifolds_DB[m-1][index_contraction]))" for L in pseudo_manifolds_DB[m-1][index_contraction]
                    mandatory_facets =Vector{UInt16}()
                    # println(perm)
                    for facet_L in mat_DB[m-1][index_contraction][findall(L)]
                        facet_bin = UInt16(0)
                        for (i,j) in perm
                            if (facet_L>>(j-1))&1==1
                                facet_bin |= UInt16(1)<<(i-1)
                            end
                        end
                        push!(mandatory_facets,facet_bin)
                    end
                    # println("facets:",[[i for i=1:(8*sizeof(facet)) if (facet>>(i-1))&1==1] for facet in mandatory_facets])
                    # println("bases:",[[i for i=1:(8*sizeof(base)) if (base>>(i-1))&1==1] for base in bases])
                    

                    # println("facets:",mandatory_facets)
                    # println("bases:",bases)
                    sort!(mandatory_facets)
                    mandatory_facets_bit = subset_bitvector(bases, mandatory_facets)
                    t1 = time()
                    all_solutions_bit = enumerate_kernel_with_constraints_bitvector(A,basis,mandatory_facets_bit)
                    if (time() - t1)> 1
                        println("Enum: ", time() - t1, " seconds")
                    end
                    if m<16
                        for K_bit in all_solutions_bit
                            facets_bin = [facet_bin for facet_bin in compl_bases[findall(K_bit)]]
                            if euler_sphere_test(facets_bin)
                                push!(pseudo_manifolds_DB[m][l],K_bit)
                            end
                            # if is_mod2_sphere(facets_bin)
                            #     push!(pseudo_manifolds_DB[m][l],K_bit)
                            # end
                        end
                    else
                        union!(pseudo_manifolds_DB[m][l],all_solutions_bit)
                    end
                end
            end
        end
    end
end

global pseudo_manifolds_DB = Dict{Int,Vector{Set{BitVector}}}()


# open("Pic_4_DB_6-9.jls", "w") do io
#     serialize(io, pseudo_manifolds_DB)
# end

# global pseudo_manifolds_DB = open("Pic_4_DB_6-9.jls", "r") do io
#     deserialize(io)
# end

build_finalDB_single_v!(pseudo_manifolds_DB,mat_DB_bin,iso_DB,15)

for (m,data_psdmfd) in enumerate(pseudo_manifolds_DB)
    println("for %(m):",sum([length(data) for data in data_psdmfd]))
end
