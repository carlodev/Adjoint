using Gridap
using GridapGmsh
using GridapPETSc
using LineSearches: BackTracking
using CSV, DataFrames, Plots

using JLD2
model = GmshDiscreteModel("Cylinder2D.msh")
wall_name= "cylinder"

writevtk(model,"model")



D = 2
order = 1
θ = 1
dt = 0.1
t0 = 0.0
tF = 10.0
reffeᵤ = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
V = TestFESpace(model,reffeᵤ,conformity=:H1,dirichlet_tags= [wall_name, "inlet", "limits"])

reffeₚ = ReferenceFE(lagrangian,Float64,order)
Q = TestFESpace(model,reffeₚ,conformity=:H1, dirichlet_tags= ["outlet"])

u_wall = VectorValue(0,0)
u_inlet = VectorValue(1,0)
hf = VectorValue(0,0)

U = TrialFESpace(V,[u_wall,u_inlet,u_inlet])
P = TrialFESpace(Q, 0.0)

Y = MultiFieldFESpace([V, Q])
X = MultiFieldFESpace([U, P])

degree = order*4
Ω = Triangulation(model)
dΩ = Measure(Ω,degree)

Rm(t,(u, p)) =  ∂t(u) + u ⋅ ∇(u) + ∇(p) - hf;  #- ν*Δ(u)
dRm((u, p), (du, dp), (v, q)) = du ⋅ ∇(u) + u ⋅ ∇(du) + ∇(dp); #- ν*Δ(du)
Rc(u) = ∇ ⋅ u;
dRc(du) = ∇ ⋅ du;


var_eq(t,(u, p), (v, q)) = ∫(∂t(u) ⋅ v)dΩ + ∫((u ⋅ ∇(u)) ⋅ v)dΩ - ∫((∇ ⋅ v) * p)dΩ + ∫((q * (∇ ⋅ u)))dΩ + ν * ∫(∇(v) ⊙ ∇(u))dΩ - ∫(hf ⋅ v)dΩ

h = collect(lazy_map(h -> h^(1 / D), get_cell_measure(Ω)))
function τ(u, h)
    r = 1
    τ₂ = h^2 / (4 * ν)
    val(x) = x
    val(x::Gridap.Fields.ForwardDiff.Dual) = x.value
    u = val(norm(u))
    
    if iszero(u)
        return τ₂
    end
    τ₃ = dt / 2
    τ₁ = h / (2 * u)
    return 1 / (1 / τ₁^r + 1 / τ₂^r + 1 / τ₃^r )    
end

τb(u, h) = (u ⋅ u) * τ(u, h);
stab_eq(t,(u, p), (v, q)) = ∫((τ ∘ (u, h) * (u ⋅ ∇(v) + ∇(q))) ⊙ Rm(t,(u, p))    +   τb ∘ (u, h) * (∇ ⋅ v) ⊙ Rc(u) )dΩ;

res_eq(t,(u, p), (v, q)) = var_eq(t,(u, p), (v, q)) + stab_eq(t,(u, p), (v, q));


dvar_eq(t, (u, p), (du, dp), (v, q)) = ∫(((du ⋅ ∇(u)) ⋅ v) + ((u ⋅ ∇(du)) ⋅ v) + (∇(dp) ⋅ v) +  (q * (∇ ⋅ du)))dΩ + ν * ∫(∇(v) ⊙ ∇(du))dΩ
	
dstab_eq(t,(u, p), (du, dp), (v, q)) = ∫(((τ ∘ (u, h) * (u ⋅ ∇(v)' +  ∇(q))) ⊙ dRm((u, p), (du, dp), (v, q))) + ((τ ∘ (u, h) * (du ⋅ ∇(v)')) ⊙ Rm(t,(u, p))) + (τb ∘ (u, h) * (∇ ⋅ v) ⊙ dRc(du)))dΩ
	
jac(t,(u, p), (du, dp), (v, q)) = dvar_eq(t,(u, p), (du, dp), (v, q)) + dstab_eq(t,(u, p), (du, dp), (v, q))


jac_t(t, (u, p), (dut, dpt), (v, q)) = ∫(dut ⋅ v)dΩ + ∫(τ ∘ (u, h) * (u ⋅ ∇(v) + ∇(q)) ⊙ dut)dΩ


D = 1
ReD = 20
ν = D/ReD

uh0 = interpolate_everywhere(VectorValue(0.0,0.0), U)
ph0 = interpolate_everywhere(0.0, P)
xh0 = interpolate_everywhere([uh0,ph0], X)


op = TransientFEOperator(res_eq,jac,jac_t,X,Y)


uh_vec = Any[] 
ph_vec = Any[]
for t in t0:dt:tF
    push!(uh_vec, uh0.free_values)
    push!(ph_vec, ph0.free_values)

end

options = "-snes_type newtonls -snes_linesearch_type basic  -snes_linesearch_damping 0.7 -snes_rtol 1.0e-3 -snes_atol 0.0 -snes_monitor -pc_type asm -sub_pc_type lu  -ksp_type gmres -ksp_gmres_restart 30  -snes_converged_reason -ksp_converged_reason -ksp_error_if_not_converged true "

GridapPETSc.with(args=split(options)) do
    solver = PETScNonlinearSolver()
    ode_solver = ThetaMethod(solver, dt, θ)
    
    sol_t = solve(ode_solver, op, xh0, t0, tF)
    createpvd("Primal_transient_solution") do pvd
 
        iter = 1
        for (xh_tn,tn) in sol_t
            uh = xh_tn[1]
            ph = xh_tn[2]
            uh_vec[iter] = deepcopy(uh.free_values)
            ph_vec[iter] = deepcopy(ph.free_values)
            iter = iter +1
            pvd[tn] = createvtk(Ω,"Primal_solution/Primal_transient_solution_$tn"*".vtu",cellfields=["uh"=>uh, "ph"=>ph])
        end
      end

end


# jldsave("PrimalSolution.jld2"; uh_vec, ph_vec)

# jldopen("PrimalSolution.jld2")





Γ = BoundaryTriangulation(model; tags=[wall_name]) 
dΓ = Measure(Γ,degree)
n_Γ = - get_normal_vector(Γ)


r((n1,n2)) = VectorValue(-n2,n1)
t_Γ = r∘n_Γ #extract tangent
ff_t = (transpose(∇(uh))⋅n_Γ) ⋅ t_Γ


#Adjoint
d_lift = VectorValue(0,1)
d_drag = VectorValue(1,0)

reffe_u_adj = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
V_adj = TestFESpace(model,reffeᵤ,conformity=:H1,dirichlet_tags= [wall_name,"outlet","limits"])



reffe_p_adj = ReferenceFE(lagrangian,Float64,order)
Q_adj = TestFESpace(model,reffeₚ,conformity=:H1,dirichlet_tags= ["inlet"])

U_adj = TrialFESpace(V_adj,[- d_drag, VectorValue(0,0), VectorValue(0,0)])
P_adj = TrialFESpace(Q_adj,0.0)

Y_adj = MultiFieldFESpace([V_adj, Q_adj])
X_adj = MultiFieldFESpace([U_adj, P_adj])




uh = FEFunction(U,uh_vec[end])
ph = FEFunction(P,ph_vec[end])

Rm_adj(u, p) =  - (uh ⋅ ∇(u)) - (∇(u) ⋅ uh ) - ∇(p);  #- ν*Δ(u)

var_eq_adj(t,(uadj, padj), (v, q)) = -1* ∫((uh ⋅ ∇(uadj)) ⋅ v)dΩ + -1* ∫((∇(uadj) ⋅ uh ) ⋅ v)dΩ - 1* ∫((∇(padj)) ⋅ v )dΩ + -1* ∫((q * (∇ ⋅ uadj)))dΩ + ν * ∫(∇(v) ⊙ ∇(uadj))dΩ
stab_eq_adj(t,(uadj, padj), (v, q)) = ∫(((τ∘(uh_adj, h))* (∇(q)) ⊙ Rm_adj(uadj, padj) ))dΩ
res_eq_adj(t,(uadj, padj), (v, q)) = var_eq_adj(t,(uadj, padj), (v, q)) + stab_eq_adj(t,(uadj, padj), (v, q))
b_eq_adj(t,(v,q)) = ∫(hf ⋅ v)dΩ 
m(t, (uadj, padj), (v, q)) = - 1* ∫(uadj ⋅ v)dΩ #+ 1* ∫(((τ∘(uh_adj, h))* (∇(q)) ⊙ uadj ))dΩ


op_adj = TransientAffineFEOperator(m, var_eq_adj , b_eq_adj,X_adj,Y_adj)


xh0_adj = xh0
uh_adj = xh0_adj[1]
ph_adj = xh0_adj[2]

options_adj = "-pc_type jacobi -ksp_type gmres -ksp_gmres_restart 30  -snes_converged_reason -ksp_converged_reason -ksp_error_if_not_converged true"

GridapPETSc.with(args=split(options_adj)) do
    solver_adj = PETScLinearSolver()
    ode_solver_adj = ThetaMethod(solver_adj, dt, θ)
    
    sol_t_adj = solve(ode_solver_adj, op_adj, xh0_adj, t0, tF)
    createpvd("Adjoint_transient_solution") do pvd
        # global uh_vec, ph_vec    
        iter = 1
        for (xh_tn_adj,tn) in sol_t_adj
            uh_adj = xh_tn_adj[1]
            ph_adj = xh_tn_adj[2]

            uh = FEFunction(U,uh_vec[end-iter])
            ph = FEFunction(P,ph_vec[end-iter])
            tn_adj = (tF - tn) + t0
            pvd[tn] = createvtk(Ω,"Adjoint_solution/Adjoint_transient_solution_$(tn_adj)"*".vtu",cellfields=["uh_adj"=>uh_adj, "ph"=>ph_adj])
            iter = iter +1
        end
      end
end


writevtk(Γ,"results_adj_surface",cellfields=["n_Γ"=> n_Γ, "dudb"=>uh_adj])

dLdb = -1 * ∫(ph_adj⋅ n_Γ⋅(n_Γ⋅∇(uh)))dΓ -1 *∫(ν*n_Γ⋅(∇(uh_adj)+ (∇(uh_adj))')⋅ (n_Γ⋅∇(uh)) )dΓ
dLdb_eval = sum(dLdb1) 

dLdb2 = -1*∫((ν.* n_Γ⋅(∇(uh_adj)) - ph_adj⋅n_Γ)⋅(∇(uh))⋅d_drag)dΓ
dLdb_eval2 = sum(dLdb2) 

df = DataFrame(CSV.File("Airfoil_Point_Gamma.csv")) 
top_ff_idx = findall(x->x>0, df.n_Γ_1) 
bottom_ff_idx = findall(x->x<0, df.n_Γ_1) 

dLdbb(bi) = -1*∫(((ν.* n_Γ⋅(∇(uh_adj)) - ph_adj⋅n_Γ)⋅(∇(uh)))⋅ (bi))dΓ






vec_top_normal = Any[]

for (i,j) in zip(df.n_Γ_0[top_ff_idx],    df.n_Γ_1[top_ff_idx])
    push!(vec_top_normal, VectorValue(i,j))
end



dLdb_top = Float64[]
for v in vec_top_normal
    sum(dLdbb(v))
push!(dLdb_top,sum(dLdbb(v)))
end



vec_bottom_normal = Any[]

for (i,j) in zip(df.n_Γ_0[top_ff_idx],    df.n_Γ_1[bottom_ff_idx])
    push!(vec_bottom_normal, VectorValue(i,j))
end



dLdb_bottom = Float64[]
for v in vec_bottom_normal
    sum(dLdbb(v))
push!(dLdb_bottom,sum(dLdbb(v)))
end


scatter(df.Points_0[top_ff_idx],dLdb_top, label = "top")
scatter!(df.Points_0[bottom_ff_idx],dLdb_bottom, label = "bottom")

scatter(df.Points_0[top_ff_idx],df.dudb_1[top_ff_idx], label = "top")
scatter!(df.Points_0[bottom_ff_idx],df.dudb_1[top_ff_idx], label = "bottom")
