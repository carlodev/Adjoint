using Gridap
using GridapGmsh
using GridapPETSc
using LineSearches: BackTracking
using CSV, DataFrames, Plots
model = GmshDiscreteModel("Cylinder2D.msh")
wall_name= "cylinder"

writevtk(model,"model")



D = 2
order = 1
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

Rm(u, p) = u ⋅ ∇(u) + ∇(p) - hf;  #- ν*Δ(u)
dRm((u, p), (du, dp), (v, q)) = du ⋅ ∇(u) + u ⋅ ∇(du) + ∇(dp); #- ν*Δ(du)
Rc(u) = ∇ ⋅ u;
dRc(du) = ∇ ⋅ du;


var_eq((u, p), (v, q)) = ∫((u ⋅ ∇(u)) ⋅ v)dΩ - ∫((∇ ⋅ v) * p)dΩ + ∫((q * (∇ ⋅ u)))dΩ + ν * ∫(∇(v) ⊙ ∇(u))dΩ - ∫(hf ⋅ v)dΩ

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

    τ₁ = h / (2 * u)
    return 1 / (1 / τ₁^r + 1 / τ₂^r )    
end

τb(u, h) = (u ⋅ u) * τ(u, h);
stab_eq((u, p), (v, q)) = ∫((τ ∘ (u, h) * (u ⋅ ∇(v) + ∇(q))) ⊙ Rm(u, p)    +   τb ∘ (u, h) * (∇ ⋅ v) ⊙ Rc(u) )dΩ;
res_eq((u, p), (v, q)) = var_eq((u, p), (v, q)) + stab_eq((u, p), (v, q));
dvar_eq( (u, p), (du, dp), (v, q)) = ∫(((du ⋅ ∇(u)) ⋅ v) + ((u ⋅ ∇(du)) ⋅ v) + (∇(dp) ⋅ v) +  (q * (∇ ⋅ du)))dΩ + ν * ∫(∇(v) ⊙ ∇(du))dΩ
	
dstab_eq((u, p), (du, dp), (v, q)) = ∫(((τ ∘ (u, h) * (u ⋅ ∇(v)' +  ∇(q))) ⊙ dRm((u, p), (du, dp), (v, q))) + ((τ ∘ (u, h) * (du ⋅ ∇(v)')) ⊙ Rm(u, p)) + (τb ∘ (u, h) * (∇ ⋅ v) ⊙ dRc(du)))dΩ
	
jac((u, p), (du, dp), (v, q)) = dvar_eq((u, p), (du, dp), (v, q)) + dstab_eq((u, p), (du, dp), (v, q))
D = 1
ReD = 20
ν = D/ReD

# const Re = 10.0
# conv(u,∇u) = Re*(∇u')⋅u
# dconv(du,∇du,u,∇u) = conv(u,∇du)+conv(du,∇u)

# a((u,p),(v,q)) = ∫( ∇(v)⊙∇(u) - (∇⋅v)*p + q*(∇⋅u) )dΩ
# c(u,v) = ∫( v⊙(conv∘(u,∇(u))) )dΩ
# dc(u,du,v) = ∫( v⊙(dconv∘(du,∇(du),u,∇(u))) )dΩ
# res((u,p),(v,q)) = a((u,p),(v,q)) + c(u,v)
# jac((u,p),(du,dp),(v,q)) = a((du,dp),(v,q)) + dc(u,du,v)

op = FEOperator(res_eq,jac,X,Y)


global uh, ph
options = "-snes_type newtonls -snes_linesearch_type basic  -snes_linesearch_damping 0.5 -snes_rtol 1.0e-10 -snes_atol 0.0 -snes_monitor -pc_type asm -sub_pc_type lu  -ksp_type gmres -ksp_gmres_restart 30  -snes_converged_reason -ksp_converged_reason -ksp_error_if_not_converged true "

GridapPETSc.with(args=split(options)) do
    solver = PETScNonlinearSolver()
    global uh, ph
    uh, ph = solve(solver,op)
end
writevtk(Ω,"primal_results",cellfields=["uh"=>uh,"ph"=>ph])

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

var_eq_adj((uadj, padj), (v, q)) = -1* ∫((uh ⋅ ∇(uadj)) ⋅ v)dΩ + -1* ∫((∇(uadj) ⋅ uh ) ⋅ v)dΩ - 1* ∫((∇(padj)) ⋅ v )dΩ + -1* ∫((q * (∇ ⋅ uadj)))dΩ + ν * ∫(∇(v) ⊙ ∇(uadj))dΩ
b_eq_adj((v,q)) = ∫(hf ⋅ v)dΩ 

op_adj = AffineFEOperator(var_eq_adj,b_eq_adj,X_adj,Y_adj)



global uh_adj,ph_adj

options_adj = "-pc_type jacobi -ksp_type gmres -ksp_gmres_restart 30  -snes_converged_reason -ksp_converged_reason -ksp_error_if_not_converged true"

GridapPETSc.with(args=split(options_adj)) do
    solver_adj = PETScLinearSolver()
    global uh_adj,ph_adj
    uh_adj, ph_adj = solve(solver_adj,op_adj)
end

writevtk(Ω,"results_adj",cellfields=["uh_adj"=>uh_adj, "ph_adj"=>ph_adj])


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
