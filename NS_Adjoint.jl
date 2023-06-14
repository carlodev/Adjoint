using Gridap
using GridapGmsh
using GridapPETSc
using LineSearches: BackTracking

model = GmshDiscreteModel("Cylinder2D.msh")


writevtk(model,"model")

D = 2
order = 1
reffeᵤ = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
V = TestFESpace(model,reffeᵤ,conformity=:H1,dirichlet_tags= ["cylinder", "inlet", "limits"])

reffeₚ = ReferenceFE(lagrangian,Float64,order)
Q = TestFESpace(model,reffeₚ,conformity=:H1, dirichlet_tags= ["outlet"])

u_wall = VectorValue(0,0)
u_inlet = VectorValue(1,0)
hf = VectorValue(0,0)

U = TrialFESpace(V,[u_wall,u_inlet,u_inlet])
P = TrialFESpace(Q, 0.0)

Y = MultiFieldFESpace([V, Q])
X = MultiFieldFESpace([U, P])

degree = order*2
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
ν = 0.01

# const Re = 10.0
# conv(u,∇u) = Re*(∇u')⋅u
# dconv(du,∇du,u,∇u) = conv(u,∇du)+conv(du,∇u)

# a((u,p),(v,q)) = ∫( ∇(v)⊙∇(u) - (∇⋅v)*p + q*(∇⋅u) )dΩ
# c(u,v) = ∫( v⊙(conv∘(u,∇(u))) )dΩ
# dc(u,du,v) = ∫( v⊙(dconv∘(du,∇(du),u,∇(u))) )dΩ
# res((u,p),(v,q)) = a((u,p),(v,q)) + c(u,v)
# jac((u,p),(du,dp),(v,q)) = a((du,dp),(v,q)) + dc(u,du,v)

op = FEOperator(res_eq,jac,X,Y)

nls = NLSolver(show_trace=true, method=:newton,  linesearch=BackTracking())
solver = FESolver(nls)

uh, ph = solve(solver,op)
writevtk(Ω,"ins-results",cellfields=["uh"=>uh,"ph"=>ph])

Γ = BoundaryTriangulation(model; tags=["cylinder"]) 
dΓ = Measure(Γ,degree)
n_Γ = - get_normal_vector(Γ)



r((n1,n2)) = VectorValue(-n2,n1)
t_Γ = r∘n_Γ #extract tangent
ff_t = (transpose(∇(uh))⋅n_Γ) ⋅ t_Γ

bc_cylinder = - ∇(uh) ⋅ VectorValue(1, 0)
writevtk(Ω,"ins-results",cellfields=["uh"=>uh,"ph"=>ph, "bc_cylinder"=>bc_cylinder])
writevtk(Γ,"ins-results_cylinder",cellfields=["uh"=>uh,"ph"=>ph, "bc_cylinder"=>bc_cylinder])

#Adjoint


reffe_u_adj = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
V_adj = TestFESpace(model,reffeᵤ,conformity=:H1,dirichlet_tags= ["cylinder","inlet", "limits"])

reffe_p_adj = ReferenceFE(lagrangian,Float64,order)
Q_adj = TestFESpace(model,reffeₚ,conformity=:H1, dirichlet_tags= ["outlet"])

U_adj = TrialFESpace(V_adj,[bc_cylinder,u_inlet,u_inlet])
P_adj = TrialFESpace(Q_adj, 0.0)

Y_adj = MultiFieldFESpace([V_adj, Q_adj])
X_adj = MultiFieldFESpace([U_adj, P_adj])

var_eq_adj((uadj, padj), (v, q)) = ∫((uh ⋅ ∇(uadj)) ⋅ v)dΩ + ∫((uadj ⋅ ∇(uh)) ⋅ v)dΩ - ∫((∇ ⋅ v) * padj)dΩ + ∫((q * (∇ ⋅ uadj)))dΩ + ν * ∫(∇(v) ⊙ ∇(uadj))dΩ
b_eq_adj((v,q)) = ∫(hf ⋅ v)dΩ 

op_adj = AffineFEOperator(var_eq_adj,b_eq_adj,X_adj,Y_adj)

ls = LUSolver()
solver_adj = LinearFESolver(ls)

uh_adj,ph_adj = solve(solver_adj,op)
writevtk(Ω,"results_adj",cellfields=["uh_adj"=>uh_adj, "ph_adj"=>ph_adj])

