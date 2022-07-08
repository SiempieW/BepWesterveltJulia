using LinearAlgebra, SparseArrays, DifferentialEquations, Plots, Printf

#3d linear wave equation, non-hom initial conditions, Dirichlet BC's

function A_Gen(M)
    h = L/M
    h2 = h^2
    A = sparse(1/h2*Tridiagonal(1*ones(M),-2*ones(M+1),1*ones(M)))
    A[1,1] = 1;         A[1,2] = 0;
    A[end,end-1]=0;     A[end,end] = 1;

    I = sparse(Diagonal(ones(M+1)));
    A_h = spzeros((M+1)^3,(M+1)^3);
    A_h[1:end,1:end] += kron(A,I,I)+kron(I,A,I)+kron(I,I,A)
    bnd,interior = partnodes(M)
    A_h[bnd,:] .= 0
    println("A made")
    return A_h
end;

function partnodes(M)
    nnodes = (M+1)^3;
    indic = ones(M+1,M+1,M+1);
    indic[2:end-1,2:end-1,2:end-1] .= 0; 
    indvec = reshape(indic,(nnodes,1));
    bnd = vec([indvec .== 1][1]);
    interior = vec([indvec .== 0][1]);
    return bnd,interior
end;

function wave3D!(du,u,p,t)
    cA,uend,temp,nonlin = p
    a = @view u[uend+1:end] #use creats a pointer copy of the specified sub-array
    b = @view u[1:uend]
    c = @view du[1:uend]
    mul!(temp,cA,a)#mul! calculates the matrix vector product of cA*a and puts the solution in temp
    @. c = temp + nonlin*(b*b + a*c)
    du[uend+1:end] .= b
end

function initial_u(M)
    function g(x)
        if L/3 <= x <= 2*L/3 
            return sin(3*pi*x/L)^2
        end
        return 0
    end
    as = LinRange(0,L,Np1)
    #u0 = [sin(3*-x)^2*sin(y)^2*sin(z)^2 for x in as, y in as, z in as]
    u0 = [g(x)*g(y)*g(z) for x in as, y in as, z in as]
    return u0
end;
function f_jac(J,u,p,t)
    J = bigA
end;

if true
    N = 40;
    Np1 = N+1;
    tend = 0.002;tspan = (0.0,tend);

    L = 1.5
    h = L/N
    c = 1500

    β = 40
    p_max = 3*10^6
    ρ₀ = 1000
 
    cA = c^2 .* A_Gen(N);

    u0 = zeros(Np1,Np1,2*Np1);
    u0[:,:,Np1+1:end] = p_max*initial_u(Np1);
    fu0 = reshape(u0,(2*Np1^3));
end;

tempalloc = zeros(Np1^3);#temporary allocation variable to use in mul!
p = (cA,Np1^3,tempalloc,β/(c^2*ρ₀));
prob = ODEProblem(wave3D!,fu0,tspan,p);#define ode problem
@time sol = solve(prob,abstol=1e-6,reltol=1e-6);

function solhan(sol) #turns sol into an 4D-array time,x,y,z
    trange = LinRange(tspan[1],tspan[2],60)
    Nt = size(trange)[1]
    U = zeros(Nt,Np1,Np1,Np1)
    for i in 1:Nt
        U[i,:,:,:] = reshape(sol(trange[i])[Np1^3+1:end],(Np1,Np1,Np1))
    end
    return U
end
function anim3D(sol)
    trange = LinRange(tspan[1],tspan[2],30)
    as = LinRange(0,L,Np1)
    Nt = size(trange)[1]
    U = solhan(sol)
    anim = @animate for i in 1:Nt
        surface(U[i,:,:,:],legend=true,title="t: $(@sprintf "%.3E" trange[i])",
        camera=[25,42],showaxis=false,ticks=false)
    end
    gif(anim,"default_plot.gif",fps=3)
end
function slice3D(sol,slice)
    trange = LinRange(tspan[1],tspan[2],30)
    as = LinRange(0,L,Np1)
    Nt = size(trange)[1]
    U = solhan(sol)
    ma = maximum(U)
    anim = @animate for i in 1:Nt
        surface(as,as,U[i,slice[1],slice[2],slice[3]],zlims=0.5*ma.*[-1,1],clims=0.5*ma.*(-1,1),legend=true,
        title="t: $(@sprintf "%.3E" trange[i])",xlabel="x (m)",ylabel="y (m)",zlabel="p (Pa))")
    end
    gif(anim,"default_plot.gif",fps=3)
end
function line3D(sol,slice)
    trange = LinRange(tspan[1],tspan[2],30)
    as = LinRange(0,L,Np1)
    Nt = size(trange)[1]
    U = solhan(sol)
    ma = maximum(U)
    anim = @animate for i in 1:Nt
        plot(as,U[i,slice[1],slice[2],slice[3]],ylims=ma.*[-1,1],legend=false,
        title="t: $(@sprintf "%.3E" trange[i])",xlabel="x (m)",ylabel="p (Pa)")
    end
    gif(anim,"default_plot.gif",fps=3)
end

anim3D(sol)
slice = [:,:,Np1÷2];
slice3D(sol,slice,"Slice $N")
slice = [:,Np1÷2,Np1÷2];
line3D(sol,slice,"Slice $N")
