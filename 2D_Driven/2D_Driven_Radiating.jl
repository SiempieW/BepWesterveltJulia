using LinearAlgebra, SparseArrays,DifferentialEquations,Plots,BenchmarkTools,Printf

#2d wave equation with non-linearity, driven side, radiating BC's

function A_Gen(M)
    h = L/M
    h2 = h^2
    A = sparse(1/h2*Tridiagonal(1*ones(M),-2*ones(M+1),1*ones(M)))
    A[1,1] = 1;         A[1,2] = 0;
    A[end,end-1]=0;     A[end,end] = 1;
    
    I = sparse(Diagonal(ones(M+1)));
    A_h = spzeros((M+1)^2,(M+1)^2);
    A_h[1:end,1:end] += kron(A,I) + kron(I,A)
    bnd,interior = partnodes(M)
    A_h[bnd,:] .= 0
    println("A made")
    return A_h
end;
function partnodes(M)
    nnodes = (M+1)^2;
    indic = ones(M+1,M+1);
    indic[2:end-1,2:end-1] .= 0; 
    indvec = reshape(indic,(nnodes,1));
    bnd = vec([indvec .== 1][1]);
    interior = vec([indvec .== 0][1]);
    return bnd,interior
end;
function unflat(u)
    return reshape(u,(Np1,Np1))
end;
function flat(u,M)
    return reshape(u,M)
end;

function driving!(du,u,t,c,Np1,uend)
    a = 10*10^3
    if t >= pi/a
        du[uend+1:uend+Np1] .= 0
        du[1:Np1] .= 0
        return nothing
    end
    for i in 1:Np1
        x = (i-1)*pi/(Np1-1)
        du[uend+i] = p_max*3*a*cos(a*t)*sin(a*t)^2*sin(x)^2
    end
    return nothing
end
function wave2D_first_order_system!(du,u,p,t)
    Np1::Int64,uend::Int64,temp,A,c0,cend,h,coeff1,coeff2 = p
    mul!(temp,A,u[uend+1:end])
    for i in Np1+1:uend÷2
        du[i] = temp[i]+coeff1*(u[i]*u[i]+u[uend+i]*du[i])
    end
    for i in uend÷2+1:uend
        du[i] = temp[i]+coeff2*(u[i]*u[i]+u[uend+i]*du[i])
    end
    du[uend+1:end] .= u[1:uend] #order raising
    @. du[uend-Np1+1:uend] = -1*cend/h*(u[uend-Np1+1:uend]-u[uend-2*Np1+1:uend-Np1])#x=L boundary condition
    driving!(du,u,t,c0,Np1,uend)#driving boundary condition at x = 0
    for i in Np1+1:Np1+Np1 #radiating boundary conditions at y=0,L
        du[i*Np1] = -1*c0/h*(u[i*Np1]-u[i*Np1-1])
        du[(i-1)*Np1+1] = -1*c0/h*(u[(i-1)*Np1+1]-u[(i-1)*Np1+2])
    end
    nothing
end
#@code_warntype wave2D_first_order_system!(zeros(2*Np1^2),fu0,p,0)
function initial_u(M)
    u = zeros(M,M)
    xas = LinRange(0,pi,M)
    yas = LinRange(0,pi,M)

    u = [sin(2*x)^3*sin(y)^2 for x in xas,y in yas]
    u[1,:] .= 0
    u[end,:] .= 0
    u[:,1] .= 0
    u[:,end] .= 0
    return u
end;

if true
    N = 50;
    Np1 = N+1;
    tend = 0.002;tspan = (0.0,tend);

    L = 1.5
    h = L/N
    c = 1500

    β = 40
    p_max = 3*10^6
    ρ₀ = 1000
    u0 = zeros(Np1,2*Np1);
    u0[:,Np1+1:end] = 0*initial_u(Np1);

    A = A_Gen(N);
    dropzeros!(A)
    fu0 = flat(u0,2*Np1*Np1);

    cend = 0.5c
    cmat = spzeros(Np1,Np1)
    cmat[1:end,Np1÷2:end] .= cend^2-c^2
    cvec = c^2*ones(Np1^2)
    cvec += reshape(cmat,Np1^2)
    
end;

tempalloc = zeros(Np1^2);#temporary allocation variable to use in mul!
p = (Np1,Np1^2,tempalloc,cvec.*A,c^2,cend^2,h,β/(c^2 * ρ₀),β/(cend^2 * ρ₀));
prob = ODEProblem(wave2D_first_order_system!,fu0,tspan,p);
@time sol = solve(prob,Tsit5(),reltol=1e-6,abstol=1e-6)

function solhan(sol) #turns sol into an 3D-array time,x,y
    trange = LinRange(tspan[1],tspan[2],60)
    Nt = size(trange)[1]
    U = zeros(Nt,Np1,Np1)
    for i in 1:Nt
        U[i,:,:] = unflat(sol(trange[i])[Np1^2+1:end])
    end
    return U
end;
anim2D(sol) = anim2D(sol,"");
function anim2D(sol,title)
    nth = 2::Int64
    xas = LinRange(0,L,Np1)
    trange = LinRange(tspan[1],tspan[2],60)
    Nt = size(trange)[1]
    U = solhan(sol)
    anim = @animate for i in 1:floor(Int64,Nt/nth)
        surface(xas,xas,U[nth*i,:,:],legend=false,title="t: $(@sprintf "%.2E" trange[nth*i])",
        zlims=p_max*[-0.5,1],camera=[20,30],xlabel="x (m)",ylabel="y (m)",zlabel="p (Pa)")
    end
    gif(anim,"default_plot.gif",fps=3)
end;
anim2D(sol,"2D driven")
function animSlice(sol,slice,title)
    nth = 2::Int64
    trange = LinRange(tspan[1],tspan[2],60)
    xas = LinRange(0,L,Np1)
    Nt = size(trange)[1]
    U = solhan(sol)
    anim = @animate for i in 1:floor(Int64,Nt/nth)
        plot(xas,U[nth*i,slice[1],slice[2]],legend=false,title="t: $(@sprintf "%.3E" trange[nth*i])",
        ylims=p_max*[-1,1],xlabel="x (m)",ylabel="p (Pa)",label="Computed solution")
        plot!(xas,p_max*2/3*ones(Np1),label="2/3*p_max")
    end
    gif(anim,"default_plot.gif",fps=3)
end;

anim2D(sol,"2D driven")
animSlice(sol,(Np1÷2,1:Np1),"2D driven")
