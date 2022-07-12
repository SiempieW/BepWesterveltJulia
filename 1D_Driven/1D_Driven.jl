using LinearAlgebra, SparseArrays,DifferentialEquations,Printf,Plots
using BenchmarkTools
#1D wave equation with non-linearity. Driven LHS,Radiating/Open rhs.

function ddtdriving(t)
    if t > tdriven
        return 0
    end
    return p_max*2*f*cos(f*t)*sin(f*t)
end
function wave!(du,u,p,t)
    Np1,temp,A,nonlin_coeff,c2 = p
    mul!(temp,A,u[Np1+1:end])
    du[1:Np1] .= temp + nonlin_coeff*(2*u[1:Np1].*u[1:Np1] + 2*u[Np1+1:end].*du[1:Np1])
    du[Np1+1:end] .= u[1:Np1]

    du[Np1] = -c2/h*(u[Np1]-u[Np1-1])    #Radiating RHS
    du[Np1+1] = ddtdriving(t)           #Driven LHS
end

function initial_u(M)
    u = zeros(M)
    for i in 1:M÷2
        u[i] = sin(2*i/M*pi)^2
    end
    u[1] = 0
    u[end] = 0
    return u
end

function A_gen(N)
    squigA = sparse(1/h2*Tridiagonal(1*ones(N),-2*ones(N+1),1*ones(N)))
    A = squigA
    A[1,1] = 1;         A[1,2] = 0;
    A[end,end-1] = 0;     A[end,end] = 1;
    return A
end
if true
    N = 500;
    Np1 = N+1;

    tend = 0.001;tspan = (0.0,tend);
    tdriven = 0.0002;f=pi/tdriven

    L = 1.5
    h = L/N;h2=h^2
    c = 1500
    β = 40
    p_max = 3*10^6
    ρ₀ = 1000

    u0 = zeros(2*Np1);
    A = c^2 * A_gen(N);
end
tempalloc = zeros(Np1);
p = (Np1,tempalloc,A,0/c^2*β/ρ₀,c^2);
prob = ODEProblem(wave!,u0,tspan,p);
@time sol = solve(prob,Tsit5(),reltol=1e-6,abstol=1e-6);

p = (Np1,tempalloc,A,1/c^2*β/ρ₀);
nonlinprob = ODEProblem(wave_first_order_system!,u0,tspan,p);
@time nonlinsol = solve(nonlinprob,Tsit5(),reltol=1e-6,abstol=1e-6);

if true #Make animation of linear solution along with non-linear solution
    trange = LinRange(tspan[1],tspan[2],60)
    anim = @animate for i in trange
        #plot(sol(i)[1:Np1],labels="diff",title="time: $(i)")#,ylims=[-1,1])
        plot(LinRange(0,L,Np1),sol(i)[Np1+1:end],title="time: $(@sprintf "%.2E" i)",
        ylims = p_max .* [0,1],xlabel="x (m)",ylabel="p (Pa)",label="Linear solution",legend=:outertop)
        plot!(LinRange(0,L,Np1),nonlinsol(i)[Np1+1:end],label="Non-Linear solution")
        plot!(LinRange(0,L,Np1),p_max*ones(Np1),label="p_max")
        #plot!(Shape([DensityLim,DensityLim,Np1,Np1], [-1,2,2,-1]),alpha=0.35)
    end
    gif(anim,"default_plot.gif",fps=3)
end
