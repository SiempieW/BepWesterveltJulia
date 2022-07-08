using DifferentialEquations,LinearAlgebra,SparseArrays,Plots,BenchmarkTools,Printf

function wave!(du,u,p,t)
    Np1::Int64,temp,A = p
    mul!(temp,A,u[Np1+1:end])
    du[1:Np1] .= temp# + 0.15*( 2*u[1:Np1].*u[1:Np1] + 2*u[Np1+1:end].*du[1:Np1])
    du[Np1+1:end] .= u[1:Np1]
end;
function initial_u(Np1)
    out = zeros(Np1)
    for i in 200:300
        out[i] = p_max*sin(pi/100*i)^6
    end
    out[1] = 0
    out[end] = 0
    return out
end;
function A_Gen(M)
    squigA = sparse(1/h2*Tridiagonal(1*ones(M-2),-2*ones(M-1),1*ones(M-2)))
    A = squigA
    A[1,1] = 1;         A[1,2] = 0;
    A[end,end-1] = 0;     A[end,end] = 1;
    return A
end;
function analytical(u0,t,interpalg,K::Int64)
    x6Np1 = LinRange(0,L,K*Np1)
    xs = LinRange(0,L,Np1)
    interp = interpalg(x6Np1,initial_u(K*Np1))
    out = zero(u0)
    for i in 1:Np1
        if 0<xs[i]-c*t<L
            if 0<xs[i]+c*t<L
                out[i] = 0.5*interp(xs[i]-c*t)+0.5*interp(xs[i]+c*t)
                continue
            end
            out[i] = 0.5*interp(xs[i]-c*t)
            continue
        end
        if 0<xs[i]+c*t<L
            out[i] = 0.5*interp(xs[i]+c*t)
        end
    end
    return out
end

if true
    N = 499;
    Np1 = N+1;

    tend = 0.0005;tspan = (0.0,tend);

    L = 1.5
    h = L/N;h2=h^2
    c = 1500
    β = 5
    p_max = 3*10^6
    ρ₀ = 1000

    A = c^2 .* A_gen(N+2);

    u0 = zeros(2*Np1);
    u0[Np1+1:end] = initial_u(Np1);
end;

tempalloc = zeros(Float64,Np1);
prob = ODEProblem(wave!,u0,tspan,(Np1,tempalloc,A));
@time sol = solve(prob,reltol=1e-6,abstol=1e-6);

function anim(sol)#Plot computed solution along with analytical solution
    trange = LinRange(tspan[1],tspan[2],60);
    xas = LinRange(0,L,Np1)
    Nt = size(trange)[1]
    anim = @animate for i in trange
        plot(xas,sol(i)[Np1+1:end],title=title*" Time: $(@sprintf "%.2E" i)",
        label="Computed Solution",ylabel = "p (Pa)",xlabel="x (m)",ylims=[0,p_max])
        plot!(xas,analytical(initial_u(Np1),i,LinearInterpolation,1),label="Analytical solution")
        plot!(xas,0.5*p_max*ones(Np1),label="0.5 p_max")
        #plot!(-0.5*p_max*ones(Np1),label="-0.5")
    end
    gif(anim,"default_plot.gif",fps=3)
end
function err_anim(sol)#Plot relative error of computed solution
    trange = LinRange(tspan[1],tspan[2],60);
    xas = LinRange(0,L,Np1)
    Nt = size(trange)[1]
    anim = @animate for t in trange
        plot(xas,(sol(t)[Np1+1:end]-analytical(initial_u(Np1),t))/p_max,title=title*" Time: $(@sprintf "%.2E" t)",
        label="Relative Error",xlabel="x (m)")
    end
    gif(anim,"default_plot.gif",fps=3)
end

anim(sol)
err_anim(sol)
