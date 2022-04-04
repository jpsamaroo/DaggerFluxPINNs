using NeuralPDE, Flux, ModelingToolkit, GalacticOptim, Optim, DiffEqFlux
using Quadrature, Cuba, CUDA, QuasiMonteCarlo
import ModelingToolkit: Interval, infimum, supremum

using ChainRulesCore, Zygote

using DaggerFlux, Dagger, DaggerWebDash
using DataFrames

ml = Dagger.MultiEventLog()
ml[:core] = Dagger.Events.CoreMetrics()
ml[:id] = Dagger.Events.IDMetrics()
ml[:wsat] = Dagger.Events.WorkerSaturation()
ml[:loadavg] = Dagger.Events.CPULoadAverages()
ml[:bytes] = Dagger.Events.BytesAllocd()
ml[:mem] = Dagger.Events.MemoryFree()
ml[:esat] = Dagger.Events.EventSaturation()
ml[:psat] = Dagger.Events.ProcessorSaturation()
lw = Dagger.Events.LogWindow(5*10^9, :core)
df = DataFrame([key=>[] for key in keys(ml.consumers)]...)
ts = Dagger.Events.TableStorage(df)
push!(lw.creation_handlers, ts)
d3r = DaggerWebDash.D3Renderer(8080; seek_store=ts)
push!(lw.creation_handlers, d3r)
push!(lw.deletion_handlers, d3r)
push!(d3r, DaggerWebDash.GanttPlot(:core, :id, :timeline, :esat, :psat, "Overview"))
push!(d3r, DaggerWebDash.LinePlot(:core, :wsat, "Worker Saturation", "Running Tasks"))
push!(d3r, DaggerWebDash.LinePlot(:core, :loadavg, "CPU Load Average", "Average Running Threads"))
push!(d3r, DaggerWebDash.LinePlot(:core, :bytes, "Allocated Bytes", "Bytes"))
push!(d3r, DaggerWebDash.LinePlot(:core, :mem, "Available Memory", "% Free"))
ml.aggregators[:logwindow] = lw
ml.aggregators[:d3r] = d3r
Dagger.global_context().log_sink = ml

using LinearAlgebra
BLAS.set_num_threads(1)

@parameters t x y
@variables u(..)
Dxx = Differential(x)^2
Dyy = Differential(y)^2
Dt = Differential(t)
t_min = 0.0
t_max = 2.0
x_min = 0.0
x_max = 2.0
y_min = 0.0
y_max = 2.0

# 2D PDE
eq  = Dt(u(t,x,y)) ~ Dxx(u(t,x,y)) + Dyy(u(t,x,y))

analytic_sol_func(t,x,y) = exp(x+y)*cos(x+y+4t)
# Initial and boundary conditions
bcs = [u(t_min,x,y) ~ analytic_sol_func(t_min,x,y),
       u(t,x_min,y) ~ analytic_sol_func(t,x_min,y),
       u(t,x_max,y) ~ analytic_sol_func(t,x_max,y),
       u(t,x,y_min) ~ analytic_sol_func(t,x,y_min),
       u(t,x,y_max) ~ analytic_sol_func(t,x,y_max)]

# Space and time domains
domains = [t ∈ Interval(t_min,t_max),
           x ∈ Interval(x_min,x_max),
           y ∈ Interval(y_min,y_max)]

# Neural network

struct SumLayer{L<:Tuple}
    layers::L
end
#(par::SumLayer)(x) = reduce((x,y)-> x .+ y, map(l -> l(x), par.layers))
(par::SumLayer)(ip) =
    Dagger.delayed(xs -> reduce((x,y) -> x .+ y, xs))([DaggerFlux.daglayer(f, ip) for f in par.layers])
Zygote.@adjoint function (sl::SumLayer)(x)
    return sl(x), dy -> (nothing, dy)
end
DaggerFlux.daglayer(par::SumLayer, ip) =
    Dagger.delayed(xs -> reduce((x,y) -> x .+ y, xs))([DaggerFlux.daglayer(f, ip) for f in par.layers])

inner = 1000
chain = Chain(Dense(3,inner,Flux.σ),
              #SumLayer((
                Dense(inner,inner,Flux.σ),
                Dense(inner,inner,Flux.σ),
                Dense(inner,inner,Flux.σ),
              #)),
              Dense(inner,1))
chain = DaggerChain(chain)
DiffEqFlux.initial_params(f::DaggerChain) = Flux.destructure(f)[1]
#= TODO: Use FastChain
chain = FastChain(FastDense(3,inner,Flux.σ),
                  FastDense(inner,inner,Flux.σ),
                  FastDense(inner,inner,Flux.σ),
                  FastDense(inner,inner,Flux.σ),
                  FastDense(inner,1))
chain = DaggerChain(Chain(chain))

#initθ = CuArray(Float64.(DiffEqFlux.initial_params(chain)))
initθ = Float64.(DiffEqFlux.initial_params(chain))
=#

strategy = GridTraining(0.05)
discretization = PhysicsInformedNN(chain,
                                   strategy)#;
                                   #init_params = initθ)

@named pde_system = PDESystem(eq,bcs,domains,[t,x,y],[u(t, x, y)])
prob = discretize(pde_system,discretization)
symprob = symbolic_discretize(pde_system,discretization)

cb = function (p,l)
    println("Current loss is: $l")
    return false
end

res = GalacticOptim.solve(prob, ADAM(0.01); cb=cb, maxiters=2500)
