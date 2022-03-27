using PackageCompiler

pkgs = ["NeuralPDE", "Flux", "ModelingToolkit", "GalacticOptim", "Optim", "DiffEqFlux",
        "Quadrature", "Cuba", "CUDA", "QuasiMonteCarlo"]
create_sysimage(pkgs; sysimage_path="PINNSysimage.so",
                      precompile_execution_file="run-slim.jl")
