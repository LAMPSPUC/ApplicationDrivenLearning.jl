# using ApplicationDrivenLearning
using JuMP
using Test

@testset "Custom Variable Manipulations" begin
    # Create a simple model for testing
    model = ApplicationDrivenLearning.Model()
    
    @testset "Policy Variable Operations" begin
        # Create Policy variables
        @variables(model, begin
            p1, ApplicationDrivenLearning.Policy
            p2, ApplicationDrivenLearning.Policy
        end)
        
        # Test addition of Policy variables
        p_sum = p1 + p2
        @test p_sum isa ApplicationDrivenLearning.Policy{<:JuMP.GenericAffExpr}
        @test p_sum.plan == p1.plan + p2.plan
        @test p_sum.assess == p1.assess + p2.assess
        
        # Test multiplication by constant
        c = 3.5
        p_scaled = c * p1
        @test p_scaled isa ApplicationDrivenLearning.Policy{<:JuMP.GenericAffExpr}
        @test p_scaled.plan == c * p1.plan
        @test p_scaled.assess == c * p1.assess
        
        # Test commutative multiplication
        p_scaled2 = p1 * c
        @test p_scaled2.plan == c * p1.plan
        @test p_scaled2.assess == c * p1.assess
    end
    
    @testset "Forecast Variable Operations" begin
        # Create Forecast variables
        @variables(model, begin
            f1, ApplicationDrivenLearning.Forecast
            f2, ApplicationDrivenLearning.Forecast
        end)
        
        # Test addition of Forecast variables
        f_sum = f1 + f2
        @test f_sum isa ApplicationDrivenLearning.Forecast{<:JuMP.GenericAffExpr}
        @test f_sum.plan == f1.plan + f2.plan
        @test f_sum.assess == f1.assess + f2.assess
        
        # Test multiplication by constant
        c = 2.7
        f_scaled = c * f1
        @test f_scaled isa ApplicationDrivenLearning.Forecast{<:JuMP.GenericAffExpr}
        @test f_scaled.plan == c * f1.plan
        @test f_scaled.assess == c * f1.assess
        
        # Test commutative multiplication
        f_scaled2 = f1 * c
        @test f_scaled2.plan == c * f1.plan
        @test f_scaled2.assess == c * f1.assess
    end
    
    @testset "Array Variable Access - Policy" begin
        # Create array of Policy variables
        @variables(model, begin
            x[1:3] >= 0, ApplicationDrivenLearning.Policy
        end)
        
        # Test accessing individual elements
        @test x[1] isa ApplicationDrivenLearning.Policy
        @test x[2] isa ApplicationDrivenLearning.Policy
        @test x[3] isa ApplicationDrivenLearning.Policy
        
        # Test accessing plan variables from array
        x_plan = x.plan
        @test length(x_plan) == 3
        @test x_plan[1] == x[1].plan
        @test x_plan[2] == x[2].plan
        @test x_plan[3] == x[3].plan
        
        # Test accessing assess variables from array
        x_assess = x.assess
        @test length(x_assess) == 3
        @test x_assess[1] == x[1].assess
        @test x_assess[2] == x[2].assess
        @test x_assess[3] == x[3].assess
        
        # Test indexing into plan array
        @test x.plan[1:2] isa Vector
        @test length(x.plan[1:2]) == 2
        @test x.plan[1:2] == [x[1].plan, x[2].plan]
        
        # Test indexing into assess array
        @test x.assess[2:3] isa Vector
        @test length(x.assess[2:3]) == 2
        @test x.assess[2:3] == [x[2].assess, x[3].assess]
    end
    
    @testset "Array Variable Access - Forecast" begin
        # Create array of Forecast variables
        @variables(model, begin
            d[1:4], ApplicationDrivenLearning.Forecast
        end)
        
        # Test accessing individual elements
        @test d[1] isa ApplicationDrivenLearning.Forecast
        @test d[2] isa ApplicationDrivenLearning.Forecast
        @test d[4] isa ApplicationDrivenLearning.Forecast
        
        # Test accessing plan variables from array
        d_plan = d.plan
        @test length(d_plan) == 4
        @test d_plan[1] == d[1].plan
        @test d_plan[4] == d[4].plan
        
        # Test accessing assess variables from array
        d_assess = d.assess
        @test length(d_assess) == 4
        @test d_assess[1] == d[1].assess
        @test d_assess[4] == d[4].assess
        
        # Test indexing into plan array
        @test d.plan[1:3] isa Vector
        @test length(d.plan[1:3]) == 3
        @test d.plan[1:3] == [d[1].plan, d[2].plan, d[3].plan]
        
        # Test indexing into assess array
        @test d.assess[2:4] isa Vector
        @test length(d.assess[2:4]) == 3
        @test d.assess[2:4] == [d[2].assess, d[3].assess, d[4].assess]
    end
    
    @testset "Mixed Operations" begin
        # Create variables for mixed operations
        @variables(model, begin
            p[1:2] >= 0, ApplicationDrivenLearning.Policy
            f[1:2], ApplicationDrivenLearning.Forecast
        end)
        
        # Test summing array elements
        p_total = p[1] + p[2]
        @test p_total isa ApplicationDrivenLearning.Policy
        @test p_total.plan == p[1].plan + p[2].plan
        
        # Test scaling array elements
        scaled_p = 5.0 * p[1]
        @test scaled_p.plan == 5.0 * p[1].plan
        
        # Test using array properties in expressions
        plan_sum = sum(p.plan)
        assess_sum = sum(p.assess)
        @test plan_sum isa JuMP.GenericAffExpr
        @test assess_sum isa JuMP.GenericAffExpr
        
        # Test accessing specific indices from array properties
        @test p.plan[1] isa JuMP.VariableRef
        @test p.assess[2] isa JuMP.VariableRef
        @test f.plan[1] isa JuMP.VariableRef
        @test f.assess[2] isa JuMP.VariableRef
    end
    
    @testset "Preserve JuMP Container Properties" begin
        # Create array variables
        @variables(model, begin
            z[1:3] >= 0, ApplicationDrivenLearning.Policy
        end)
        
        # Test that original JuMP container properties still work
        # (if the container has these properties)
        @test length(z) == 3
        @test size(z) == (3,)
        
        # Test that we can still iterate
        count = 0
        for var in z
            @test var isa ApplicationDrivenLearning.Policy
            count += 1
        end
        @test count == 3
    end
end
