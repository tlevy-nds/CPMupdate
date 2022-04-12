classdef LassoGlmBC < BinaryClassifier
    properties
        lambdahat
    end
    
    methods
        function obj = LassoGlmBC()
            obj = obj@BinaryClassifier();
            
            obj.mdl = struct('thetahat', {{}}, 'Sigmahat', {{}}, 'R', {{}}, 'tuning', [], 'mean', {{}}, 'std', {{}});
        end
        
        dynamicLR(obj, X, y);   
    end
    
    methods (Static, Access=private)
        [coefs, idx] = select_lambda(B, FitInfo, nfeatures)
        [tuning] = tuning_fcn(lambda_, x, y, theta_, Sigma_)
    end
end
