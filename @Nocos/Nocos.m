classdef Nocos < matlab.mixin.Copyable
    properties
        filename
        
        % hyperparameters
        lambda
        nfeatures
        
        % standardization coefficients
        mus
        sigs
        
        % regression coefficients
        lassoCoeffs
        biasTerm
        
        % logistic regression lasso
        B1              % biased coefficient vector
        unbiasedModel   % unbiased GeneralizedLinearModel object
        
        % Bayesian lasso
        B2
        
        % likelihood parameters
        likelihoodDistPos
        likelihoodDistNeg
        pdPos
        pdNeg
%         nuPos
%         nuNeg
%         kPos
%         kNeg
%         muPos
%         muNeg
%         stdPos
%         stdNeg
        gmmPos
        gmmNeg
        
        losBins
        
        % empirical likelihood parameters
        fPos
        xPos
        fNeg
        xNeg
        
        % priors
        priorPos
        priorNeg
        
        % selected predictors
        predictorNames
        
        trainingCvguids
        calCorrection
        
        % logistic recalibration
        coefs
        alphaNew
        betaOverall
    end
    
    methods(Access = protected)
      % Override copyElement method:
      function cpObj = copyElement(obj)
         % Make a shallow copy of all four properties
         cpObj = copyElement@matlab.mixin.Copyable(obj);
         % Make a deep copy of the DeepCp object
         % cpObj.pdPos{1} = copy(obj.pdPos{1});
         % cpObj.pdNeg{1} = copy(obj.pdNeg{1});
      end
   end
    
    methods
        function obj = Nocos(varargin)
            obj.losBins = [0 Inf];
            % obj.losBins = [0 7*24; 7*24 14*24; 14*24 Inf];
            
            obj.alphaNew = 0;
            obj.betaOverall = 1;
            
            if nargin == 0
                return
            elseif nargin == 1 && ischar(varargin{1})
                filename = varargin{1};
                obj.load(filename);
                return
            elseif nargin == 10 && isnumeric(varargin{3})
                [X, Y, los, predictorNames, imputeMethod, resamplingMethod, nfeatures, lambda, useCV, likelihoodDist] = deal(varargin{:});                
                obj.train(X, Y, los, predictorNames, imputeMethod, resamplingMethod, nfeatures, lambda, useCV, likelihoodDist, controlRandomNumberGeneration);
            elseif nargin == 8                
                [X, Y, imputeMethod, resamplingMethod, lambdaCriterion, predictorNames, numLambda, K] = deal(varargin{:});       
                obj.trainglm(X, Y, imputeMethod, resamplingMethod, lambdaCriterion, predictorNames, numLambda, K);
            elseif nargin == 9 && iscellstr(varargin{5})
                [X, Y, imputeMethod, resamplingMethod, predictorNames, lambda, alpha_, N, cv] = deal(varargin{:});
                obj.trainblm(X, Y, imputeMethod, resamplingMethod, predictorNames, lambda, alpha_, N, cv);
            end
        end
        
        [vals, keepInds, Bs] = train(obj, X, Y, los, predictorNames, imputeMethod, resamplingMethod, nfeatures, lambda, useCV, likelihoodDist, controlRandomNumberGeneration, coefficientAnalysis, usePiecewise)
        keepInds = trainglm(obj, X, Y, imputeMethod, resamplingMethod, lambdaCriterion, predictorNames, numLambda, K, plotflag, controlRandomNumberGeneration)
        keepInds = trainblm(obj, X, Y, imputeMethod, resamplingMethod, predictorNames, lambda, alpha_, N, cv, plotflag, controlRandomNumberGeneration)
        
        estimate_distributions(obj, vals, Y_, los, likelihoodDist, usePiecewise)
        
        [xs, Ps, rocStruct, keepinds] = predict(obj, X, Psurvive, Y, los, predictorNames, imputeMethod, plotflag, axroc)    
        [PsBiased, PsUnbiased, keepInds] = predict_lr(obj, X, Psurvive, Y, los, predictorNames, imputeMethod, plotflag, axroc)
        
        generate_examples(obj, X)
        
        load(obj, filename)
        
        save(obj, filename)
        
        zs = standardize(obj, X, setflag)
        
        generate_eval_function(obj, filename)
        
        function update_intercept(obj, p, y)
            % Method 2
            lp = log(p ./ (1 - p));            
            myFcn = @(x) mean((y - 1./(1 + exp(-(x + lp)))).^2);
            obj.alphaNew = fminsearch(myFcn, 0);
        end
        
        function logistic_recalibration(obj, p, y)
            % Method 3
            lp = log(p ./ (1 - p));
            calCoefs = glmfit(lp, y, 'binomial', 'Link', 'logit');
            obj.alphaNew = calCoefs(1);
            obj.betaOverall = calCoefs(2);
        end
        
        function selective_reestimation(obj, p, y, X)
            % Method 4
            % TODO not sure how to do this yet
        end
        
        function reestimation(obj, y, X, trainingParams)
            % Method 5
            % TODO predInds
            % predInds = obj.coefs(2:end) ~= 0;
            % myPredictors = obj.FitInfo.PredictorNames;
            
            assert(size(X, 2) == length(obj.predictorNames));
            obj.train(X, y, [], obj.predictorNames, 'mean', 'oversampleMinorityClass', obj.nfeatures, 0, 10, 'best');

            % reformat so X is the same size
            % temp = zeros(size(X, 2), 1);
            % temp(predInds) = obj.coefs(2:end);
            % obj.coefs = [obj.coefs(1); temp];
            % obj.FitInfo.PredictorNames = myPredictors;
        end
        
        function selective_reestimation_selective_extension(obj, p, y, X)
            % Method 6
        end
        
        function reestimation_selective_extension(obj, p, y, X)
            % Method 7
        end
        
        function reestimation_extension(obj, y, X, nfeatures, trainingParams)
            % Method 8
            % TODO
            obj.train(X, y, [], obj.predictorNames, 'mean', 'oversampleMinorityClass', nfeatures, [], 10, 'best');
        end
    end
    
    methods (Static)
        [fpr, tpr, thresh, auc, p] = roc(xs, Y)  % if xs is 2D I can compute p
        
        [zs, keepinds] = impute_data(zs, Y, imputeMethod)
        
        ind = choose_lambda(B, fitinfo, X, Y, nfeatures)
    end
end