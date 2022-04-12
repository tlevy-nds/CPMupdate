classdef Performance < handle
    % Class to compute performance metrics from classification model outputs and ground truth
    
    properties
        x          % linear predictor
        y          % outcome
        P          % probability
        cvguidimp  % keep track of patient identifiers
        tpr        % true positive rate
        fpr        % false positive rate
        thresh     % decision threshold (based on linear predictor)
        Pthresh    % decision threshold (based on probability)
        auc        % area under the ROC curve
        ci         % confidence interval
        prauc      % area under the PR curve
        precision  % precision
        recall     % recall
        opPts      % operating points
        
        axroc      % handle to ROC curve axes
        axpr       % handle to PR curve axes
        
        markerInd  % index for interactively placing markers at the oprerating points
    end
    
    properties (Constant)
        markers = '.pd';
        markerSizes = [24 12 10];
    end
    
    methods
        function obj = Performance(x, y, P, useP, cvguidimp)
            % constructor
            if nargin == 0
                return
            end
            
            if ~exist('useP', 'var') || isempty(useP) || ~exist('P', 'var') || isempty(P)
                useP = false;
            end
            if ~exist('cvguidimp', 'var')
                cvguidimp = [];
            end
            obj.init(x, y, P, useP, cvguidimp)
        end
        
        function init(obj, x, y, P, useP, cvguidimp)
            assert(isvector(x), 'x must be a vector');
            assert(isvector(y), 'y must be a vector');
            if ~exist('useP', 'var') || isempty(useP) || ~exist('P', 'var') || isempty(P)
                useP = false;
            end

            if ~exist('cvguidimp', 'var')
                cvguidimp = [];
            end

            obj.cvguidimp = cvguidimp;
            
            uy = unique(y, 'sorted');
            if length(uy) == 1
                obj.auc = NaN;
                fprintf(1, 'only 1 unique value\n');
                return
            end
            assert(length(uy) == 2, 'y must have 2 unique values');
            if mean(x(y == uy(1))) < mean(x(y == uy(2)))            
                obj.x = reshape(x, 1, []);
            else
                obj.x = -reshape(x, 1, []);
            end
            obj.y = reshape(y, 1, []);
            
            if useP
                rocVar = reshape(P, 1, []);
            else
                rocVar = obj.x;
            end
            
            if exist('P', 'var')
                obj.P = reshape(P, 1, []);
                % temp = sortrows([obj.x' obj.P'], 1, 'descend');
                % obj.Pthresh = interp1(obj.x', obj.P', obj.thresh, 'linear', 'extrap');
                
                [~, ia] = unique(rocVar', 'sorted');  % TODO unique P is not the correct dimension, testing ROC curves based on P
                
                obj.Pthresh = flip(obj.P(ia)');
                obj.Pthresh = obj.Pthresh([1 1:end]);
                % obj.Pthresh = temp([1 1:end], 2);
                
%                 if isequal(x(~isnan(x) & ~isnan(P)), P(~isnan(x) & ~isnan(P)))
%                     obj.x = obj.x(ia);
%                     obj.y = obj.y(ia);
%                     rocVar = rocVar(ia);
%                 end
            end            
            
            [obj.fpr, obj.tpr, obj.thresh, obj.auc] = perfcurve(obj.y, rocVar, uy(2), 'XCrit', 'fpr', 'YCrit', 'tpr');
            obj.ci = roc_ci(obj.auc, nnz(y==uy(1)), nnz(y==uy(2)), 0.05);
            
            [obj.recall, obj.precision, ~, obj.prauc] = perfcurve(obj.y, rocVar, uy(2), 'XCrit', 'reca', 'YCrit', 'prec');
            
            obj.opPts = obj.operatingPoints();
            obj.markerInd = 1;
        end
        
        p = deLong(obj, roc)
        
        [cm, fpr_, tpr_] = operatingPoints(obj)
        
        [ax, pl] = plot_roc(obj, ax)
        [ax, pl] = plot_pr(obj, ax)
        
        [r2, m, b, Eavg, E90, citl, brierScore, EavgCI] = p_vs_p(obj, mystride, useBarPlot, valset, horizonDays, nImputations, plotflag)
        
        nb = net_benefit(obj, hFig, dispName, myCol, plotAllNone)
    end
    
    methods (Access=private)
        window_button_down_fcn(obj, src, evt)
    end
end