function estimate_distributions(obj, vals, Y_, los, likelihoodDistIn, usePiecewise)

if ~exist('usePiecewise', 'var') || isempty(usePiecewise)
    usePiecewise = true;
end

uy = unique(Y_);
assert(length(uy) == 2);
[negOutcome, posOutcome] = deal(uy(1), uy(2));

[obj.pdNeg, obj.pdPos] = deal(cell(1, size(obj.losBins, 1)));
for ilos = 1:size(obj.losBins, 1)
    % losinds = los > obj.losBins(ilos, 1) & los <= obj.losBins(ilos, 2);
    
    if ~exist('likelihoodDistIn', 'var') || isempty(likelihoodDistIn)
        likelihoodDistPos = 'best';
        likelihoodDistNeg = 'best';
    else
        likelihoodDistPos = likelihoodDistIn;
        likelihoodDistNeg = likelihoodDistIn;
    end
    
    if ~isempty(vals)
        % [obj.fPos, obj.xPos] = ksdensity(vals(Y_ == posOutcome & losinds));
        % [obj.fNeg, obj.xNeg] = ksdensity(vals(Y_ == negOutcome & losinds));
        [obj.fPos, obj.xPos] = ksdensity(vals(Y_ == posOutcome));
        [obj.fNeg, obj.xNeg] = ksdensity(vals(Y_ == negOutcome));
    end
    
    if ~isequal(likelihoodDistIn, 'gmm')
        if isequal(likelihoodDistIn, 'best')
            plotflag = false;
            
            if usePiecewise
                % [pd, likelihoodDist] = best_dist_piecewise(vals(losinds), Y_(losinds), [0.3 0.85], 'stable', plotflag);
                [pd, likelihoodDist] = best_dist_piecewise(vals, Y_, [0.3 0.85], 'stable', plotflag);
            else
                [pd, likelihoodDist] = best_dist(vals, Y_, plotflag);
            end            
            
            if iscell(pd)
                obj.pdNeg{ilos} = pd{1};
                obj.pdPos{ilos} = pd{2};                
            else
                obj.pdNeg{ilos} = pd(1);
                obj.pdPos{ilos} = pd(2);                
            end
            likelihoodDistNeg = likelihoodDist{1};
            likelihoodDistPos = likelihoodDist{2};
        else
            % obj.pdNeg{ilos} = fitdist(vals(Y_ == negOutcome & losinds), likelihoodDistNeg);
            % obj.pdPos{ilos} = fitdist(vals(Y_ == posOutcome & losinds), likelihoodDistPos);
            obj.pdNeg{ilos} = fitdist(vals(Y_ == negOutcome), likelihoodDistNeg);
            obj.pdPos{ilos} = fitdist(vals(Y_ == posOutcome), likelihoodDistPos);
        end
    end
    %     obj.likelihoodDistPos = likelihoodDistPos;
    %     obj.likelihoodDistNeg = likelihoodDistNeg;
    
    % TODO what if a different distribution is selected?
    switch likelihoodDistPos
        case 'gmm'
            breakflag = false;
            for ngp = 6:-1:3
                for jj = 1:300
                    try
                        obj.gmmPos = fitgmdist(zs(Y_ == posOutcome, inds), ngp, 'CovarianceType', 'diagonal');
                        breakflag = true;
                        break
                    catch
                        continue
                    end
                end
                if breakflag
                    break
                end
            end
            if jj == 300 && ngp == 3
                error('did not complete');
            end
    end
    
    switch likelihoodDistNeg
        case 'gmm'
            breakflag = false;
            for ngn = 6:-1:3
                for jj = 1:300
                    try
                        obj.gmmNeg = fitgmdist(zs(Y_ == negOutcome, inds), ngn, 'CovarianceType', 'diagonal');
                        breakflag = true;
                        break
                    catch
                        continue
                    end
                end
                if breakflag
                    break
                end
            end
            if jj == 300 && ngn == 3
                error('did not complete');
            end
    end
    
    % obj.priorPos(ilos) = nnz(Y_ == posOutcome & losinds) / length(Y_(losinds));
    obj.priorPos(ilos) = nnz(Y_ == posOutcome) / length(Y_);
    obj.priorNeg(ilos) = 1 - obj.priorPos(ilos);
end