function [pd, likelihoodDist] = best_dist_piecewise(vals, Y, qs, centerDist, plotflag)

if ~exist('plotflag', 'var') || isempty(plotflag)
    plotflag = false;
end
if ~exist('qs', 'var') || isempty(qs)
    qs = [0.1 0.9];
end

% distnames = {'beta', 'binomial', 'birnbaumsaunders', 'burr', 'exponential', 'extreme value', ...
%     'gamma', 'generalized extreme value', 'generalized pareto', 'half normal', ...
%     'inversegaussian', 'logistic', 'loglogistic', 'lognormal', 'nakagami', 'negative binomial', ...
%     'normal', 'poisson', 'rayleigh', 'rician', 'stable', 'tlocationscale', 'weibull'};
% distfcns = {@evfitc, @expfitc, @gamfitc, @logistfitc, @loglogistfitc, @lognfitc, @normfitc, @raylfitc, @wblfitc};

% univariate distributions

% And I know how long since the last measurement

clear('d1', 'd2');

uy = unique(Y);
xi = cell(1, length(uy)); 
for iy = 1:length(uy)
    [f_, xi_] = ksdensity(vals(Y == uy(iy)));
    xi{iy} = interp1(xi_, f_, vals, 'linear', 'extrap');
end

max1 = -Inf;
max2 = -Inf;
% for jj = 1:length(distfcns)
       
        % [xx1, xx2, xx3] = deal([vals vals]);
        % xx1(vals > qvals(2), 2) = Inf;
        % xx2(vals > qvals(2), 2) = Inf;
        % xx2(vals < qvals(1), 1) = -Inf;
        % xx3(vals < qvals(1), 1) = -Inf;
        % [pars1, covars1, SE1, gval1, exitflag1] = distfcns{jj}(xx1(Y==uy(1), :), 2, [0 0]);
        % [pars2, covars2, SE2, gval2, exitflag2] = distfcns{jj}(xx2(Y==uy(1), :), 1);
        % [pars3, covars3, SE3, gval3, exitflag3] = distfcns{jj}(xx3(Y==uy(1), :), 1);
        % [pars1b, covars1b, SE1b, gval1b, exitflag1b] = distfcns{jj}(xx1(Y==uy(2), :), 1);
        % [pars2b, covars2b, SE2b, gval2b, exitflag2b] = distfcns{jj}(xx2(Y==uy(2), :), 1);
        % [pars3b, covars3b, SE3b, gval3b, exitflag3b] = distfcns{jj}(xx3(Y==uy(2), :), 1);
        
        % I can't get the code I downloaded to work, try piecewise
        % distribution with pareto tails instead
        % https://www.mathworks.com/help/stats/paretotails.html
        
        success = false;
        failct = 0;
        uvy = unique(vals(Y==uy(1), :));
        uvy2 = uvy;
        while ~success && failct < 20
            try
                temp{1} = paretotails(uvy2, qs(1), qs(2), @(x) myfun2(x, centerDist));
                success = true;
            catch ex
                failct = failct + 1;
                uvy2 = randsample(uvy, length(uvy), true);
            end
        end
        
        success = false;
        failct = 0;
        uvy = unique(vals(Y==uy(2), :));
        uvy2 = uvy;
        while ~success && failct < 20
            try
                temp{2} = paretotails(uvy2, qs(1), qs(2), @(x) myfun2(x, centerDist));
                success = true;
            catch ex
                failct = failct + 1;
                uvy2 = randsample(uvy, length(uvy), true);
            end
        end
        
        
        % fitdist only does right censoring
        % temp = fitdist(vals(qinds), distnames{jj}, 'By', Y(qinds), 'Censoring', vals > qvals(1));
        % temp = fitdist(vals(qinds), distnames{jj}, 'By', Y(qinds), 'Censoring', vals < qvals(1) & );
        % temp = fitdist(vals(qinds), distnames{jj}, 'By', Y(qinds), 'Censoring', vals > qvals(1));

        % temp = fitdist(vals, distnames{jj}, 'By', Y);
  
        % maximum likelihood
        temp1 = temp{1}.pdf(vals);
        temp2 = temp{2}.pdf(vals);
        inds = temp1 > 0 & temp2 > 0;
        ll1 = sum(log10(temp1(inds)), 'omitnan');
        ll2 = sum(log10(temp2(inds)), 'omitnan');
        
        % least squares on pdf
        % ll1 = -norm(temp{1}.pdf(vals) - xi{1}, 2);
        % ll2 = -norm(temp{2}.pdf(vals) - xi{2}, 2);
        
        if ll1 > max1
            max1 = ll1;
            d1 = temp{1};
%             ld1 = distnames(jj);
        end
        if ll2 > max2
            max2 = ll2;
            d2 = temp{2};
%             ld2 = distnames(jj);
        end

% end

if ~exist('d1', 'var')
    disp('');
end
pd = {d1 d2};
% likelihoodDist = [ld1 ld2];
likelihoodDist = {class(d1) class(d2)};

prior = zeros(2, 1);
if plotflag
    prior(1) = nnz(Y==uy(1)) / length(Y);
    prior(2) = 1 - prior(1);
    
    figure(); ax = gca(); hold(ax, 'on');
    uy = unique(Y);
    for iy = 1:length(uy)
        [f, xi] = ksdensity(vals(Y == uy(iy)));
        plot(ax, xi, f * prior(iy));
    end
    set(ax, 'ColorOrderIndex', 1);
    
    
    plot(ax, vals, pd{1}.pdf(vals) * prior(1), '.');
    plot(ax, vals, pd{2}.pdf(vals) * prior(2), '.');
    
%     title(ax, sprintf('%s, %s', ld1{1}, ld2{1}));
    
    hold(ax, 'off');
    legend({'Died' 'Survived'}, 'Location', 'northwest');
end

function [p,xi] = myfun2(x, centerDist)
pd = fitdist(x, centerDist);
xi = linspace(min(x),max(x),length(x)*2);
p = cdf(pd,xi);

