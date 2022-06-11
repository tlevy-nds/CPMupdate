function [r2, m, b, Eavg_, E90, citl, brierScore, EavgCI] = p_vs_p(obj, mystride, useBarPlot, valset, horizonDays, nImputations, plotflag)

if isempty(obj.P)
    fprintf(1, 'P not set\n');
    return
end

if ~exist('mystride', 'var') || isempty(mystride)
    mystride = .25;
end
if ~exist('useBarPlot', 'var') || isempty(useBarPlot)
    useBarPlot = true;
end
if ~exist('nImputations', 'var') || isempty(nImputations)
    nImputations = 1;
end
if ~exist('plotflag', 'var') || isempty(plotflag)
    plotflag = true;
end

inds1 = isnan(obj.P) & obj.x < 0;
inds2 = isnan(obj.P) & obj.x >= 0;
obj.P(inds1) = 0;
obj.P(inds2) = 1;

mybins = 0:mystride:1-mystride;
P = obj.P;

Y_2 = obj.y;
uy = unique(obj.y, 'sorted');
[negOutcome, posOutcome] = deal(uy(1), uy(2));

N = arrayfun(@(ii) length(Y_2(P > ii & P <= ii+mystride)), mybins);
P2 = arrayfun(@(ii) nnz(Y_2(P > ii & P <= ii+mystride) == posOutcome) / length(Y_2(P > ii & P <= ii+mystride)), mybins);

% https://www.researchgate.net/figure/Calibration-of-the-predictive-model-in-the-modeling-group-All-patients-were-grouped_fig3_294730975
% I want to compute R^2 for a weighted least squares
ctrs = mybins+mystride/2;

% https://en.wikipedia.org/wiki/Coefficient_of_determination
inds = ~isnan(P2);
ybar = sum(N(inds).*P2(inds))/sum(N(inds));
sstot = sum(N(inds).*(P2(inds) - ybar).^2);
ssres = sum(N(inds).*(P2(inds) - ctrs(inds)).^2);
r2 = 1 - ssres / sstot;

bw = 0.01;     % bin width
sc = 0.9;      % y-scale
sw = 0.05;    % sliding window

temp = sortrows([P' Y_2'], 1);

% Hosmer-Lemeshow https://en.wikipedia.org/wiki/Hosmer%E2%80%93Lemeshow_test
% G = 10;
% for igrp = 1:G
%     O1g = nnz(P > (igrp - 1) / G & P <= igrp / G);
%     E1g = (igrp / G - 0.5) ;
%     
% end

% temp2 = sgolayfilt((temp(:, 2)+1)/2, 3, round(length(P)*sw)+1);
if all(ismember(unique(temp(:, 2)), [-1 1]))
    temp2 = movmean((temp(:, 2)+1)/2, round(length(P)*sw), 'Endpoints', 'shrink');
    % temp2 = smooth((temp(:, 2)+1)/2, temp(:, 1), 'rlowess');
elseif all(ismember(unique(temp(:, 2)), [0 1]))
    temp2 = movmean(temp(:, 2), round(length(P)*sw), 'Endpoints', 'shrink');
    % temp2 = smooth(temp(:, 1), temp(:, 2), 'rlowess');
end

x = temp(:, 1);
y = temp2;
naninds = ~isnan(x) & ~isnan(y);
x = x(naninds);
y = y(naninds);

Eavg = mean(abs(y - x));  % Eavg(0, 1)  % https://onlinelibrary.wiley.com/doi/full/10.1002/sim.8281
% How many samples per iteration and how many iterations?
if nargout > 7
    N = 10;
    alpha_ = 0.05;
    k = norminv(.5 + (1 - alpha_)/2);
    btstrpEavg = arrayfun(@(ii) mean(mean(bootstrp(round(1.0 * length(y)), @(x_, y_) abs(y_ - x_), x, y))), 1:N);
    figure;ksdensity(btstrpEavg);hold on;ksdensity(abs(y - x));plot([Eavg;Eavg], ylim', '-k', [mean(btstrpEavg); mean(btstrpEavg)], ylim', '-m');hold off;
    EavgCI = k * std(btstrpEavg) / sqrt(N);
end

Emax = max(abs(y - x));
E90 = quantile(abs(y - x), 0.9);

% net benefit
% TODO pg 313 For example, a threshold of 10% means that w = 1/9: the FP classifications are valued at one-ninth of the TP classification.
% w is a penalization factor not related to the predicted probability

ws = (0:.1:.9) ./ (1 - (0:.1:.9));

if length(obj.Pthresh) ~= length(obj.fpr) && (obj.Pthresh(1) == obj.Pthresh(2) || isnan(obj.Pthresh(1)) && isnan(obj.Pthresh(2)))
    obj.Pthresh = obj.Pthresh(2:end);
end

% NB = NaN(length(obj.Pthresh), length(ws));
NB = NaN(length(obj.tpr), length(ws));
for iw = 1:length(ws)
    w = ws(iw);    
    % w = 0.75;  % obj.Pthresh ./ (1 - obj.Pthresh);  % harm to benefit
    % NB(:, iw) = (obj.tpr - w .* obj.fpr) / length(obj.Pthresh);    
    NB(:, iw) = (obj.tpr - w .* obj.fpr) / nnz(~isnan(obj.Pthresh));
end
% figure;
% plot(obj.Pthresh, NB, '.-');xlabel('Predicted Probability');xlim([0 1]);ylabel('Net Benefit');grid('on');
% hold on;
% legend(arrayfun(@(x) sprintf('w = %1.2f', x), (0:.1:.9) ./ (1 - (0:.1:.9)), 'UniformOutput', false));

% x = cell2mat(arrayfun(@(x, y) repmat(x, [1 y]), (mybins)+mystride/2, N(inds), 'UniformOutput', false))';
% y = cell2mat(arrayfun(@(x, y) repmat(x, [1 y]), P2(inds), N(inds), 'UniformOutput', false))';

% x = log(temp(:, 1) ./ (1 - temp(:, 1)));

% p = polyfit(P, Y_2, 1);
% This is not the calibration slope and calibration intercept. I get that from logistic regression.
p = polyfit(x, y, 1);  % DOESN'T WORK 
m = p(1);

% calibration-in-the-large is b|a=1, not b
b = p(2);

citl = fminsearch(@(citl) norm(y - x - citl, 2), 0);
brierScore = mean((y - x).^2);

if plotflag
    if useBarPlot
        figure();ax = gca();bar(ax, (mybins)+mystride/2, P2);grid on;
        set(ax, 'XTick', [mybins 1]);
        xlabel(ax, 'P_N_O_C_O_S');ylabel(ax, 'P_a_c_t_u_a_l');
        for ii = 1:length(P2)
            text(ax, mybins(ii)+mystride/2, P2(ii), num2str(N(ii)), 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');
        end
    else
        
        xbnds = round(linspace(1, length(x), 11));
        [x2, y2] = deal(zeros(length(xbnds) - 1, 1));
        for ii = 1:length(xbnds) - 1
            inds = xbnds(ii):xbnds(ii+1);
            x2(ii) = mean(x(inds));
            y2(ii) = mean(y(inds));
        end
        figure();ax = gca();plot(ax, min(1.00, max(0.0, x2)), y2, '.', 'MarkerSize', 18);        
        % Lowess smoothing
        % hold(ax, 'on');plot(ax, min(1.00, max(0.0, x)), y, '-');hold(ax, 'off');
        
        xlabel(ax, 'P_N_O_C_O_S');ylabel(ax, 'Actual Proportion');
        % figure;plot((mybins)+mystride/2, P2);
        % hc = histcounts(temp(:, 1), 'BinEdges', (0:bw:1+bw) - bw/2);
        hc1 = histcounts(temp(temp(:, 2) == posOutcome, 1), 'BinEdges', (0:bw:1+bw) - bw/2);
        hc2 = histcounts(temp(temp(:, 2) == negOutcome, 1), 'BinEdges', (0:bw:1+bw) - bw/2);
        yyaxis(ax, 'right');hold(ax, 'on');
        % nImputations = 1; % size(temp, 1) / 1793;
        bar(ax, 0:bw:1, hc1/nImputations, 'FaceColor', [1 0 0], 'BarWidth', 1);
        bar(ax, 0:bw:1, -hc2/nImputations, 'FaceColor', [0 1 0], 'BarWidth', 1);
        hold(ax, 'off');
        
        if horizonDays == 7
            ylim(ax, [-100 2000]);
        elseif horizonDays == 28
            ylim(ax, [-100 1400]);
        end
        % ylim([0 max(hc)/sc]);
        ylabel(ax, 'Count');yyaxis(ax, 'left');
    end
    hold(ax, 'on');pl1 = plot(ax, [0;1], [0;1], '-k');hold(ax, 'off');grid(ax, 'on');
    % hold on;pl2 = plot([0; 1], m*[0; 1] + b, '-r');hold off;
    ylim(ax, [0 1]);xlim(ax, [0 1]);
    if exist('valset', 'var')
        title(ax, sprintf('%s Validation (%d Days)', valset, horizonDays));
    end
    % legend([pl1 pl2], {sprintf('NSE = %1.2f, Eavg(0, 1) = %1.2f, Emax(0, 1) = %1.2f', r2, Eavg, Emax) sprintf('slope = %1.3f, CITL = %1.2f', m, citl)}, 'Location', 'northwest');
    % legend([pl1 pl2], {sprintf('E_a_v_g(0, 1) = %1.2f, E_9_0(0, 1) = %1.2f', Eavg, E90) sprintf('slope = %1.2f, CITL = %1.3f', m, citl)}, 'Location', 'northwest');
    % legend([pl1 pl2], {sprintf('NSE = %1.2f, E_a_v_g(0, 1) = %1.2f, E_9_0(0, 1) = %1.2f, BS = %1.2f', r2, Eavg, E90, brierScore) ...
    %     sprintf('slope = %1.2f, CITL = %1.3f', m, citl)}, 'Location', 'northwest');
    
    % figure;bar((mybins)+mystride/2, N);grid on;xlabel('P_N_O_C_O_S');ylabel('N');set(gca, 'XTick', [mybins 1]);xlim([0 1]);
    
    B1 = 200;
    [Eavg_, m_] = deal(zeros(B1, 1));
    N_ = length(obj.y);
    for irep = 1:B1
        inds = randsample(N_, N_, true);
        % p = newModel.model.predict2(xx(inds, :));
        % m_(irep) = slope_metric(reshape(obj.P(inds), [], 1), reshape(obj.y(inds), [], 1));
        Eavg_(irep) = eavg_metric(reshape(obj.P(inds), [], 1), reshape(obj.y(inds), [], 1));
    end
    
    xl = xlim(ax);
    yl = ylim(ax);
    text(ax, xl(1) + 0.05 * (xl(2)-xl(1)), yl(2) - 0.05 * (yl(2) - yl(1)), ...
        sprintf([ ...
        ... % 'Calibration Slope = %1.3f [%1.3f, %1.3f]\n' ...
        'ICI = %1.3f [%1.3f %1.3f]'], ...
        ... % mean(m_, 'omitnan'), quantile(m_, 0.025), quantile(m_, 0.975), ...
        mean(Eavg_, 'omitnan'), quantile(Eavg_, 0.025), quantile(Eavg_, 0.975) ...
        ), 'EdgeColor', [0 0 0]);
end

r2 = [mean(Eavg_, 'omitnan'), (0.5 * (quantile(Eavg_, 0.975) - quantile(Eavg_, 0.025)) / norminv(1 - 0.05/2, 0, 1)) ^ 2, length(Eavg_)];