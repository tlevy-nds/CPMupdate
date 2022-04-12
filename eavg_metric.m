function Eavg = eavg_metric(p, y)

temp = sortrows([reshape(p, [], 1) reshape(y, [], 1)], 1);
sw = 0.05;    % sliding window
temp2 = movmean(temp(:, 2), round(length(p)*sw), 'Endpoints', 'shrink');
Eavg = mean(abs(temp(:, 1) - temp2), 1);  % Eavg(0, 1)  % https://onlinelibrary.wiley.com/doi/full/10.1002/sim.8281
