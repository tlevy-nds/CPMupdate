function [m, c] = slope_metric(p, y, myDist, myLink)

% reshape p to 2 dimensions
szp = size(p);
p = reshape(p, szp(1), []);

if ~exist('myDist', 'var') || isempty(myDist)
    myDist = 'binomial';
end

if ~exist('myLink', 'var') || isempty(myLink)
    myLink = 'logit';
    lp = log(p ./ (1 - p));
else
    lp = p;
end

% loop over second dimension
m = zeros(1, size(p, 2));
for ii = 1:size(p, 2)
    c = glmfit(lp, y, myDist, 'Link', myLink);
    m(ii) = c(2);
    % c = polyfit(p(:, ii), y, 1);
    % m(ii) = c(1);    
end
% reshape back to the original shape
m = reshape(m, [1 szp(2:end)]);


