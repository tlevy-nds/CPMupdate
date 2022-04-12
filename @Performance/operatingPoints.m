function [cm] = operatingPoints(obj)
% assert(length(obj.y) == length(obj.thresh)-1);
uy = unique(obj.y, 'sorted');

% assert(mean(obj.x(obj.y == uy(1)), 'omitnan') < mean(obj.x(obj.y == uy(2)), 'omitnan'))
if mean(obj.x(obj.y == uy(1))) < mean(obj.x(obj.y == uy(2)))
    % pass
else
    fprintf(1, 'assertion failed in operatingPoint: %f >= %f\n', mean(obj.x(obj.y == uy(1)), 'omitnan'), mean(obj.x(obj.y == uy(2)), 'omitnan'));
end

% I don't use this anymore and it took a while to compute so I subsampled it. There will be zeros for the skipped indices.
threshes = unique(round(linspace(1, length(obj.thresh), 1000)));
cm = zeros(2, 2, length(obj.thresh));
for ithresh = threshes
    cm(:, :, ithresh) = confusionmat(obj.y == uy(2), obj.x >= obj.thresh(ithresh));
end
