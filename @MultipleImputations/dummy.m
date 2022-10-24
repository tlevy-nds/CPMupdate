function [Tdummy, TdummyAll] = dummy(obj, keepInds)
if ~exist('keepInds', 'var')
    keepInds = 1:size(obj.imp{1}, 2);
end

Tdummy = cell(1, obj.m);
TdummyAll = [];
for ii = 1:obj.m    
    Tdummy{ii} = dummytable(obj.imp{ii}(:, keepInds), true);
    TdummyAll = [TdummyAll; Tdummy{ii}];
end
    