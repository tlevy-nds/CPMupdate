function zs = standardize(obj, X, setflag)

if ~exist('setflag', 'var') || isempty(setflag)
    setflag = false;
end

% set flag or empty
if setflag
    obj.mus = mean(X, 1, 'omitnan');
    obj.sigs = std(X, 1, 1, 'omitnan');
    obj.sigs(range(X, 1) == 0) = 1;
end

assert(~isempty(obj.mus) && ~isempty(obj.sigs));

zs = (X - reshape(obj.mus, 1, [])) ./ reshape(obj.sigs, 1, []);