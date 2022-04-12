function ind = choose_lambda(B, fitinfo, X, Y, nfeatures)
% TODO use AIC or BIC
% choose lambda based on some criterion
ind = find(fitinfo.DF <= nfeatures & fitinfo.DF > 0, 1, 'first');
% if the criterion isn't met, use this second criterion
if isempty(ind)
    [~, ind] = min(abs(fitinfo.DF - nfeatures));
end