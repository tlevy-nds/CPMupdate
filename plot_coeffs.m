allpredNames = unique([predNames{:}]);
w2 = NaN(size(w, 1), length(allpredNames));
for iw = 1:size(w, 1)
    predInds = get_pred_inds(allpredNames, predNames{iw});
    w2(iw, predInds) = w{iw};
end
hFigCoeffs = figure;axCoefs = gca(); plot(axCoefs, xvals(1:2:end), w2, '.-', 'MarkerSize', 8);
legend(axCoefs, strrep(allpredNames, '_', ' '));
grid(axCoefs, 'on'); xlim(axCoefs, xvals([1 end]));
xlabel(axCoefs, sprintf('Patients after %s', datestr(devDate, 'mmmm dd, yyyy')));
% ylabel(axCoefs,
title(axCoefs, sprintf('%s %s', modelType, strrep(updateMethod, '_', ' ')));
saveas(hFigCoeffs, sprintf('%s\\coeffs_%s_%d_%s.fig', figdir, modelType, horizon, updateMethod));