horizons = [28 7];
modelTypes = {'nocos', 'LR', 'xgboost'};
runbatch = true;  %#ok
for ihorizon = 1:length(horizons)
    horizon = horizons(ihorizon);
    for imodeltype = 1:length(modelTypes)
%         if ihorizon == 1 && imodeltype == 1
%             continue
%         end
        rng('default');

        modelType = modelTypes{imodeltype};
        if ~isfile(sprintf('figs\\auc_ici_vs_date_%s_%d.fig', modelType, horizon))
            sliding_window_update;
        end
        close('all');
        clear('X', 'y2');
    end
end
runbatch = false;

if exist('hcleanup', 'var')
    delete(hcleanup)
    clear('hcleanup');
end