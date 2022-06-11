horizons = [28 7];
modelTypes = {'nocos', 'LR', 'xgboost'};
runbatch = true;  %#ok

leaveoutHospitals = {'LIJ'};  % {'FHH', 'FRK', 'GC', 'HNT', 'LHH', 'LIJ', 'NSUH', 'PLV', 'SIUH', 'SIUHS', 'SSH', 'SY'};
[aurocvals, auprvals, icivals] = deal(zeros(length(leaveoutHospitals), 3, length(horizons), length(modelTypes)));
for iloh = 1:length(leaveoutHospitals)
    leaveoutHospital = leaveoutHospitals{iloh};
    for ihorizon = 1:length(horizons)
        horizon = horizons(ihorizon);
        for imodeltype = 1:length(modelTypes)
            rng('default');

            modelType = modelTypes{imodeltype};
            if ~isfile(sprintf('figs\\auc_ici_vs_date_%s_%d.fig', modelType, horizon))
                t0 = clock();
                sliding_window_update;
                t1 = clock();
                dt = etime(t1, t0);
                fprintf(1, '%s %d took %f seconds\n', modelType, horizon, dt);
            end
            close('all');
            clear('X', 'y2');
        end
    end
end
runbatch = false;

% free resources
if exist('hcleanup', 'var')
    delete(hcleanup)
    clear('hcleanup');
end