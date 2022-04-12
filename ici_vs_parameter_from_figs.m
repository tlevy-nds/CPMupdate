basedir = 'C:\Users\CBEM_NDDA_L1\Documents\slidingwindowupdate\figs\';
% subdirs = {'redo nocos 12-13-2021 v2 7 days' 'results 12-13-2021' 'results 12-13-2021';
%            'redo nocos 12-13-2021 v3'        'results 12-13-2021' 'results 12-13-2021'};
% subdirs = repmat({'results ici thresh 0.03 window size 2000'}, [2 3]);
subdirs = repmat({'causal 2'}, [2 3]);

% saDirs = {'ICI thresh 0.01', 'ICI thresh 0.03', 'ICI thresh 0.05' 'ICI thresh 0.07' 'ICI thresh 0.09' 'ICI thresh 0.11' 'ICI thresh 0.13' 'ICI thresh 0.15'};
% saDirs = {'window size 200' 'window size 500', 'window size 1000', 'window size 2000', 'window size 5000', 'window size 10000'};
% saDirs = {'impute 1' 'impute 2' 'impute 3' 'impute 4' 'impute 5'};
saDirs = {'acausal' 'baseline'};

horizons = [7 28];
modelTypes = {'nocos', 'LR', 'xgboost'};
updateMethods = {'no updates', 'update_intercept', 'logistic_recalibration', 'reestimation', 'reestimation_extension'};

hmsg = MessageUpdater();
hcleanup = onCleanup(@() delete(hmsg));

% 3 is for ICI, lower bound, upper bound
icis = zeros(length(saDirs), 3, length(horizons), length(modelTypes), length(updateMethods));

for isa = 1:length(saDirs)
    sadir = saDirs{isa};

    for ihorizon = 2  % 1:length(horizons)
        horizon = horizons(ihorizon);

        for imodelType = 1  % 1:length(modelTypes)
            modelType = modelTypes{imodelType};

            if isequal(sadir, 'baseline')                
                % subdir = 'baseline rng';
                subdir = subdirs{ihorizon, imodelType};
            else
                subdir = sadir;
            end

            for iupdateMethod = 1:length(updateMethods)
                updateMethod = updateMethods{iupdateMethod};

                filename = sprintf('%s\\%s\\%dday\\%s\\cal_%s_%d_%s.fig', basedir, subdir, horizon, modelType, ...
                    modelType, horizon, updateMethod);

                h = openfig(filename, 'invisible');
                myText = get(findobj(h, 'Type', 'Text'), 'String');
                close(h);
                if iscell(myText)
                    if length(myText) == 2
                        icis(isa, :, ihorizon, imodelType, iupdateMethod) = sscanf(myText{2}, 'ICI = %f [%f %f]');
                    else
                        icis(isa, :, ihorizon, imodelType, iupdateMethod) = sscanf(myText{1}, 'ICI = %f [%f %f]');
                    end
                else
                    icis(isa, :, ihorizon, imodelType, iupdateMethod) = sscanf(myText, 'ICI = %f [%f %f]');
                end

                hmsg.update_message(sprintf('%s %d %s %s', sadir, horizon, modelType, updateMethod));
            end
        end
    end
end
delete(hcleanup);

% (:, 1:3, 2, 1, :) is 28-day Nocos
y = squeeze(icis(:, 1, 2, 1, :));
neg = squeeze(icis(:, 1, 2, 1, :)) - squeeze(icis(:, 2, 2, 1, :));
pos = squeeze(icis(:, 3, 2, 1, :)) - squeeze(icis(:, 1, 2, 1, :));
% x = repmat([0.01:0.02:0.15]', [1 size(y, 2)]);
% x = repmat([200 500 1000 2000 5000 10000]', [1 size(y, 2)]);
% x = repmat([1 2 3 4 5]',  [1 size(y, 2)]);
x = repmat([0 1]',  [1 size(y, 2)]);
figure();
co = get(gca(), 'ColorOrder');
colororder(co([2 3 4 1 5:end], :));
errorbar(x, y, neg, pos);
legend(strrep(updateMethods, '_', ' '));
grid('on');
ylabel('Overall ICI');
% xlabel('ICI Update Threshold'); 
% xlabel('Window Length');