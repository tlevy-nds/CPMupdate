function p = predict(obj, X)

predInds = get_pred_inds(X.Properties.VariableNames, obj.predictorNames);

[~, Ps] = obj.mdl.predict(X{:, predInds});

p = obj.recal(Ps);
