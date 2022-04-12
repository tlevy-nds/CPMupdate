function p = predict(obj, X)

predInds = get_pred_inds(X.Properties.VariableNames, obj.predictorNames);

[~, temp1] = obj.mdl.predict(X(:, predInds));
Ps = temp1(:, 2);

p = obj.recal(Ps);
