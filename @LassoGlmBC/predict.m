function p = predict(obj, X)

predInds = get_pred_inds(X.Properties.VariableNames, obj.predictorNames);

Ps = glmval(obj.mdl.thetahat{1}, X{:, predInds}, 'logit');

p = obj.recal(Ps);
