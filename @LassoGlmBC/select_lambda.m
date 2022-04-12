function [coefs, idx] = select_lambda(B, FitInfo, nfeatures)
% select lambda so that there are approximately 8 features
idx = find(FitInfo.DF <= nfeatures & FitInfo.DF > 0, 1, 'first');
% if the criterion isn't met, use this second criterion
if isempty(idx)
    [~, idx] = min(abs(FitInfo.DF - nfeatures));
end

B0 = FitInfo.Intercept(idx);
predInds =  B(:, idx) ~= 0;
% disp(char(myPredictors(predInds)));
coefs = [B0; B(:,idx)];