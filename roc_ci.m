function [ci1, err] = roc_ci(auc, n1, n2, alpha_)

if ~exist('alpha_', 'var') || isempty(alpha_)
    alpha_ = 0.05;
end

q1 = auc/(2-auc);
q2 = 2*auc^2/(1+auc);
se = sqrt((auc*(1-auc) + (n1-1)*(q1-auc^2) + (n2-1)*(q2-auc^2))/(n1*n2));
za2 = norminv(.5 + (1-alpha_)/2);
ci1 = auc + za2*[-se se];
err = za2 * se;