function [tuning] = tuning_fcn(lambda_, x, y, theta_, Sigma_)

% estimate thetahat given lambda_
R_ = Sigma_ ./ (lambda_ * lambda_');
yhat_ = (1./(1 + exp(-(theta_' * x))))';
Dl = -x * (y - yhat_);
D2l = inv(R_) + x * diag(yhat_ .* (1 - yhat_)) * x';

thetahat = theta_ - D2l \ Dl;
Sigmahat = inv(D2l);
R = Sigmahat ./ (lambda_ * lambda_');

% compute likelihood and prior using thetahat
yhat = (1./(1 + exp(-(thetahat' * x))))';
likelihood = sum(log(yhat .* y + (1 - yhat) .* (1 - y)));
R = (R + R') / 2;  % ensure symmetry because of numerical precision
prior = log(mvnpdf(thetahat, thetahat, R));

% compute the Jacobian and Hessian using thetahat and Sigmahat
D2l = inv(R) + x * diag(yhat .* (1 - yhat)) * x';
% d = length(x);
tuning = -0.5 * log(det(D2l)) + likelihood + prior;  % I added the - sign before det
% tuning = (2*pi)^(d/2) * sqrt(1/det(D2l)) * likelihood * prior;  % I added the - sign before det
% tuning = (2*pi)^(d/2) * sqrt(det(inv(D2l))) * likelihood * prior;  % I added the - sign before det