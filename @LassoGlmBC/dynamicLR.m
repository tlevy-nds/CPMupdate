function dynamicLR(obj, X, y)

options = optimoptions('fmincon');
options.Display = 'off';

predInds = get_pred_inds(X.Properties.VariableNames, obj.predictorNames);
x = [ones(1, size(X, 1)); X{:, predInds}'];
theta_ = obj.mdl.thetahat{1};
Sigma_ = obj.mdl.Sigmahat{1};
d = length(theta_);
p = 1 ./ (1 + exp(-(theta_' * x)));  % predict without applying recalibration coefficients
yhat_ = p';

fun = @(lambda_) -tuning_fcn(lambda_, x, y, theta_, Sigma_);
try
    obj.lambdahat = fmincon(fun, .99 * ones(d, 1), [], [], [], [], zeros(d, 1), ones(d, 1), [], options);
catch
    obj.lambdahat = 0.9 * ones(d, 1);
end
% lambdas{ii} = [lambdas{ii} lambdahat];

% lambdahat = sqrt(lambda_(ilambda) * ones(d, 1));

% R = cat(3, R, Sigma_ ./ diag(lambdahat));
% R_ = diag(diag(R(:, :, end)));
obj.mdl.R{1} = Sigma_ ./ (obj.lambdahat * obj.lambdahat');

R_ = obj.mdl.R{1};

% https://stats.stackexchange.com/questions/68391/hessian-of-logistic-function
Dl = -x * (y - yhat_);   % Jacobean  % (1. made this negative and ...)
D2l = inv(R_) + x * diag(yhat_ .* (1 - yhat_)) * x';  % Hessian

% https://datascience.stackexchange.com/questions/43178/intuition-behind-using-the-inverse-of-a-hessian-matrix-for-automatically-estimat

obj.mdl.thetahat{1} = theta_ - D2l \ Dl;
obj.mdl.Sigmahat{1} = inv(D2l);

% These buffers fill from the front, unlike thetahat
% timeBuffer = [curDtm timeBuffer];
% tempBuffer = [temp tempBuffer];
% tempBuffer2 = cat(3, temp2, tempBuffer2);

% if causalBuffer
%     updateInds = find(curDtm(2) - timeBuffer(1, :) > days(horizon/24));
%     if ~isempty(updateInds)
%         thetahat{ii} = [thetahat{ii}, tempBuffer(:, updateInds(1))];
%         Sigmahat{ii} = cat(3, Sigmahat{ii}, tempBuffer2(:, :, updateInds(1)));
%         tempBuffer(:, updateInds) = [];
%         tempBuffer2(:, :, updateInds) = [];
%         timeBuffer(:, updateInds) = [];
%     else
%         thetahat{ii} = [thetahat{ii}, thetahat{ii}(:, end)];
%         Sigmahat{ii} = cat(3, Sigmahat{ii}, Sigmahat{ii}(:, :, end));
%     end
% else
%     if Nbuf == 0
%         thetahat{ii} = [thetahat{ii}, temp];
%         Sigmahat{ii} = cat(3, Sigmahat{ii}, temp2);
%     elseif size(tempBuffer, 2) == Nbuf
%         thetahat{ii} = [thetahat{ii}, tempBuffer(:, end)];
%         Sigmahat{ii} = cat(3, Sigmahat{ii}, tempBuffer2(:, :, end));
%         tempBuffer = [temp tempBuffer(:, 1:end-1)];
%         tempBuffer2 = cat(3, temp2, tempBuffer2(:, :, 1:end-1));
%     else
%         tempBuffer = [temp tempBuffer];
%         tempBuffer2 = cat(3, temp2, tempBuffer2);
%         thetahat{ii} = [thetahat{ii}, tempBuffer(:, end)];
%         Sigmahat{ii} = cat(3, Sigmahat{ii}, tempBuffer2(:, :, end));
%     end
% end

% thetahat{ii} = [thetahat{ii}, temp];
% Sigmahat{ii} = cat(3, Sigmahat{ii}, inv(D2l));  % (2. made this positive)

% recompute tuning
R_ = obj.mdl.Sigmahat{1} ./ (obj.lambdahat * obj.lambdahat');
% compute likelihood and prior using thetahat
yhat_ = (1./(1 + exp(-(obj.mdl.thetahat{1}' * x))))';
likelihood = sum(log(yhat_ .* y + (1 - yhat_) .* (1 - y)));
R_ = (R_ + R_') / 2;  % ensure symmetry because of numerical precision
prior = log(mvnpdf(obj.mdl.thetahat{1}, obj.mdl.thetahat{1}, R_));
% compute the Jacobian and Hessian using thetahat and Sigmahat
D2l = inv(R_) + x * diag(yhat_ .* (1 - yhat_)) * x';
% d = length(x);
obj.mdl.tuning{1} = -0.5 * log(det(D2l)) + likelihood + prior;
