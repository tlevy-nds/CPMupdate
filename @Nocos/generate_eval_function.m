function generate_eval_function(obj, filename)

% The region of support that I calculated may require k's < 0
assert(all([obj.pdPos{1}.LowerParameters(1), ...
    obj.pdNeg{1}.LowerParameters(1), ...
    obj.pdPos{1}.UpperParameters(1), ...
    obj.pdNeg{1}.UpperParameters(1)] < 0));

if ~exist('filename', 'var') || isempty(filename)
    [myfile, mypath] = uiputfile({'m-file (*.m)', '*.m'; 'All files (*.*)', '*.*'}, 'Save evaluate function', 'eval_nocos.m');
    if isequal(filename, 0)
        return
    end
    filename = [mypath myfile];
end

fid = fopen(filename, 'w');
hcleanup = onCleanup(@() fclose(fid));

%% Write the file
[~, name] = fileparts(filename);
fprintf(fid, ['function [pp7, pp28, x] = %s(' repmat('%s, ', [1  length(obj.predictorNames)-1]) '%s, survival)\n'], name, obj.predictorNames{:});
fprintf(fid, ['if ~exist(''survival'', ''var'') || isempty(survival)\n' ...
    '    %% if survival is true, return P, otherwise, return 1-P\n' ...
    '    survival = true;\n' ...
    'end\n' ...
    '%%%% Standardize the input data and evaluate the trained linear regression model\n' ...
    '%% mean of each measurement from the training set\n']);
fprintf(fid, ['mu_ = [' repmat('%f; ', [1 length(obj.predictorNames)-1]) '%f];\n'], obj.mus);
fprintf(fid, '%% standard deviation of each measurement from the training set\n');
fprintf(fid, ['sigma_ = [' repmat('%f; ', [1 length(obj.predictorNames)-1]) '%f];\n'], obj.sigs);
fprintf(fid, '%% model coefficients\n');
fprintf(fid, ['coefficients = [' repmat('%f; ', [1 length(obj.predictorNames)-1]) '%f];\n'], obj.lassoCoeffs);
fprintf(fid, 'bias = %f;\n\n', obj.biasTerm);

fprintf(fid, '%% create vector of measurements\n');
fprintf(fid, ['inputVector = [' repmat('%s; ', [1  length(obj.predictorNames)-1]) '%s];\n\n'], obj.predictorNames{:});

fprintf(fid, '%% standardize x using a z-score\n');
fprintf(fid, 'z = (inputVector - mu_) ./ sigma_;\n\n');

fprintf(fid, 'z(isnan(z)) = 0;  %% mean imputation\n\n');

fprintf(fid, '%% perform linear regression using the model\n');
fprintf(fid, 'x = sum(z .* coefficients) + bias;\n\n');

fprintf(fid, ['%%%% Evaluate Bayes rule to compute the posterior probability\n' ...
    '%% model priors from the training set\n']);
fprintf(fid, 'priorSurvival = %f;\n', obj.priorPos);
fprintf(fid, 'priorDeath = %f;\n', obj.priorNeg);
fprintf(fid, ['%% likelihood function parameters from the training set\n' ...
    '%% Pareto tails and another distribution (approximated as a quartic polynomial) in the center\n' ...
    '%% shape parameters (lower tail and upper tail for survival and death)\n']);
fprintf(fid, 'k1s = %f;\nk1d = %f;\nk2s = %f;\nk2d = %f;\n', ...
    obj.pdPos{1}.LowerParameters(1), obj.pdNeg{1}.LowerParameters(1), obj.pdPos{1}.UpperParameters(1), obj.pdNeg{1}.UpperParameters(1));
fprintf(fid, '%% scale parameters (lower tail and upper tail for survival and death)\n');
fprintf(fid, 'sigma1s = %f;\nsigma1d = %f;\nsigma2s = %f;\nsigma2d = %f;\n', ...
    obj.pdPos{1}.LowerParameters(2), obj.pdNeg{1}.LowerParameters(2), obj.pdPos{1}.UpperParameters(2), obj.pdNeg{1}.UpperParameters(2));
fprintf(fid, '%% threshold parameters (lower tail and upper tail for survival and death)\n');

theta1s = obj.pdPos{1}.icdf(obj.pdPos{1}.boundary(1));
theta1d = obj.pdNeg{1}.icdf(obj.pdNeg{1}.boundary(1));
theta2s = obj.pdPos{1}.icdf(obj.pdPos{1}.boundary(2));
theta2d = obj.pdNeg{1}.icdf(obj.pdNeg{1}.boundary(2));

fprintf(fid, 'theta1s = %f;\ntheta1d = %f;\ntheta2s = %f;\ntheta2d = %f;\n', theta1s, theta1d, theta2s, theta2d);
fprintf(fid, '%% quantiles\n');
fprintf(fid, 'p1s = %f;\np1d = %f;\np2s = %f;\np2d = %f;\n', ...
    obj.pdPos{1}.boundary(1), obj.pdNeg{1}.boundary(1), obj.pdPos{1}.boundary(2), obj.pdNeg{1}.boundary(2));
fprintf(fid, ['%% calibration correction for 28-day probability\n\n' ...
              'calCorrection = [%f %f];\n'], obj.calCorrection(1), obj.calCorrection(2));

fprintf(fid, ['%% Assign likelihoodSurvival to zero when x is outside of the generalized Pareto distribution''s region of support\n' ...
              '%% This will prevent NaNs or complex numbers when evaluatingn the distribution\n']);
fprintf(fid, ['if x < sigma1s/k1s + theta1s\n' ...
              '    ss = -1;\n' ...
              '    likelihoodSurvival = 0;\n']);
fprintf(fid, ['elseif x < theta1s\n' ...
    '    likelihoodSurvival = p1s*(1/sigma1s)*(1+k1s*(theta1s-x)/sigma1s)^(-1-1/k1s);\n' ...
    'elseif x < theta2s\n']);

% create the quartic curve fit
dts = theta2s - theta1s;
x_s = linspace(theta1s + dts/100, theta2s - dts/100, 101);
[p_s, S_s, mu_s] = polyfit(x_s, obj.pdPos{1}.pdf(x_s), 4);
figure();subplot(2, 1, 1);plot(x_s, obj.pdPos{1}.pdf(x_s));title('Survived');

fprintf(fid, ['    z = (x - %f) / %f;\n' ...
    '    likelihoodSurvival = %f*z^4 + %f*z^3 + %f*z^2 + %f*z + %f;\n'], ...
    mu_s(1), mu_s(2), p_s(1), p_s(2), p_s(3), p_s(4), p_s(5));
fprintf(fid, ['elseif x < theta2s - sigma2s/k2s\n' ...
    '    likelihoodSurvival = (1-p2s)*(1/sigma2s)*(1+k2s*(x-theta2s)/sigma2s)^(-1-1/k2s);\n' ...
    'else\n' ...
    '    ss = 1;\n' ...
    '    likelihoodSurvival = 0;\n' ...
    'end\n\n']);

fprintf(fid, ['%% Assign likelihoodDeath to zero when x is outside of the generalized Pareto distribution''s region of support\n' ...
              '%% This will prevent NaNs or complex numbers when evaluatingn the distribution\n']);
fprintf(fid, ['if x < sigma1d/k1d + theta1d\n' ...
              '    dd = -1;\n' ...
              '    likelihoodDeath = 0;\n']);

fprintf(fid, ['elseif x < theta1d\n' ...
    '    likelihoodDeath = p1d*(1/sigma1d)*(1+k1d*(theta1d-x)/sigma1d)^(-1-1/k1d);\n' ...
    'elseif x < theta2d\n']);

% create the quartic curve fit
dtd = theta2d - theta1d;
x_d = linspace(theta1d + dtd/100, theta2d - dtd/100, 101);
[p_d, S_d, mu_d] = polyfit(x_d, obj.pdNeg{1}.pdf(x_d), 4);
subplot(2, 1, 2);plot(x_d, obj.pdNeg{1}.pdf(x_d));title('Died');

fprintf(fid, ['    z = (x-%f) / %f;\n' ...
    '    likelihoodDeath = %f*z^4 + %f*z^3 + %f*z^2 + %f*z + %f;\n'], ...
    mu_d(1), mu_d(2), p_d(1), p_d(2), p_d(3), p_d(4), p_d(5));
fprintf(fid, ['elseif x < theta2d - sigma2d/k2d\n' ...
    '    likelihoodDeath = (1-p2d)*(1/sigma2d)*(1+k2d*(x-theta2d)/sigma2d)^(-1-1/k2d);\n' ...
    'else\n' ...
    '    dd = 1;\n' ...
    '    likelihoodDeath = 0;\n' ...
    'end\n\n']);

fprintf(fid, ['%% Assign pp7 to 1 when both distributions are evaluated outside of the region of support on the side closest to the survival distribution''s center\n' ...
              '%% Assign pp7 to 0 when both distributions are evaluated outside of the region of support on the side closest to the death distribution''s center\n']);
fprintf(fid, ['if likelihoodSurvival == 0 && likelihoodDeath == 0\n' ...
              '    assert(ss == dd, ''ss ~= dd'');  %% The distributions are expected to overlap\n' ...
              '    if mean([theta1s theta2s]) < mean([theta1d theta2d])\n' ...
              '        if ss == -1\n' ...
              '            pp7 = 1;\n' ...
              '        else\n' ...
              '            pp7 = 0;\n' ...
              '        end\n' ...
              '    else\n' ...
              '        if ss == -1\n' ...
              '            pp7 = 0;\n' ...
              '        else\n' ...
              '            pp7 = 1;\n' ...
              '        end\n' ...
              '    end\n' ...
              'else\n' ...
              '    pp7 = likelihoodSurvival * priorSurvival / ...\n' ...
              '        (likelihoodSurvival * priorSurvival + likelihoodDeath * priorDeath);\n' ...
              'end\n\n']);

fprintf(fid, ['%% The posterior probability can be clipped at 0.1 on the low end 0.95 on the high end\n' ...
    '%% check for underflow on each end and assign pp7 to max or min accordingly\n' ...
    'if likelihoodSurvival < 1e-7 && likelihoodDeath < 1e-7\n' ...
    '    ms = mean([theta1s theta2s]);\n' ...
    '    md = mean([theta1d theta2d]);\n' ...
    '    mboth = mean([ms md]);\n' ...
    '    if x < mboth && ms < md || x > mboth && ms > md\n' ...
    '        pp7 = 1;\n' ...
    '    else\n' ...
    '        pp7 = 0;\n' ...
    '    end\n' ...
    'else\n' ...
    '    pp7 = likelihoodSurvival * priorSurvival / ...\n' ...
    '        (likelihoodSurvival * priorSurvival + likelihoodDeath * priorDeath);\n' ...
    'end\n\n' ...
    'pp28 = calCorrection(1) * pp7 + calCorrection(2);' ...
    'pp7 = min(max(pp7, 0.1), 0.95);\n' ...
    'pp28 = min(max(pp28, 0.1), 0.95);\n\n']);

fprintf(fid, ['if ~survival\n' ...
    '    %% convert to a mortality calculator\n' ...
    '    pp7 = 1 - pp7;\n' ...
    '    pp28 = 1 - pp28;\n' ...
    'end\n']);

%%
delete(hcleanup);