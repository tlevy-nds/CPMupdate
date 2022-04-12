function [n, n1, n2, n3, n4] = get_min_sample_size(y, P, varargin)
% Implementation of
% Calculating the sample size required for developing a clinical prediction model
% y   - outcome vector of 1's and 0's
% P     - number of candidate predictors (max 30)
% Name/Value pairs
% Alpha - alpha level of the confidence interval (default = 0.05)
% Delta - absolute margin of error (default = 0.05)
% MAPE  - mean absolute prediction error between observed and true outcome probabilities (default = 0.05)
% S     - shrinkage factor (default = 0.9, 10% shrinkage)
% R2k   - conservative estimate in RNagalkerke2 (default = 0.15, explains this much of the variability)
p = inputParser();
p.addRequired('y', @(x) isnumeric(x) && isvector(x) && all(ismember(unique(x), [0 1])));
p.addRequired('P', @(x) isnumeric(x) && isscalar(x) && x >= 0 && x <= 30);
p.addParameter('Alpha', 0.05, @(x) isnumeric(x) && isscalar(x) && x > 0 && x < 0.5);
p.addParameter('Delta', 0.05, @(x) isnumeric(x) && isscalar(x) && x > 0);
p.addParameter('MAPE', 0.05, @(x) isnumeric(x) && isscalar(x) && x > 0);
p.addParameter('S', 0.9, @(x) isnumeric(x) && isscalar(x) && x > 0 && x < 1);
p.addParameter('R2k', 0.15, @(x) isnumeric(x) && isscalar(x) && x > 0 && x < 1);
p.parse(y, P, varargin{:});
y = p.Results.y;
P = p.Results.P;
alpha_ = p.Results.Alpha;
delta_ = p.Results.Delta;
mape = p.Results.MAPE;
S = p.Results.S;
R2k = p.Results.R2k;

%% STEP 1: What sample size will produce a precise estimate of the overall outcome risk or mean outcome value?
% number of samples required to estimate the intercept
E = min(nnz(y == 0), nnz(y == 1));
phihat = E / length(y);
k = norminv(1 - alpha_ / 2);
% k = tinv(1 - alpha_ / 2, n1 - P - 1);
n1 = (k / delta_)^2 * phihat * (1 - phihat);    % n1 = 243 < 11817

%% STEP 2: What sample size will produce predicted values that have a small mean error across all individuals?
n2 = exp((-0.508 + 0.259 * log(phihat) + 0.504 * log(P) - log(mape)) / 0.544);  % n2 = 1042 < 11817, assuming P = 30
% EPP = n2 * phihat / P;  % 6.8

%% STEP 3: What sample size will produce a small required shrinkage of predictor effects?
% 10% shrinkage to prevent overfitting
% R_CoxSnell^2 reflects SNR
n = length(y);
Lnull_log = E*log(E/n)+(n-E)*log(1-E/n);
Rcs2_max = 1 - exp(2 * Lnull_log/n);

% Not sure of the best way to estimate Rcs2
% Rcs2 = 1 - exp(-log(D/n)); % ???
% see https://en.wikipedia.org/wiki/Deviance_(statistics)
% and https://en.wikipedia.org/wiki/Coefficient_of_determination
Rcs2 = R2k * Rcs2_max;
n3 = P / ((S - 1) * log(1 - Rcs2 / S));  % n3 = 2712 < 11817

%% Step 4: What sample size will produce a small optimism in apparent model fit?
% RNagelkerke2 = Rcs2 / max(Rcs2) is a fundamental overall measure of model fit
% apparent and optimism adjusted
S_ = Rcs2 / (Rcs2 + delta_ * Rcs2_max);
n4 = P / ((S_ - 1) * log(1 - Rcs2 / S_));  % n4 = 1183 < 11817

%% determine max n
n = max([n1 n2 n3 n4]);
