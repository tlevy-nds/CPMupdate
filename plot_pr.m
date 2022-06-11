function meanVarN = plot_pr(ax, P, y, col, updateMethod)

[recall, precision, ~, prauc] = perfcurve(y, P, 1, 'XCrit', 'reca', 'YCrit', 'prec');

% https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.637.4054&rep=rep1&type=pdf
n = size(y, 1);
alpha_ = 0.05;
nu_ = log(prauc/(1-prauc));
tau_ = (n*prauc*(1-prauc))^-0.5;
phi_ = norminv(1-alpha_/2);
temp = deal([exp(nu_ - phi_*tau_) / (1 + exp(nu_ - phi_ * tau_)), exp(nu_ + phi_*tau_) / (1 + exp(nu_ + phi_ * tau_))]);
[praucLow, praucHigh] = deal(temp(1), temp(2));

hold(ax, 'on');
plot(ax, recall, precision, 'Color', col, 'LineWidth', 2, ...
    'DisplayName', sprintf('%s AUC = %1.3f [%1.3f, %1.3f]', strrep(updateMethod, '_', ' '), prauc, praucLow, praucHigh));
hold(ax, 'off');

logit = @(x) log(x ./ (1 - x));

meanVarN = [prauc, ((logit(praucHigh) - logit(praucLow)) / norminv(1 - alpha_/2, 0, 1)) ^ 2, length(y)];