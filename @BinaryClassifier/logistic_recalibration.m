function logistic_recalibration(obj, X, y)

p = obj.predict(X);
assert(max(abs(imag(p))) < 1e-4);
p = real(p);
p = min(1 - 1e-5, p);
lp = log(p ./ (1 - p));
inds = isfinite(lp);
calCoefs = glmfit(lp(inds), y(inds), 'binomial', 'Link', 'logit');

obj.alphaNew = calCoefs(1) + calCoefs(2) * obj.alphaNew;
obj.betaOverall = obj.betaOverall * calCoefs(2);
