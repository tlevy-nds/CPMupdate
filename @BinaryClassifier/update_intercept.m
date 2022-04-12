function update_intercept(obj, X, y)

p = obj.predict(X);
lp = log(p ./ (1 - p));
inds = isfinite(lp);
myFcn = @(x) mean((y(inds) - 1./(1 + exp(-(x + lp(inds))))).^2);
obj.alphaNew = obj.alphaNew + fminsearch(myFcn, 0);
