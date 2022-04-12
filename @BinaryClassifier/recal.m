function p = recal(obj, Ps)

if obj.alphaNew ~= 0 || obj.betaOverall ~= 1
    lp = log(Ps ./ (1 - Ps));
    lp2 = obj.alphaNew + obj.betaOverall * lp;
    p = 1 ./ (1 + exp(-lp2));
else
    p = Ps;
end
