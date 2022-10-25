function p = deLong(obj, roc)
assert(isa(roc, 'Performance'));

if isempty(obj.cvguidimp) || isempty(roc.cvguidimp)
    assert(isequal(roc.y, obj.y));
    ia = true(size(obj.y));
    ib = true(size(roc.y));
else
    ia = false(size(obj.y));
    ib = false(size(roc.y));
    [C, ia_, ib_] = intersect(obj.cvguidimp, roc.cvguidimp, 'rows');
    ia(ia_) = true;
    ib(ib_) = true;
    assert(isequal(roc.y(ib), obj.y(ia)));  % I asume the order is preserved
end

if exist('fastDeLong', 'file') ~= 2
    addpath('FULLPATH\DeLongUI-master');
end

uy = unique(obj.y, 'sorted');
uy2 = unique(roc.y, 'sorted');

[negOutcome, posOutcome] = deal(uy(1), uy(2));
[negOutcome2, posOutcome2] = deal(uy2(1), uy2(2));

indsPos = obj.y == posOutcome & ia;
indsNeg = obj.y == negOutcome & ia;

indsPos2 = roc.y == posOutcome2 & ib;
indsNeg2 = roc.y == negOutcome2 & ib;

spsizes = [nnz(indsPos) nnz(indsNeg)];
ratings = [obj.x(indsPos) obj.x(indsNeg); roc.x(indsPos2) roc.x(indsNeg2)];

[aucs, delongcov] = fastDeLong(struct('spsizes', spsizes, 'ratings', ratings));
p = calpvalue(aucs, delongcov) / 2;  % divide by 2 to convert to a 1-sided test
