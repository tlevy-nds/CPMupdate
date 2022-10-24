function [stacked, w1, w2] = stack(obj)

stacked = [];
for iimp = 1:obj.m
    stacked = [stacked; obj.imp{iimp}];
end

w1 = ones(size(stacked, 1), 1) / obj.m;

missing = missingness2(obj.orig);
f = sum(missing, 2) / size(missing, 2);
w2 = repmat((1 - f) / obj.m, [obj.m 1]);