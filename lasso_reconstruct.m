function vals = lasso_reconstruct(X, B, intercept, nfeatures)
[~, si] = sort(abs(B), 'descend');
nfeatures = min(nfeatures, nnz(B));
vals = X(:, si(1:nfeatures)) * reshape(B(si(1:nfeatures)), [], 1) + intercept;
