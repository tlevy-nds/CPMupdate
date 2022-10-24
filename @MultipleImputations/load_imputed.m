function load_imputed(obj, impFiles)

for iimp = 1:obj.m
    % load the imputed data set
    filename = impFiles{iimp};
    imp_ = readtable(filename);
    
    % remove Var1
    if ismember('Var1', imp_.Properties.VariableNames)
        imp_ = removevars(imp_, {'Var1'});
    end
    
    % convert to the appropriate data type    
    for ipred = 1:length(imp_.Properties.VariableNames)
        vn = imp_.Properties.VariableNames{ipred};
        if ismember(vn, obj.orig.Properties.VariableNames)
            if isnumeric(obj.orig.(vn))
                inds = isnan(obj.orig.(vn));
                vals = imp_.(vn);
                vals(~inds) = obj.orig.(vn)(~inds);
            elseif iscategorical(obj.orig.(vn))
                catNames = categories(obj.orig.(vn));
                vals  = categorical(imp_.(vn), catNames);
                inds = isundefined(obj.orig.(vn)) | ismember(obj.orig.(vn), 'NA');
                vals(~inds) = obj.orig.(vn)(~inds);
            elseif isdatetime(obj.orig.(vn))
                inds = isnat(obj.orig.(vn));
                vals = datetime(imp_.(vn), 'InputFormat', 'yyyy-MM-dd''T''HH:mm:SS''Z''');
                vals(~inds) = obj.orig.(vn)(~inds);
            end
            
            if isequal(vn, 'KidneyDisease') && ismember('DiabeticCKD', obj.orig.Properties.VariableNames) && ismember('HypertensiveCKD', obj.orig.Properties.VariableNames)
                assert(nnz((obj.orig.KidneyDisease(~inds) | obj.orig.DiabeticCKD(~inds) | obj.orig.HypertensiveCKD(~inds)) ~= vals(~inds)) <= 12);
            else
                % verify that the not-missing data is the same
                if length(vn) < 4 || ~isequal(vn(end-3:end), '_log')
                    if isnumeric(obj.orig.(vn))                        
                        if ~(nnz(abs(obj.orig.(vn) - vals) > 1e-6 & ~inds) <= 4)  % some values are right censored  
                            fprintf(1, 'fail\n');
                        end
                    else
                        if ~(nnz(obj.orig.(vn) ~= vals & ~inds) <= 4)  % some values are right censored    
                            fprintf(1, 'fail\n');
                        end
                    end
                end
            end
            
            imp_.(vn) = vals;
        elseif length(vn) >= 4 && isequal(vn(end-3:end), '_log')
            % Passive imputation is supposed to handle this      
            imp_.(vn) = log(imp_.(vn(1:end-4)));
        end
    end    
    
    obj.imp{iimp} = imp_;   
end