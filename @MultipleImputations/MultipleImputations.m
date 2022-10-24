classdef MultipleImputations < handle
    properties
        orig
        imp
    end
    
    properties (Dependent)
        m
    end
    
    methods
        function obj = MultipleImputations(orig, impFiles)
            % The original data can be imported into Matlab and the table can be saved in a mat file
            % imputed data can be loaded from csv files (after manually adjusting their headers)
            % and the variable types can be made to match the original
            assert(istable(orig));
            obj.orig = orig;
            
            for iimp = 1:length(impFiles)
                assert(isfile(impFiles{iimp}), 'imputed file %s not found', impFiles{iimp});
            end
            
            if ~isempty(impFiles)
                obj.imp = cell(1, length(impFiles));
                obj.load_imputed(impFiles);                
            end
        end
        
        [stacked, w1, w2] = stack(obj)
        
        [Tdummy, TdummyAll] = dummy(obj, keepInds)
        
        % This will produce obj.m results
        % [B, FitInfo] = lassoglm(obj, varargin)  % use extract_nvp and include nfeatures
        
        % Implements S1, S2, S3, or a variation of it from 
        % "Effect of Variable Selection Strategy on the Performance of Prognostic Models When Using Multiple Imputation"
        % [B, FitInfo] = pool(obj, method, opts)  % opts is the cutoff (1 for S1, ceil(obj.m / 2) for S2, obj.m for S3)
        
        function value = get.m(obj)
            value = length(obj.imp);
        end
    end
    
    methods (Access = private)
        load_imputed(obj, varargin)        
    end
end