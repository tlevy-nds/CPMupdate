function [Tdummy, mystruct] = dummytable(T, useRef, modNames)
% https://www.mathworks.com/matlabcentral/answers/261556-retain-dummy-variable-labels-from-converting-categorical-to-dummyvar
% Tdummy = dummytable(T) - convert categorical variables in table to dummy
% variables
%
% This function takes the categorical variables in a table and converts
% them to separate dummy variables with intelligent names.  This way they
% can be used in the Classification Learner App and the variable names make
% sense for feature selection, etc.
%
% Usage:
%
%     Tdummy = dummytable(T)
%
% Inputs:
%
%     T:        Table with categoricals or categorical variable
%
% Outputs: 
%
%     Tdummy:   T with categorical variables turned into dummy variables with
%               intelligent names
%
% Example:
%
%        % Simple Table
%        T = table(rand(10,1),categorical(cellstr('rbbgbgbbgr'.')),...
%           'VariableNames',{'Percent','Color'});
%        disp(T)
% 
%        % Turn it into a dummy table 
%        Tdummy = dummytable(T);
%        disp(Tdummy)
%
% See Also: dummyvar, table, categorical, classificationLearner
% Copyright 2015 The MathWorks, Inc.
% Sean de Wolski Apr 13, 2014

    if ~exist('modNames', 'var') || isempty(modNames)
        modNames = false;
    end

    if ~exist('useRef', 'var') || isempty(useRef)
        useRef = false;
    end

    if useRef
        k = 1;
    else
        k = 0;
    end

      % Error checking
      narginchk(1,3)    
      validateattributes(T,{'categorical', 'table'},{},mfilename,'T',1);
      % If it's a categorical, do out best to convert it to a table with an
      % intelligent variable name
      if iscategorical(T)
          % Try to use existing variable name
          cname = inputname(1);
          if isempty(cname)
              % It's a MATLAB Expression, default to Var1
              cname = 'Var1';
          end
          T = table(T,'VariableNames',{cname});
      end 
      % Identify categoricals and their names
      cats = varfun(@iscategorical,T,'OutputFormat','uniform');
      % Short circuit if there are no categoricals
      if ~any(cats)
          Tdummy = T;
          return
      end            
      % Store everything in a cell.  w will be the total width of the table
      % with each variable dummyvar'd
      w = nnz(~cats)+sum(varfun(@(x)numel(categories(x))-k,T(:,cats),'OutputFormat','uniform'));
      % Preallocate storage
      datastorage = cell(1,w);
      namestorage = cell(1,w);
      % Engine
      idx = 0; % Start nowhere in cell
      mystruct = [];
      for ii = 1:width(T)
          idx = idx+1;
          % Loop over table deciding what to do with each variable
          if cats(ii)
              % It's a categorical,
              % Extract it and build keep its categories and dummyvar
              Tii = T{:,ii};
              categoriesii = categories(Tii)';
              ncatii = numel(categoriesii); % How many?
              % Build dummy var as a row cell with columns in each
              dvii = num2cell(dummyvar(Tii), 1); % Dummy var then cell                                    
              % Build names
              namesii = strcat(T.Properties.VariableNames{ii}, '_', categoriesii);
              % Insert
              datastorage(idx:(idx+ncatii-1-k)) = dvii(k+1:end);
              namestorage(idx:(idx+ncatii-1-k)) = namesii(k+1:end);              
              
              myfield = T.Properties.VariableNames{ii};
              mycats = categoriesii;
              myinds = idx:(idx+ncatii-1-k);
              mynames = namesii(k+1:end);
              validnames = matlab.lang.makeValidName(mynames);
              mystruct = [mystruct struct('field', {myfield}, 'cats', {mycats}, 'inds', myinds, 'names', {mynames}, 'validnames', {validnames})];
              
              % Increment
              idx = idx+ncatii-1-k;
          else
              % Extract non categorical into current storage location
              datastorage{idx} = T{:,ii};
              namestorage(idx) = T.Properties.VariableNames(ii);
              
              myfield = T.Properties.VariableNames{ii};
              myinds = idx;
              mystruct = [mystruct struct('field', {myfield}, 'cats', {{}}, 'inds', myinds, 'names', {myfield}, 'validnames', {myfield})];
          end
      end
      % Build Tdummy with comma separated list expansion
      if modNames
          Tdummy = table(datastorage{:},'VariableNames', matlab.lang.makeValidName(namestorage));
      else
          Tdummy = table(datastorage{:},'VariableNames',namestorage);  % matlab.lang.makeValidName()
      end
      
end