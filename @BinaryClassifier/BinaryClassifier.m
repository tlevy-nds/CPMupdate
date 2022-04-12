classdef BinaryClassifier < matlab.mixin.Copyable
   properties
       mdl
       
       predictorNames
       
       alphaNew
       betaOverall
   end
   
   properties (Constant)
      updateMethods = {'update_intercept', 'logistic_recalibration', 'reestimation', 'reestimation_extension'};
   end
   
   methods(Access = protected)
      % Override copyElement method:
      function cpObj = copyElement(obj)
         % Make a shallow copy of all four properties
         cpObj = copyElement@matlab.mixin.Copyable(obj);
         cpObj.mdl = copy(obj.mdl);
         
         % Make a deep copy of the DeepCp object
         % cpObj.pdPos{1} = copy(obj.pdPos{1});
         % cpObj.pdNeg{1} = copy(obj.pdNeg{1});
      end
   end
   
   methods
       function obj = BinaryClassifier()
           obj.alphaNew = 0;
           obj.betaOverall = 1;
       end
       
       update_intercept(obj, X, y)
       logistic_recalibration(obj, X, y)
       reestimation(obj, X, y, varargin)    
       reestimation_extension(obj, X, y, varargin)            
   end
   
   methods (Access=protected)
       p = recal(obj, Ps)       
   end
   
   methods (Abstract)
       train(obj, X, y, varargin)
       p = predict(obj, X)
       w = weights(obj)  % corresponding to predictorNames
   end
end