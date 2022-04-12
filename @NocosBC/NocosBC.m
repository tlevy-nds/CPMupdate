classdef NocosBC < BinaryClassifier
    properties
    end
    
    methods
        function obj = NocosBC()
            obj = obj@BinaryClassifier();
            
            obj.mdl = Nocos();
        end
    end
end
