classdef MessageUpdater < handle
    properties
        messagestr
    end
    
    methods
        function obj = MessageUpdater()
            obj.messagestr = '';
        end
        
        function delete(obj)
            obj.update_message('');
        end
        
        function update_message(obj, newmessage)        
            backspaces = repmat('\b', size(obj.messagestr));
            obj.messagestr = newmessage;
            fprintf(1, [backspaces obj.messagestr]);
        end
        
        function clear(obj)
            obj.messagestr = '';
        end
    end
end
