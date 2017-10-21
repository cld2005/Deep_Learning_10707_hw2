classdef testClass
    %UNTITLED2 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        list= [1 2 3 4 5];
        x=2;
        y=0;
    end
    
    methods
        function y=maxEle(obj)
            y=max(obj.list);
        end
        
        function y=call(obj,n)
            y=sigmoid(n);
        end
    end
    
end

