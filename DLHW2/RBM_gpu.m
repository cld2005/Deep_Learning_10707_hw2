classdef RBM_gpu  < handle
    %UNTITLED3 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        x_train=[];
        x_validate=[];
        x_test=[];
        num_visible;
        num_hiddenn;
        weights=[];
        bias_vh=[];
        bias_hv=[];
        train_error=[];
        vali_error=[];

        
        
    end
    
    methods

        function RBM_load_data(obj)
            [obj.x_train,obj.x_validate,obj.x_test]=LoadData();
            obj.x_train=gpuArray(obj.x_train);
            obj.x_validate=gpuArray(obj.x_validate);
            obj.x_test=gpuArray(obj.x_test);
        end
        function init(obj,num_visible,num_hiddenn)
            obj.num_hiddenn=num_hiddenn;
            obj.num_visible=num_visible;
            obj.RBM_load_data();
            %obj.weights=rand(obj.num_visible,obj.num_hiddenn);
            obj.weights=normrnd(0,0.1,[obj.num_hiddenn obj.num_visible ]);% 100*784 initialize to random gaussian
            obj.weights=gpuArray(obj.weights);
            obj.bias_vh=zeros(1,obj.num_hiddenn,'gpuArray');%1*100
            obj.bias_hv=zeros(1,obj.num_visible,'gpuArray');%1*784
        end
        function [train_error,vali_error] = train(obj,num_visible,num_hiddenn,learning_rate,batch_size,epoches,k)
            obj.init(num_visible,num_hiddenn);
            train_error=zeros(epoches,1,'gpuArray');
            vali_error=zeros(epoches,1,'gpuArray');
            
            
            for epoch = 1:epoches
                fprintf('Epoch %d\n',epoch);
                %{
                if mod(epoch,200)==0
                    learning_rate=learning_rate/2;
                     fprintf('learning_rate %d\n',learning_rate);
                end
                %}
                bathc_cross_entropy=0;
                count=0;
                for batch=1:floor(3000/batch_size)-1
                    count=count+1;
                    start_bond  = 1+(batch-1)*batch_size;
                    end_bond=batch*batch_size;
                    batch_data = obj.x_train(start_bond:end_bond,:);
                    positive_v = batch_data;

                    
                    positive_h = obj.h_given_v(positive_v);
                    
                    negative_v=positive_v;
                    negative_h = positive_h;
                    k=max(1,k);
                    for i=1:k
                        negative_v = obj.v_given_h(negative_h);
                        negative_h = obj.h_given_v(negative_v);
                    end
                    
                    d_weights = (positive_h'*positive_v - negative_h'*negative_v)/batch_size;                    
                    d_bias_vh = mean(positive_h-negative_h,1);
                    d_bias_hv = mean(positive_v-negative_v,1);
                    
                    obj.weights = obj.weights + learning_rate*d_weights;
                    obj.bias_hv =  obj.bias_hv + learning_rate*d_bias_hv;
                    obj.bias_vh =  obj.bias_vh + learning_rate*d_bias_vh;
                    
                    bathc_cross_entropy = bathc_cross_entropy+ obj.cal_cross_entropy(positive_v);
                    
                end
                train_error(epoch,1)= bathc_cross_entropy/count;
                vali_error(epoch,1) = obj.cal_cross_entropy(obj.x_validate);
                
               fprintf('training cross entropy %f\n',train_error(epoch,1));
               fprintf('validate cross entropy %f\n',vali_error(epoch,1));

            end
            
            obj.vali_error=vali_error;
            obj.train_error=train_error;
        end
        
        function h = h_given_v(obj,v)
            h= v*transpose(obj.weights);
            h=h+obj.bias_vh;
            h = arrayfun(@(x) 1.0/(1.0+exp(-x)),h);
            %h=arrayfun(@(x)binornd(1,x) ,h);
            %h=binornd(1,h);
        end
        
        function v = v_given_h(obj,h)
            temp= h*obj.weights;
            v = temp+obj.bias_hv;
            v = arrayfun(@(x) 1.0/(1.0+exp(-x)),v);
            v=binornd(1,v);
        end
        
        function bina = binarize (obj,arr)
            bina=binornd(1,arr);
        end
        
        
        function cross_entropy=cal_cross_entropy(obj,positive_v)
            h = positive_v*transpose(obj.weights)+obj.bias_vh;
            h = arrayfun(@(x) 1.0/(1.0+exp(-x)),h);
            
            v = h*obj.weights+obj.bias_hv;
            v = arrayfun(@(x) 1.0/(1.0+exp(-x)),v);
            temp =positive_v.*log(v)+(1-positive_v).*log(1-v);
            temp = sum(temp,2);
            cross_entropy = -mean (temp);
        end
        
        function clear_data(obj)
            obj.x_train=[];
            obj.x_validate=[];
            obj.x_test=[];
        end

    end
    
end

