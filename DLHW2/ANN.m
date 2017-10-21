classdef ANN < handle
    %UNTITLED4 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        
        layers = [];
        num_of_layers = 0;
        biases_all_zero={};
        weights_all_zero={}; % for 2 hidden layers 100*784/100*100/10*100
        biases={};
        weights={}; % for 2 hidden layers 100*784/100*100/10*100
        preactication={}; % for 2 hidden layers 784/100/100/10
        postactivation={}; % for 2 hidden layers 784/100/100/10
        output=[];
        x_train=[];
        y_train=[];
        x_validate=[];
        y_validate=[];
        x_test=[];
        y_test=[];
        train_error=[];
        vali_error=[];
        
        train_corss_entropy=[];
        vali_corss_entropy=[];
        lumbda=0;
        activation='sig';
        empty_cache = struct('gamma',1,'beta',0,'eps',0.00001);
        cache ={};
        batch_mode =0;
        eps=0.00001;
        pre_loaded_weights=[];
        load_exernal_weights=0;
        
    
    end
    
    methods
        function set_pre_load_weight(obj,weight)
            obj.pre_loaded_weights=weight;
        end
        function set_load_exernal_weights(obj,val)
            obj.load_exernal_weights=val;
        end
        function set_batch_mode(obj,mode)
            obj.batch_mode=mode;
        end
        function set_active_func(obj,x)
            obj.activation=x;
        end
        function y=act_fuc(obj,x)
           %{
            if obj.activation=='sig'
                y=sigmoid(x);
            elseif obj.activation=='relu'
                y= poslin(x);
            elseif obj.activation=='tanh'
                y=tanh(x);
            else
                y=sigmoid(x);
            end
            %}
            
            y=sigmoid(x);
        end
        
        function y=d_act_fuc(obj,x)
            %{
            if obj.activation=='sig'
                y=d_sigmoid(x);
            elseif obj.activation=='relu'
                if(x>0)
                    y=1;
                else
                    y=0;
                end
            elseif obj.activation=='tanh'
                y=1-(tanh(x))^2;
            else
                y=d_sigmoid(x);
            end
            %}
            y=d_sigmoid(x);
        end
        
        function clear_training_data(obj)
            obj.x_train=[];
            obj.y_train=[];
        end
        function set_lumbda (obj, l)
            obj.lumbda=l;
        end
        function init(obj,num_hidden_layer,num_hidden_neuron)
            obj.num_of_layers = num_hidden_layer+2;
            obj.layers(end+1)=784;% input layer is always 784
            for i=1:num_hidden_layer
                 obj.layers(end+1)=num_hidden_neuron; % add hidden layers
            end
            obj.layers(end+1)=10; % output layer
            
            weigits_size = horzcat(transpose(obj.layers(2:end)),transpose(obj.layers(1:end-1)));
            
            for i=2:obj.num_of_layers
                size_x = weigits_size(i-1,1);
                size_y =weigits_size(i-1,2);
                b=  sqrt(6)/(sqrt(obj.layers(i)+obj.layers(i-1)));
                obj.weights{i}=2*b*(rand(size_x,size_y))-b; % the first layer does not have weights
                obj.biases{i}  =  zeros(obj.layers(i),1);
     
                obj.cache{i} = obj.empty_cache;
  
            end
            
            if obj.load_exernal_weights~=0
                 obj.weights{2}=obj.pre_loaded_weights;
            end
            
            for i=1:length(obj.layers)
                
                
                obj.preactication{i} = zeros(obj.layers(i),1);
                obj.postactivation{i} = zeros(obj.layers(i),1);
            end
            
            obj.weights_all_zero=obj.create_new_all_zero(obj.weights);    

            obj.biases_all_zero=obj.create_new_all_zero(obj.biases);

            
            
        end
        function ANN_load_data (obj)
            [obj.x_train,obj.y_train,obj.x_validate,obj.y_validate,obj.x_test,obj.y_test] = dataLoad();
        end
        
        function [out,cache] = batch_norm_fprop (obj,x,cache)
            N=size(x,1);
            mu=mean(x);
            xmu = x-mu;
            sq = xmu.^2;
            var = sum(sq)/N;
            sqrtvar=sqrt(var+cache.eps);
            ivar = 1/sqrtvar;
            xhat = xmu *ivar;
            gammax = cache.gamma*xhat;
            out = gammax + cache.beta;
            
            cache.xhat = xhat;
            cache.xmu=xmu;
            cache.ivar=ivar;
            cache.sqrtvar=sqrtvar;
            cache.var=var;
            
        end
        
        function [corss_entropy_error, correct]=forward_prop(obj,x,y)
            obj.postactivation{1}=x';

            result = zeros(10,1);
            
            result(int32(y)+1)=int32(1);
            for i = 2:obj.num_of_layers
                raw_preactivation = obj.weights{i}*obj.postactivation{i-1}+obj.biases{i};
                %if(i<obj.num_of_layers &&obj.batch_mode==1)
                if(obj.batch_mode==1)
                    [obj.preactication{i},obj.cache{i}]=obj.batch_norm_fprop(raw_preactivation,obj.cache{i});
                else
                    obj.preactication{i} =raw_preactivation;
                end
                
                %obj.preactication{i} = obj.weights{i}*obj.postactivation{i-1}+obj.biases{i};
                if i~= obj.num_of_layers
                obj.postactivation{i} = arrayfun(@obj.act_fuc,obj.preactication{i});
                elseif i==obj.num_of_layers
                obj.postactivation{i}=obj.preactication{i};
                end
            end
            
            obj.output = softmax(obj.postactivation{end});
            
            omiga=0;
            if(obj.lumbda>0)
                for i=1:size(obj.weights,2)
                    omiga=omiga+sum(sum(obj.weights{i}));
                end 
            end
            
            corss_entropy_error = -1*log(dot(obj.output,result))+obj.lumbda*omiga;
            [~,indout]=max( obj.output);
            [~,indres]=max( result);
            correct= (indout==indres);
        end
        function [d_weight, d_bias,d_gamma,d_beta] = back_prop (obj,y)
            d_weight={};
            d_bias={};
            d_gamma={};
            d_beta={};
            result = zeros(10,1);
 
            result(int32(y)+1)=int32(1);

            grad_out = (obj.output- result);
            for i=(obj.num_of_layers):-1:2%first layer is the input x
                %if(i<obj.num_of_layers&&obj.batch_mode==1)
                if(obj.batch_mode==1)
                    [dx,dgamma,dbeta]=obj.batch_norm_bprop(grad_out,obj.cache{i});
                    grad_out=dx;
                    d_gamma{i}=dgamma;
                    d_beta{i}=dbeta;
                end
                d_weight{i}=grad_out*(transpose(obj.postactivation{i-1}));%?????? check 
                d_bias{i}=grad_out;
                grad_h = transpose(obj.weights{i})*grad_out;
                grad_out=grad_h.*arrayfun(@obj.d_act_fuc,obj.preactication{i-1});
          
            end
            
            
            
            
        end
        
        function [dx,dgamma,dbeta] = batch_norm_bprop (obj,dout,cache)
            N= size(dout,1);
            D = size(dout,2);
            %9
            dbeta = sum(dout);
          
            dgammax = dout;
              %8
            dgamma = dot(dgammax,cache.xhat);
            dxhat = dgammax*cache.gamma;
            %7
            divar = dot(dxhat,cache.xmu);
            dxmu1 = dxhat *cache.ivar;
            %6
            dsqrtvar=-1/(cache.sqrtvar)^2*divar;
            %5
            dvar = 0.5/(sqrt(cache.var+cache.eps))*dsqrtvar;
            %4
            dsq = 1/N* ones(N,D) *dvar;
            
            dxmu2 = 2 *cache.xmu.*dsq;
            dx1 = (dxmu1+dxmu2);
            dmu = -1 * sum(dxmu1+dxmu2);
            
            dx2 = 1/N *ones(N,D)*dmu;
            dx = dx1+dx2;
        end
        

        
        function zero_cell_array = create_new_all_zero (~,cell_array)
            zero_cell_array={};
            for i =1:size(cell_array,2)
                zero_cell_array{i} = zeros(size(cell_array{i},1),size(cell_array{i},2));
            end
        end
        function [train_error,vali_error] = train(obj,num_hidden_layer,num_hidden_neuron,learning_rate,batch_size,epoches,momentum)
        obj.init(num_hidden_layer,num_hidden_neuron);
        train_error=zeros(epoches,2);
        vali_error=zeros(epoches,2);
            for epoch = 1:epoches

                fprintf('Epoch %d\n',epoch);
                sample_count=0;
                epoch_cross_entropy_error=0;
                epoch_success_count=0;
                
                batch_d_weight=obj.weights_all_zero;   

                batch_d_bias=obj.biases_all_zero;
                
                
       

                
                for batch = 0:floor(3000/batch_size)-1
                    
                    d_weight=obj.weights_all_zero;

                    d_bias=obj.biases_all_zero;
                    d_gamma = cell(1,obj.num_of_layers);
                    d_beta = cell(1,obj.num_of_layers);
                    if obj.batch_mode==1
                        for i=2:obj.num_of_layers
                            d_gamma{i}=[0];
                            d_beta{i}=[0];
                        end
                    end

                   for sub_index = 1:batch_size
                        sample_index = batch*batch_size+sub_index;
                        [error_value,correct_count] = obj.forward_prop(obj.x_train(sample_index,:),obj.y_train(sample_index));
                        epoch_cross_entropy_error=epoch_cross_entropy_error+error_value;
                        epoch_success_count=epoch_success_count+correct_count;
                        sample_count=sample_count+1;
                        [sub_d_weight,sub_d_bias,sub_d_gamma,sub_d_beta] = obj.back_prop(obj.y_train(sample_index));
   
                        d_weight= cellfun(@(c1,c2) c1+c2,d_weight,sub_d_weight,'UniformOutput',false);
                        d_bias= cellfun(@(c1,c2) c1+c2,d_bias,sub_d_bias,'UniformOutput',false);
                        if obj.batch_mode==1
                            d_gamma = cellfun(@(c1,c2) c1+c2,sub_d_gamma,d_gamma,'UniformOutput',false);
                            d_beta = cellfun(@(c1,c2) c1+c2,sub_d_beta,d_beta,'UniformOutput',false);
                        end
                    end
                    
                    batch_d_weight=cellfun(@(c1,c2,c3) momentum*c1+(1.0/batch_size)*(c2)+ obj.lumbda*2*c3,batch_d_weight,d_weight,obj.weights,'UniformOutput',false); 
                    %batch_d_weight=d_weight;
                    batch_d_bias=cellfun(@(c1,c2) momentum*c1+(1.0/batch_size)*(c2) ,batch_d_bias,d_bias,'UniformOutput',false);
                    %batch_d_bias=d_bias;
                    
                    obj.weights = cellfun(@(c1,c2) c1-learning_rate*c2,obj.weights,batch_d_weight,'UniformOutput',false); 
                    obj.biases = cellfun(@(c1,c2) c1-learning_rate*c2,obj.biases,batch_d_bias,'UniformOutput',false);
                    if obj.batch_mode==1

                        for i=2:obj.num_of_layers-1
                            obj.cache{i}.gamma =  obj.cache{i}.gamma - learning_rate*d_gamma{i}/batch_size;
                            obj.cache{i}.beta =  obj.cache{i}.beta - learning_rate*d_beta{i}/batch_size;
                        end

                    end


                end % end batch
                ave_error = epoch_cross_entropy_error/sample_count;
                classification_err_rate = 100*(1-epoch_success_count/sample_count);

                [validation_err,validation_classification_err_rate ]=  obj.validate(obj.x_validate,obj.y_validate);
                train_error(epoch,1)=ave_error;
                train_error(epoch,2)=classification_err_rate;
                
                vali_error(epoch,1)=validation_err;
                vali_error(epoch,2)=validation_classification_err_rate;
                fprintf('training cross entropy %f, error rate %f\n',ave_error,classification_err_rate);
                fprintf('validate cross entropy %f, error rate %f\n',validation_err,validation_classification_err_rate);
            end % end epoch
            obj.train_error=train_error;
            obj.vali_error=vali_error;
            obj.clear_training_data();
        end % end train
        
        function [corss_entropy_error_rate, error_rate] = validate(obj,x_validate,y_validate)
            corss_entropy_error=zeros(size(x_validate,1),1);
            correct=zeros(size(x_validate,1),1);
            for i=1:size(x_validate,1)
                [e,c]=obj.forward_prop(x_validate(i,:),y_validate(i));
                corss_entropy_error(i)=e;
                correct(i)=c;
            end
            
            corss_entropy_error_rate= mean(corss_entropy_error);
            error_rate = 100- mean(correct)*100;

        end
         
        

    end
    
end

