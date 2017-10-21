classdef AE < handle
    %UNTITLED4 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        
        layers = [];
        num_of_layers = 0;
        biases_all_zero={};
        weights_all_zero={}; % for 2 hidden layers 100*784/100*100/10*100
        biases={};
        weights=[]%use tie weights
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
        empty_cache = struct('gamma',1,'beta',0,'eps',0.00001);
        cache ={};
        batch_mode =0;
        eps=0.00001;
        pre_loaded_weights=[];
        load_exernal_weights=0;
        dropout_rate=0;
        
    
    end
    
    methods


        function set_dropout_rate(obj,rate)
            obj.dropout_rate=rate;
        end 
 
        function y=act_fuc(~,x)

            
            y=sigmoid(x);
        end
        
        function y=d_act_fuc(~,x)

            y=d_sigmoid(x);
        end
        
        function clear_training_data(obj)
            obj.x_train=[];
            obj.x_validate=[];
            obj.x_test=[];
            
        end

        function init(obj,num_hidden_neuron,num_hidden_layer)
            obj.num_of_layers = 3;
            obj.layers(end+1)=784;% input layer is always 784
            for i=1:num_hidden_layer
                 obj.layers(end+1)=num_hidden_neuron; % add hidden layers
            end
            obj.layers(end+1)=784; % output layer always 784
            
            %weigits_size = horzcat(transpose(obj.layers(2:end)),transpose(obj.layers(1:end-1)));
            
            for i=2:obj.num_of_layers
                obj.biases{i}  =  zeros(obj.layers(i),1);
            end
            
            obj.weights = normrnd(0,0.1, [num_hidden_neuron,784] );
            
            
            for i=1:length(obj.layers)
                obj.preactication{i} = zeros(obj.layers(i),1);
                obj.postactivation{i} = zeros(obj.layers(i),1);
            end
            
            obj.weights_all_zero = zeros(num_hidden_neuron,784);

            obj.biases_all_zero=obj.create_new_all_zero(obj.biases);

            
            
        end
        function ANN_load_data (obj)
            [obj.x_train,obj.x_validate,obj.x_test] = LoadData();
        end
        

        
        function [corss_entropy_error]=forward_prop(obj,x)
            obj.postactivation{1}=x';
            
            if obj.dropout_rate~=0
                mask = randsrc(784,1,[1 0; 1-obj.dropout_rate obj.dropout_rate]);
                obj.postactivation{1}=mask.*obj.postactivation{1};
            end

            
            for i = 2:obj.num_of_layers
                if i==2
                    obj.preactication{i} = obj.weights*obj.postactivation{i-1}+obj.biases{i};
                elseif i ==3
                    
                    obj.preactication{i} = transpose(obj.weights)*obj.postactivation{i-1}+obj.biases{i};
                end

                
               
                obj.postactivation{i} = arrayfun(@obj.act_fuc,obj.preactication{i});

            end
            
            obj.output = obj.postactivation{3};
            corss_entropy_error = -sum(x'.*log(obj.output)+(1-x').*log(1-obj.output));
     
   
        end
        function [d_weight_out, d_bias] = back_prop (obj,x)
            x=x';
            d_weight={};
            d_bias={};



            grad_out = (obj.output- x);
            for i=(obj.num_of_layers):-1:2%first layer is the input x
 
                d_weight{i}=grad_out*(transpose(obj.postactivation{i-1}));%?????? check 
                d_bias{i}=grad_out;
                %grad_h = transpose(obj.weights)*grad_out;
                
                if i==2
                    grad_h = transpose(obj.weights)*grad_out;
                elseif i ==3
                    grad_h = obj.weights*grad_out;

                end
                
                grad_out=grad_h.*arrayfun(@obj.d_act_fuc,obj.preactication{i-1});
          
            end

            
            d_weight_out = (d_weight{3}'+d_weight{2})*0.5;
            
            
            
            
            
            
            
        end
        

        

        
        function zero_cell_array = create_new_all_zero (~,cell_array)
            zero_cell_array={};
            for i =1:size(cell_array,2)
                zero_cell_array{i} = zeros(size(cell_array{i},1),size(cell_array{i},2));
            end
        end
        
        
        
        
        function [train_error,vali_error] = train(obj,num_hidden_layer,num_hidden_neuron,learning_rate,batch_size,epoches,momentum)
        obj.init(num_hidden_neuron,num_hidden_layer);
        obj.train_error=zeros(epoches,1);
        obj.vali_error=zeros(epoches,1);
        obj.ANN_load_data();
        
            for epoch = 1:epoches

                fprintf('Epoch %d\n',epoch);
                sample_count=0;
                epoch_cross_entropy_error=0;
                
                batch_d_weight=obj.weights_all_zero;   

                batch_d_bias=obj.biases_all_zero;
                
                
       

                
                for batch = 0:floor(3000/batch_size)-1
                    
                    d_weight=obj.weights_all_zero;

                    d_bias=obj.biases_all_zero;
                    
    
                    

                   for sub_index = 1:batch_size
                        sample_index = batch*batch_size+sub_index;
                        x=obj.x_train(sample_index,:);
                        [error_value] = obj.forward_prop(x);
                        
                        epoch_cross_entropy_error=epoch_cross_entropy_error+error_value;
                        
                        sample_count=sample_count+1;
                        [sub_d_weight,sub_d_bias] = obj.back_prop(x);
   
                        %d_weight= cellfun(@(c1,c2) c1+c2,d_weight,sub_d_weight,'UniformOutput',false);
                        d_weight=d_weight+sub_d_weight;
                        d_bias= cellfun(@(c1,c2) c1+c2,d_bias,sub_d_bias,'UniformOutput',false);
                        
                        if obj.batch_mode==1
                            d_gamma = cellfun(@(c1,c2) c1+c2,sub_d_gamma,d_gamma,'UniformOutput',false);
                            d_beta = cellfun(@(c1,c2) c1+c2,sub_d_beta,d_beta,'UniformOutput',false);
                        end
                    end
                    
                    %batch_d_weight=cellfun(@(c1,c2,c3) momentum*c1+(1.0/batch_size)*(c2)+ obj.lumbda*2*c3,batch_d_weight,d_weight,obj.weights,'UniformOutput',false); 
                    
                    batch_d_weight=momentum*batch_d_weight+(1.0/batch_size)*d_weight+obj.lumbda*2*obj.weights;
                    %batch_d_weight=d_weight;
                    batch_d_bias=cellfun(@(c1,c2) momentum*c1+(1.0/batch_size)*(c2) ,batch_d_bias,d_bias,'UniformOutput',false);
                    %batch_d_bias=d_bias;
                    
                    %obj.weights = cellfun(@(c1,c2) c1-learning_rate*c2,obj.weights,batch_d_weight,'UniformOutput',false); 
                    
                    obj.weights = obj.weights-learning_rate*batch_d_weight;
                    obj.biases = cellfun(@(c1,c2) c1-learning_rate*c2,obj.biases,batch_d_bias,'UniformOutput',false);



                end % end batch
                ave_error = epoch_cross_entropy_error/sample_count;

                [validation_err ]=  obj.validate(obj.x_validate);
                obj.train_error(epoch,1)=ave_error;
                obj.vali_error(epoch,1)=validation_err;
                fprintf('training cross entropy %f\n',ave_error);
                fprintf('validate cross entropy %f\n',validation_err);
            end % end epoch
            train_error=obj.train_error;
            vali_error=obj.vali_error;
  
            obj.clear_training_data();
        end % end train
        
        function [corss_entropy_error_rate] = validate(obj,x_validate)
            corss_entropy_error=zeros(size(x_validate,1),1);
            for i=1:size(x_validate,1)
                [e]=obj.forward_prop(x_validate(i,:));
                corss_entropy_error(i)=e;
            end
            
            corss_entropy_error_rate= mean(corss_entropy_error);

        end
         
        

    end
    
end

