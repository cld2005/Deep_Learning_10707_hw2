classdef LM < handle
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        layers=[];
        train = [];
        validate = [];
        biases={};
        weights={};
        preactivation={};
        postactivation={};
        output=[];
        dict = [];
        num_hidden_neuron;
        num_of_layers;
        
        biases_all_zero={};
        weights_all_zero={}; 
        
        word_embed = [];
        
    end
    
    methods
        function zero_cell_array = create_new_all_zero (~,cell_array)
            zero_cell_array={};
            for i =1:size(cell_array,2)
                zero_cell_array{i} = gpuArray(zeros(size(cell_array{i},1),size(cell_array{i},2)));
            end
        end
        function LM_load_data(obj)
             [obj.train,obj.validate,obj.dict]=LoadData();
             obj.train=gpuArray(obj.train);
             obj.validate=gpuArray(obj.validate);
             obj.dict=obj.dict;
        end
        
        function init(obj,num_hidden_neuron,batch_size)
            obj.num_hidden_neuron=num_hidden_neuron;
            obj.num_of_layers=3;
            obj.layers(end+1)=16*3; %input layer is always 48
            obj.layers(end+1)=obj.num_hidden_neuron; % set the size of hidden layer
            obj.layers(end+1)=8000; % output layer is always 8000
            obj.LM_load_data();
            obj.word_embed =  gpuArray(normrnd(0,0.1,[8000 16 ])) ;
            
            weigits_size = horzcat(transpose(obj.layers(2:end)),transpose(obj.layers(1:end-1)));
            
            for i=2:obj.num_of_layers
                size_x = weigits_size(i-1,1);
                size_y =weigits_size(i-1,2);
                obj.weights{i} = gpuArray(normrnd(0,0.1,[size_x size_y ])) ;
                obj.biases{i}  = gpuArray( zeros(obj.layers(i),1));
            end
            
            
            for i=1:length(obj.layers)
                obj.preactivation{i} = gpuArray(zeros(obj.layers(i),batch_size));
                obj.postactivation{i} = gpuArray(zeros(obj.layers(i),batch_size));
            end
            obj.weights_all_zero=obj.create_new_all_zero(obj.weights);    
            obj.biases_all_zero=obj.create_new_all_zero(obj.biases);
        end
        
        function [corss_entropy_error, correct]=forward_prop(obj,train)
            x = train(:,1:size(train,2)-1);
            temp = obj.word_embed(x',:);
            temp=temp';
            obj.postactivation{1}=reshape (temp(:), [48 size(train,1)]);
            result = gpuArray(zeros(8000,size(train,1)));
            ind = sub2ind(size(result),train(:,size(train,2))',[1:size(train,1)]);
            
            result(ind)=1;
            for i = 2:obj.num_of_layers
                 obj.preactivation{i}=obj.weights{i}*obj.postactivation{i-1}+obj.biases{i};
                 obj.postactivation{i}=obj.preactivation{i};
            end
            obj.output = softmax(obj.postactivation{end});
            corss_entropy_error = -1*log(dot(obj.output,result));
             [~,indout]=max( obj.output);
             [~,indres]=max( result);
             correct =(indout==indres);
        end
        
        function [d_weight, d_bias,grad_h] = back_prop (obj,train)
            d_weight={};
            d_bias={};
            grad_h=[];
            
            result = gpuArray(zeros(8000,size(train,1)));
            ind = sub2ind(size(result),train(:,size(train,2))',[1:size(train,1)]);
            result(ind)=1;
            
            grad_out = (obj.output- result);
            
            for i=(obj.num_of_layers):-1:2
                d_weight{i}=grad_out*(transpose(obj.postactivation{i-1}));
                d_bias{i}=grad_out;
                grad_h = transpose(obj.weights{i})*grad_out;
                grad_out=grad_h;
            end
            
            
        end
        
         function [train_error,vali_error] = train_method(obj,num_hidden_neuron,learning_rate,batch_size,epoches,momentum)
             
             obj.init(num_hidden_neuron,batch_size);
             train_error=zeros(epoches,2);
             vali_error=zeros(epoches,2);
             
             
             for epoch = 1:epoches
                 fprintf('Epoch %d\n',epoch);
                 sample_count=0;
                 epoch_cross_entropy_error=0;
                 epoch_success_count=0;
                 batch_d_weight=obj.weights_all_zero;   
                 batch_d_bias=obj.biases_all_zero;
                 count=0;
                 for batch = 1:floor(size(obj.train,1)/batch_size)-1
                      d_weight=obj.weights_all_zero;

                      d_bias=obj.biases_all_zero;
                      
                      sample_count=sample_count+batch_size;
                      count=count+1;
                      start_bond  = 1+(batch-1)*batch_size;
                      end_bond=batch*batch_size;
                      batch_data = obj.train(start_bond:end_bond,:);
                      [error_value,correct_count]=obj.forward_prop(batch_data);
                      epoch_cross_entropy_error=epoch_cross_entropy_error+sum(error_value);
                      epoch_success_count=epoch_success_count+sum(correct_count);
                      
                      [sub_d_weight,sub_d_bias,grad_h]=obj.back_prop(batch_data);
                      
                      d_weight= cellfun(@(c1,c2) c1+c2,d_weight,sub_d_weight,'UniformOutput',false);
                      d_bias= cellfun(@(c1,c2) c1+c2,d_bias,sub_d_bias,'UniformOutput',false);
                      
                    batch_d_weight=cellfun(@(c1,c2) momentum*c1+(1.0/batch_size)*(c2),batch_d_weight,d_weight,'UniformOutput',false); 
                    %batch_d_weight=d_weight;
                    batch_d_bias=cellfun(@(c1,c2) momentum*c1+(1.0/batch_size)*(c2) ,batch_d_bias,d_bias,'UniformOutput',false);
                    %batch_d_bias=d_bias;
                    
                    obj.weights = cellfun(@(c1,c2) c1-learning_rate*c2,obj.weights,batch_d_weight,'UniformOutput',false); 
                    obj.biases = cellfun(@(c1,c2) c1-learning_rate*c2,obj.biases,batch_d_bias,'UniformOutput',false);
                    

                 end
                 train_error(epoch,1)=gather(epoch_cross_entropy_error)/sample_count;
                 train_error(epoch,2)=(1-gather(epoch_success_count)/sample_count)*100;
                 

             end
         end 
    end
    
end

