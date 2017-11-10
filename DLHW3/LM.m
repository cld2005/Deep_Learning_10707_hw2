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
        embed_size=16;
        validation_batch_size = 128;
        m;
        train_error=[];
        vali_error=[];
        min_perplexity = 9999;
        linear = 1;
        index_map;
        
    end
    
    methods
        function set_linear(obj,linear)
            obj.linear=linear;
        end
        function clear_data(obj)
           obj.train = [];
           obj.validate = [];
        end
        function set_embed_size(obj,size)
            obj.embed_size=size;
        end
        
        function zero_cell_array = create_new_all_zero (~,cell_array)
            zero_cell_array={};
            for i =1:size(cell_array,2)
                zero_cell_array{i} = gpuArray(zeros(size(cell_array{i},1),size(cell_array{i},2)));
            end
        end
        
        function LM_load_data(obj)
             [obj.train,obj.validate,obj.m,obj.dict]=LoadData();
             obj.train=gpuArray(obj.train);
             obj.validate=gpuArray(obj.validate);
             obj.dict=obj.dict;
             obj.index_map = containers.Map(obj.dict,[1:size(obj.dict,1)]);
        end
        
        function [prediction]=predict_n_words(obj,start,n)
            prediction=cell(n,1);
            for i = 1:n
                result=obj.forward_predict(start);
                result_local=gather(result);
                prediction{i}=gather(result_local);
                if result_local==8000
                    break;
                end
                start=horzcat(start,result);
                start=start(:,end-2:end);
            end 
        end
        
        function init(obj,num_hidden_neuron,batch_size)
            obj.num_hidden_neuron=num_hidden_neuron;
            obj.num_of_layers=3;
            obj.layers(end+1)=obj.embed_size*3; %input layer is always 48 or embed size *3
            obj.layers(end+1)=obj.num_hidden_neuron; % set the size of hidden layer
            obj.layers(end+1)=8000; % output layer is always 8000
            obj.LM_load_data();
            obj.word_embed =  gpuArray(normrnd(0,0.1,[8000 obj.embed_size ])) ;
            
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
            obj.postactivation{1}=reshape (temp(:), [obj.embed_size*3 size(train,1)]);
            result = gpuArray(zeros(8000,size(train,1)));
            ind = sub2ind(size(result),train(:,size(train,2))',[1:size(train,1)]);
            
            result(ind)=1;
            for i = 2:obj.num_of_layers
                 obj.preactivation{i}=obj.weights{i}*obj.postactivation{i-1}+obj.biases{i};
                 if obj.linear==1 || i==obj.num_of_layers
                    obj.postactivation{i}=obj.preactivation{i};
                 else
                    obj.postactivation{i}=tanh(obj.preactivation{i});
                 end
            end
            obj.output = softmax(obj.postactivation{end});
            f=dot(obj.output,result);
            corss_entropy_error =sum( -1*log(f));
             [~,indout]=max( obj.output);
             [~,indres]=max( result);
             correct =(indout==indres);
        end
        
        function [corss_entropy_error, l]=forward_validation(obj,validation)
            x = validation(:,1:size(validation,2)-1);
            temp = obj.word_embed(x',:);
            temp=temp';
            postactivation={};
            preactivation={};
            postactivation{1}=reshape (temp(:), [obj.embed_size*3 size(validation,1)]);
            result = gpuArray(zeros(8000,size(validation,1)));
            ind = sub2ind(size(result),validation(:,size(validation,2))',[1:size(validation,1)]);
            
            result(ind)=1;
            for i = 2:obj.num_of_layers
                 preactivation{i}=obj.weights{i}*postactivation{i-1}+obj.biases{i};
                 if obj.linear==1 || i==obj.num_of_layers
                 postactivation{i}=preactivation{i};
                 else
                 postactivation{i}=tanh(preactivation{i});
                 end
            end
            output = softmax(postactivation{end});
            f=dot(output,result);
            corss_entropy_error =sum( -1*log(f));
            l=sum(log( sum(output.*result))/log(2));
          
        end
        
        function [predict_result]=forward_predict(obj,predict)
            x = predict(:,1:size(predict,2));
            temp = obj.word_embed(x',:);
            temp=temp';
            postactivation={};
            preactivation={};
            postactivation{1}=reshape (temp(:), [obj.embed_size*3 size(predict,1)]);
            result = gpuArray(zeros(8000,size(predict,1)));
            ind = sub2ind(size(result),predict(:,size(predict,2))',[1:size(predict,1)]);
            
            result(ind)=1;
            for i = 2:obj.num_of_layers
                 preactivation{i}=obj.weights{i}*postactivation{i-1}+obj.biases{i};
                 if obj.linear==1 || i==obj.num_of_layers
                 postactivation{i}=preactivation{i};
                 else
                 postactivation{i}=tanh(preactivation{i});
                 end
            end
            output = softmax(postactivation{end});
            
            [~,I]=max(output);
            
            predict_result=I';

          
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
                if obj.linear==1
                     grad_out=grad_h;
                else
                     grad_out=grad_h.*(1.-(tanh(obj.preactivation{i-1})).^2);
                end
            end
            
            
        end
        
         function [train_error,vali_error] = train_method(obj,num_hidden_neuron,learning_rate,batch_size,epoches,momentum)
             
             obj.init(num_hidden_neuron,batch_size);
             train_error=zeros(epoches,2);
             vali_error=zeros(epoches,2);
             
             
             for epoch = 1:epoches
                 tic
                 fprintf('Epoch %d\n',epoch);
                 sample_count=0;
                 epoch_cross_entropy_error=0;
                 epoch_success_count=0;

                 count=0;
                 for batch = 1:floor(size(obj.train,1)/batch_size)-1
                      batch_d_weight=obj.weights_all_zero;   
                      batch_d_bias=obj.biases_all_zero;
                      batch_d_embed=gpuArray(zeros(batch_size*3,obj.embed_size));
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
                      
                      d_embed = reshape (grad_h, [obj.embed_size size(grad_h,1)*size(grad_h,2)/obj.embed_size ])';
                      
                      embed_index = batch_data(:,1:3);
                      embed_index=reshape(embed_index' , [1 size(embed_index,1)*size(embed_index,2)])';
                      
                      d_weight= cellfun(@(c1,c2) c1+c2,d_weight,sub_d_weight,'UniformOutput',false);
                      d_bias= cellfun(@(c1,c2) c1+sum(c2,2),d_bias,sub_d_bias,'UniformOutput',false);
                      
                    batch_d_weight=cellfun(@(c1,c2) momentum*c1+(1.0/batch_size)*(c2),batch_d_weight,d_weight,'UniformOutput',false); 
                    %batch_d_weight=d_weight;
                    batch_d_bias=cellfun(@(c1,c2) momentum*c1+(1.0/batch_size)*(c2) ,batch_d_bias,d_bias,'UniformOutput',false);
                    %batch_d_bias=d_bias;
                    batch_d_embed = batch_d_embed+d_embed;
                    obj.weights = cellfun(@(c1,c2) c1-learning_rate*c2,obj.weights,batch_d_weight,'UniformOutput',false); 
                    obj.biases = cellfun(@(c1,c2) c1-learning_rate*c2,obj.biases,batch_d_bias,'UniformOutput',false);
                    
                    word_embed_temp=gather(obj.word_embed);
                    embed_index_temp=gather(embed_index);
                    batch_d_embed_temp= gather(batch_d_embed);
                    
                    for i =1:size(batch_d_embed,1)

                        word_embed_temp(embed_index_temp(i),:)=word_embed_temp(embed_index_temp(i),:)-learning_rate*batch_d_embed_temp(i);
                        
                    end
                    
                    obj.word_embed=gpuArray(word_embed_temp);
                   

                 end
                 train_error(epoch,1)=gather(epoch_cross_entropy_error)/sample_count;
                 train_error(epoch,2)=(1-gather(epoch_success_count)/sample_count)*100;
                 sample_count=0;
                 validation_corss_entropy_error=0;
                 l_sum=0;
                 for batch = 1:ceil(size(obj.validate,1)/obj.validation_batch_size)
                      start_bond  = 1+(batch-1)*obj.validation_batch_size;
                      end_bond=min(batch*obj.validation_batch_size,size(obj.validate,1));
                      sample_count=sample_count+(end_bond-start_bond)+1;
                      [corss_entropy_error, l_batch]=obj.forward_validation( obj.validate(start_bond:end_bond,:));
                      validation_corss_entropy_error=validation_corss_entropy_error+corss_entropy_error;
                      l_sum=l_sum+l_batch;
                 end
                 l=l_sum/size(obj.validate,1);
                 perplexity=2^(-l);
                 validation_corss_entropy_error=validation_corss_entropy_error/sample_count;
                 vali_error(epoch,1)=gather(validation_corss_entropy_error);
                 vali_error(epoch,2)=gather(perplexity);
                 fprintf('training cross entropy %f, error rate %f\n', train_error(epoch,1), train_error(epoch,2));
                 fprintf('validation cross entropy %f, perplexity %f\n', validation_corss_entropy_error, perplexity);
                 if vali_error(epoch,2)< obj.min_perplexity 
                     save(['best_model_size_' num2str(num_hidden_neuron) '_linear_' num2str(obj.linear) '.mat'],'obj');
                     obj.min_perplexity=vali_error(epoch,2);
                 end
                 toc
                
             end
             
             obj.train_error = train_error;
             obj.vali_error = vali_error;

         end 
    end
    
end

