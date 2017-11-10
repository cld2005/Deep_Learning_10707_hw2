function [predict_result]=forward_predict(obj,predict)
    x = predict(:,1:size(predict,2)-1);
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
