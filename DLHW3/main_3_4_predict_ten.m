lm=load('best_model_size_128_linear_0.mat');
lm=lm.obj;
index_map = containers.Map(lm.dict,[1:size(lm.dict,1)]);

start_text={'the','united','states';'city','of','new';'the','study','shows',;'the','company','has',;'at','the','end'};
for round = 1:size(start_text,1)
    start_index=zeros(1,size(start_text,2));
    for i=1:size(start_text,2)
        start_index(1,i)=index_map(start_text{round,i});
    end
    result=lm.predict_n_words(start_index,10);
    
    for i=1:size(start_text,2)
        fprintf('%s ',start_text{round,i});
    end
     fprintf('-> ');
    for i=1:size(result,1);
        fprintf('%s ',lm.dict{result{i}});
    end
    fprintf('\n');
end
