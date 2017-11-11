lm=load('best_model_size_128_linear_0.mat');
lm=lm.obj;
index_map = containers.Map(lm.dict,[1:size(lm.dict,1)]);

words={'city','town';'business','company';'administration','government';'car','vehicle';'big','city';'no','yes';'university','college'};

%words={'administration','government'};
for round = 1:size(words,1)
    start_index=zeros(1,size(words,2));
    for i=1:size(words,2)
        start_index(1,i)=index_map(words{round,i});
    end
    
    word1 =lm.word_embed(start_index(1,1),:);
    word2 = lm.word_embed(start_index(1,2),:);
    
    fprintf('%s , %s: %f\n',words{round,1},words{round,2},norm(word1-word2))

end
