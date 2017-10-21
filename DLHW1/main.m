rounds=3;
anns=cell(rounds,1);
close all;
learning_rates=[0.01 0.2 0.5];
for i=1:rounds
    fprintf('round %d\n',i );
    %rng(i);
    num_hidden_layer=1;
    num_hidden_neuron=100;
    learning_rate=learning_rates(i);
    batch_size=1;
    epoches=1;
    momentum=0;
    anns{i} = ANN();
    anns{i}.ANN_load_data();
    [train_error,vali_error]=anns{i}.train(num_hidden_layer,num_hidden_neuron,learning_rate,batch_size,epoches,momentum);
    
    %filter_plot(anns{i},num_hidden_neuron,i);
end
plot_stats_all(anns);
%for i=1:rounds
    %plot_stats(anns{i}.train_error,anns{i}.vali_error);
%end
for i=1:size(anns,1)
    ann = anns{i};
    save(['ann_learning_rate' num2str(learning_rates(i)) '.mat'],'ann');
end



