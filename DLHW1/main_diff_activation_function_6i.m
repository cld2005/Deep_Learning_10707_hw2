rounds=3;
anns=cell(rounds,1);
close all;
act_func =['relu' 'tanh' 'sig'];
for i=1:rounds
    fprintf('round %d\n',i );
    %rng(i);
    num_hidden_layer=1;
    num_hidden_neuron=100;
    learning_rate=0.1;
    batch_size=1;
    epoches=200;
    momentum=0.5;
    anns{i} = ANN();
    anns{i}.ANN_load_data();
    anns{i}.set_lumbda(0.00001);
    anns{i}.set_active_func(act_func(i));
    [train_error,vali_error]=anns{i}.train(num_hidden_layer,num_hidden_neuron,learning_rate,batch_size,epoches,momentum);
    try
        ann=anns{i};
        save(['ann_anns_diff_act_func_' num2str(i) '.mat'],'ann');
    catch exception
        fprintf('save round %d failed\n',i);
    end
    
    %filter_plot(anns{i},num_hidden_neuron,i);
end
plot_stats_all(anns);

try
    save('anns_diff_act_func.mat','anns');
catch exception
    fprintf('final save failed\n');
end




