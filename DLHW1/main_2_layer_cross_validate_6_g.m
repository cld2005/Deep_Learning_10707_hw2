rounds=2;
anns=cell(rounds,1);
close all;
lumbdas=[0.00001,0.00005];
for i=1:rounds
    fprintf('round %d\n',i );
    %rng(i);
    num_hidden_layer=2;
    num_hidden_neuron=100;
    learning_rate=0.1;
    batch_size=1;
    epoches=200;
    momentum=0.5;
    anns{i} = ANN();
    anns{i}.ANN_load_data();
    anns{i}.set_lumbda(lumbdas(i));
    [train_error,vali_error]=anns{i}.train(num_hidden_layer,num_hidden_neuron,learning_rate,batch_size,epoches,momentum);
    ann = anns{i};
    try
        save(['ann_2_layer_corss_validation' num2str(lumbdas(i)) '.mat'],'ann');
    catch exception
        fprintf('save round %d failed\n',i);
    end
    ann=[];
    %filter_plot(anns{i},num_hidden_neuron,i);
end
plot_stats_all(anns);

try
    save('anns_2_layer_corss_validation.mat','anns');
catch exception
    fprintf('final save failed\n');
end



