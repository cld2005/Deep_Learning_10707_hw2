rounds=4;
anns=cell(rounds,1);
close all;
hidden_layer_size = [20 100 200 500];
for i=1:rounds
    fprintf('round %d\n',i );
    %rng(i);
    num_hidden_layer=2;
    num_hidden_neuron=hidden_layer_size(i);
    learning_rate=0.1;
    batch_size=1;
    epoches=200;
    momentum=0.5;
    anns{i} = ANN();
    anns{i}.ANN_load_data();
    anns{i}.set_lumbda(0);
    [train_error,vali_error]=anns{i}.train(num_hidden_layer,num_hidden_neuron,learning_rate,batch_size,epoches,momentum);
    ann = anns{i};
    try
        save(['ann_2_layer_hidden_layer_size' num2str(hidden_layer_size(i)) '.mat'],'ann');
    catch exception
        fprintf('save round %d failed\n',i);
    end
    ann=[];
    %filter_plot(anns{i},num_hidden_neuron,i);
end
plot_stats_all(anns);

try
    save('anns_2_layer_hidden_layer_size.mat','anns');
catch exception
    fprintf('final save failed\n');
end



