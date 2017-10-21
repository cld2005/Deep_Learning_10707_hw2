rounds=4;
anns=cell(rounds,1);
close all;
num_hidden_size=[20,100,200,500];
for i=1:rounds
    fprintf('round %d\n',i );
    %rng(i);
    num_hidden_layer=1;
    num_hidden_neuron=num_hidden_size(i);
    learning_rate=0.01;
    batch_size=1;
    epoches=200;
    momentum=0.5;
    anns{i} = ANN();
    anns{i}.ANN_load_data();
    [train_error,vali_error]=anns{i}.train(num_hidden_layer,num_hidden_neuron,learning_rate,batch_size,epoches,momentum);
    try
        save(['ann_hidden_layer' num2str(num_hidden_size(i)) '.mat'],'ann');
    catch exception
        fprintf('save round %d failed\n',i);
    end
    
    %filter_plot(anns{i},num_hidden_neuron,i);
end
plot_stats_all(anns);

try
    save('anns_hidden_layer.mat','anns');
catch exception
    fprintf('final save failed\n');
end




