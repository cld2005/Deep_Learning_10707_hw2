rounds=2;
anns=cell(rounds,1);
close all;
preload_set = [1 0];
for i=1:rounds
    fprintf('round %d\n',i );
    %rng(i);
    num_hidden_layer=1;
    num_hidden_neuron=100;
    learning_rate=0.1;
    batch_size=1;
    epoches=100;
    momentum=0;
    anns{i} = ANN();
    anns{i}.ANN_load_data();
    anns{i}.set_load_exernal_weights(preload_set(i));
    rbm=load('../DLHW2/answer/5/a/rbm_a.mat');
    rbm=rbm.rbms{1};
    anns{i}.set_pre_load_weight(gather(rbm.weights));
    [train_error,vali_error]=anns{i}.train(num_hidden_layer,num_hidden_neuron,learning_rate,batch_size,epoches,momentum);
    
end
plot_stats_all(anns);

for i=1:size(anns,1)
    ann = anns{i};
    save('hw2_5d.mat','ann');
end



