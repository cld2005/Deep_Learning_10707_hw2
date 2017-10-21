rounds=1;
anns=cell(rounds,1);
close all;
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
    anns{i}.set_load_exernal_weights(1);
    ae=load('../DLHW2/answer/5/f/autoencoder_5f.mat');
    ae=ae.autoencoder;
    anns{i}.set_pre_load_weight(ae.weights);
    [train_error,vali_error]=anns{i}.train(num_hidden_layer,num_hidden_neuron,learning_rate,batch_size,epoches,momentum);
    
end
plot_stats_all(anns);


save('hw2_ann_5f.mat','anns');




