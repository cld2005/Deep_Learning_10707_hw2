close all;
size_of_hidden_neuron=[50 100 200 500];

rounds = size(size_of_hidden_neuron,2);
daes=cell(rounds,1);
for i=1:rounds
    fprintf('round %d\n',i );
    if i==2
        continue
    end
    
    num_hidden_layer=1;
    num_hidden_neuron=size_of_hidden_neuron(i);
    learning_rate=0.1;
    batch_size=1;
    epoches=200;
    momentum=0;
    daes{i} = AE();
    daes{i}.set_dropout_rate(0.1);
    [train_error,vali_error] = daes{i}.train(num_hidden_layer,num_hidden_neuron,learning_rate,batch_size,epoches,momentum);
    dae=daes{i};
    try
    save(['denoising_autoencoder_5g' num2str(size_of_hidden_neuron(i)) '.mat'],'dae');
    catch exception
    fprintf('save  round %d failed', i);
    end
end

plot_stats_all(daes);


%filter_plot(autoencoder,num_hidden_neuron,1);

try
save('denoising_autoencoder_5g.mat','daes');
catch exception
fprintf('save final round failed');
end