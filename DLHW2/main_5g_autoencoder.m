close all;
size_of_hidden_neuron=[50 100 200 500];

rounds = size(size_of_hidden_neuron,2);
aes=cell(rounds,1);
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
    aes{i} = AE();
    [train_error,vali_error] = aes{i}.train(num_hidden_layer,num_hidden_neuron,learning_rate,batch_size,epoches,momentum);
    ae=aes{i};
    try
    save(['autoencoder_5g' num2str(size_of_hidden_neuron(i)) '.mat'],'ae');
    catch exception
    fprintf('save  round %d failed', i);
    end
end

plot_stats_all(aes);


%filter_plot(autoencoder,num_hidden_neuron,1);

try
save('autoencoder_5g.mat','aes');
catch exception
fprintf('save final round failed');
end