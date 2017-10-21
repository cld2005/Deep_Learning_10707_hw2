close all;
num_hidden_layer=1;
num_hidden_neuron=100;
learning_rate=0.1;
batch_size=1;
epoches=200;
momentum=0;
autoencoder = AE();
autoencoder.set_dropout_rate(0.1);

[train_error,vali_error] = autoencoder.train(num_hidden_layer,num_hidden_neuron,learning_rate,batch_size,epoches,momentum);

filter_plot(autoencoder,num_hidden_neuron,1);

try
save('autoencoder_5f.mat','autoencoder');
catch exception
fprintf('save final round failed');
end