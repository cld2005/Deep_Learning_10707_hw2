close all;
lms=cell(1,1);
lms{1} = LM();
num_hidden_neuron = 256;
learning_rate=0.1;
batch_size=256;
epoches=100;
momentum=0;

%lm.init(num_hidden_neuron,batch_size);

%[error_value,correct_count]=lm.forward_prop(lm.train(1:10,:));
lms{1}.set_linear(1)
lms{1}.train_method(num_hidden_neuron,learning_rate,batch_size,epoches,momentum);
plot_stats_all(lms)
lms{1}.clear_data();
save(['size_' num2str(num_hidden_neuron) '_linear_' num2str(lms{1}.linear) '_final.mat'],'lms');
