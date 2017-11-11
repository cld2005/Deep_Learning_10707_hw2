close all;
lms=cell(1,1);
lms{1} = LM();
lms{1}.disable_save_best_model();
lms{1}.set_embed_size(2);
num_hidden_neuron = 128;
learning_rate=0.01;
batch_size=256;
epoches=100;
momentum=0;

%lm.init(num_hidden_neuron,batch_size);

%[error_value,correct_count]=lm.forward_prop(lm.train(1:10,:));
dx = 0.1; dy = 0.1
lms{1}.set_linear(0)
lms{1}.train_method(num_hidden_neuron,learning_rate,batch_size,epoches,momentum);
plot_stats_all(lms)
lms{1}.clear_data();
embed_500 = gather(lms{1}.word_embed(1:500,:));
figure
hold on
scatter(embed_500(:,1),embed_500(:,2))
text(embed_500(:,1)', embed_500(:,2)', lms{1}.dict(1:500));
hold off
save(['size_' num2str(num_hidden_neuron) '_linear_' num2str(lms{1}.linear) '_embed_2_final.mat'],'lms');
