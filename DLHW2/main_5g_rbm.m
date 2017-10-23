clear all;
close all;
num_hiddenn_sizes=[50 100 200 500];
rounds=size(num_hiddenn_sizes,2);
rbms=cell(rounds,1);

for i=1:rounds
    
    fprintf('round %d\n',i );

    num_visible=784;
    num_hiddenn=num_hiddenn_sizes(i);
    learning_rate=0.1;
    batch_size=100;
    epoches=200;
    k=1;
    rbms{i} = RBM_gpu();
    [train_error,vali_error] = rbms{i}.train(num_visible,num_hiddenn,learning_rate,batch_size,epoches,k);
    %filter_plot(rbms{i},num_hiddenn,i);
    rbms{i}.clear_data();
    rbm=rbms{i};
    try
    save(['rbm_5g' num2str(num_hiddenn_sizes(i)) '.mat'],'rbm');
    catch exception
    fprintf('save  round %d failed', i);
    end
end

plot_stats_all(rbms);


try
    save('rbm_5g.mat','rbms');
catch exception
    fprintf('final save failed\n');
end