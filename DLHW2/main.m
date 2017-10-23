close all;
rounds=1;
rbms=cell(rounds,1);
for i=1:rounds
    fprintf('round %d\n',i );

    num_visible=784;
    num_hiddenn=100;
    learning_rate=0.1;
    batch_size=200;
    epoches=1000 ;
    k=5;
    rbms{i} = RBM_gpu();
    [train_error,vali_error] = rbms{i}.train(num_visible,num_hiddenn,learning_rate,batch_size,epoches,k);
    filter_plot(rbms{1},100,i);
    rbms{i}.clear_data();
end

plot_stats_all(rbms);


try
    save('rbm_a.mat','rbms');
catch exception
    fprintf('final save failed\n');
end