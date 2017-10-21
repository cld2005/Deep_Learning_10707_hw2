close all;
ks=[20];
rounds=size(ks,2);
rbms=cell(rounds,1);
for i=1:rounds
    fprintf('round %d\n',i );

    num_visible=784;
    num_hiddenn=100;
    learning_rate=0.1;
    batch_size=1;
    epoches=200;
    k=ks(i);
    rbms{i} = RBM();
    [train_error,vali_error] = rbms{i}.train(num_visible,num_hiddenn,learning_rate,batch_size,epoches,k);
    filter_plot(rbms{1},100,i);
    rbms{i}.clear_data();
    try
    save('rbm_b.mat','rbms');
    catch exception
    fprintf('save round %d failed\n',i);
    end
end

plot_stats_all(rbms);


try
    save('rbm_b.mat','rbms');
catch exception
    fprintf('final save failed\n');
end