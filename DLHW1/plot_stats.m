function plot_stats(train_error,vali_error)
%close all;
figure
plot(train_error(:,1))
title('Combined cross-entropy Plots')
hold on
plot(vali_error(:,1))
legend('training cross-entropy','Validation cross-entropy ')
hold off


figure
plot(train_error(:,2))
title('Combined Classfiction Error Plots')
hold on
plot(vali_error(:,2))
legend('training classfication error','Validation classfication error')
hold off
end