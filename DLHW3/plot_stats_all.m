function plot_stats_all(lms)
figure
hold on
title('Combined cross-entropy Plots')
legendInfo={};
line_width=2;
for i=1:size(lms,1)   

    
    plot(lms{i}.train_error(:,1),'LineWidth',line_width)
    legendInfo{end+1}=['training cross-entropy hidden layer size ' num2str(lms{i}.num_hidden_neuron)];
    plot(lms{i}.vali_error(:,1),'LineWidth',line_width)
    legendInfo{end+1}=['Validation cross-entropy hidden layer size ' num2str(lms{i}.num_hidden_neuron)]; 
end
legend(legendInfo);
hold off


figure
hold on
title('Perplexity Plots')
legendInfo={};

for i=1:size(lms,1)   

    plot(lms{i}.vali_error(:,2),'LineWidth',line_width)
    legendInfo{end+1}=['Validation Perplexity hidden layer size ' num2str(lms{i}.num_hidden_neuron)]; 
end
legend(legendInfo);



hold off



end