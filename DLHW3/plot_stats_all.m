function plot_stats_all(lms)
figure
hold on
title('Combined cross-entropy Plots')
legendInfo={};
line_width=2;
for i=1:size(lms,1)   
    if isempty(lms{i})
        continue;
    end 
    
    plot(lms{i}.train_error(:,1),'LineWidth',line_width)
    legendInfo{end+1}=['training cross-entropy' num2str(i)];
    plot(lms{i}.vali_error(:,1),'LineWidth',line_width)
    legendInfo{end+1}=['Validation cross-entropy' num2str(i)]; 
end
legend(legendInfo);

hold off
end