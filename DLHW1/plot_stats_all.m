function plot_stats_all(anns)
%close all;
figure
hold on
title('Combined cross-entropy Plots')
legendInfo={};
line_width=2;
for i=1:size(anns,1)    
    plot(anns{i}.train_error(:,1),'LineWidth',line_width)
    legendInfo{end+1}=['training cross-entropy' num2str(i)];
    plot(anns{i}.vali_error(:,1),'LineWidth',line_width)
    legendInfo{end+1}=['Validation cross-entropy' num2str(i)];
    
end
legend(legendInfo);

hold off

% for i=1:size(anns,1)
% 
%     figure
%     plot(anns{i}.train_error(:,2))
%     title('Combined Classfiction Error Plots')
%     hold on
%     plot(anns{i}.vali_error(:,2))
%     legend('training classfication error','Validation classfication error')
%     hold off
% end
figure
hold on
title('Combined classication error')
legendInfo={};
for i=1:size(anns,1)    
    plot(anns{i}.train_error(:,2),'LineWidth',line_width)
    legendInfo{end+1}=['training classication error' num2str(i)];
    plot(anns{i}.vali_error(:,2),'LineWidth',line_width)
    legendInfo{end+1}=['Validation classication error' num2str(i)];
    
end
legend(legendInfo);

hold off
end