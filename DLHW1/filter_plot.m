function filter_plot(ann,filters,round)
%mat2gray
f = figure;
p = uipanel('Parent',f,'BorderType','none'); 
p.Title = ['filter plot round' num2str(round)]; 
p.TitlePosition = 'centertop'; 
p.FontSize = 12;
p.FontWeight = 'bold';
%title(['filter plot round' num2str(round)]);
for i=0:filters-1
    minv=min(ann.weights{2}(i+1,:));
    maxv=max(ann.weights{2}(i+1,:));
    norm=uint8((ann.weights{2}(i+1,:)-minv)*255/(maxv-minv));
    subplot(10,ceil(filters/10),i+1,'Parent',p),imshow(((reshape(norm,[28,28]))));
end