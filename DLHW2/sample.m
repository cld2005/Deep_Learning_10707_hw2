function  samples  = sample(rbm,step,num_sample)


    %rbm.RBM_load_data();
    
    %x = rbm.x_train(1:num_sample,:);
    x = rand(100,784);
    p_v=x;
    p_h = rbm.h_given_v(p_v);
    
    
    n_v = rbm.v_given_h(p_h);
    n_h = p_h;
    
    for i=1:step
        if mod(i,100)==0
            fprintf('Step %d\n',i);
        end
        n_v=rbm.v_given_h(n_h);
        n_h=rbm.h_given_v(n_v);
    end
    
    samples = n_v;
    
    
    
    
    
    f = figure;
    p = uipanel('Parent',f,'BorderType','none'); 
    p.Title = 'sample plot'; 
    p.TitlePosition = 'centertop'; 
    p.FontSize = 12;
    p.FontWeight = 'bold';
    %title(['filter plot round' num2str(round)]);
    for i=0:num_sample-1
        minv=min(samples(i+1,:));
        maxv=max(samples(i+1,:));
        norm=uint8((samples(i+1,:)-minv)*255/(maxv-minv));
        subplot(10,ceil(num_sample/10),i+1,'Parent',p),imshow(((reshape(norm,[28,28]))));
    end
    

end

