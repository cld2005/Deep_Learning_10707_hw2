function [x_train,y_train,x_validate,y_validate,x_test,y_test] = dataLoad()

    fprintf('reading training data ...\n')
    tr_data=textread('digitstrain.txt','','delimiter',',');
    tr_data = tr_data(randperm(size(tr_data,1)),:);%shuffle data
    x_train=tr_data(:,1:784);
    y_train=tr_data(:,785);

    fprintf('reading validation data ...\n')
    va_data=textread('digitsvalid.txt','','delimiter',',');
    x_validate=va_data(:,1:784);
    y_validate=va_data(:,785);   

    fprintf('reading test data ...\n')
    ts_data=textread('digitstest.txt','','delimiter',',');
    x_test=ts_data(:,1:784);
    y_test=ts_data(:,785);

end
