function [ x_train,x_validate,x_test] = LoadData()
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
    fprintf('reading training data ...\n')
    tr_data=textread('../DLHW1/digitstrain.txt','','delimiter',',');
    tr_data = tr_data(randperm(size(tr_data,1)),:);%shuffle data
     x_train=tr_data(:,1:784);
    
    
    fprintf('reading validation data ...\n')
    va_data=textread('../DLHW1/digitsvalid.txt','','delimiter',',');
    x_validate=va_data(:,1:784);

    fprintf('reading test data ...\n')
    ts_data=textread('../DLHW1/digitstest.txt','','delimiter',',');
    x_test=ts_data(:,1:784);

end

