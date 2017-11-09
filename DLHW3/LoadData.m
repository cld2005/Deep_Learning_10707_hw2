function [ train,validate,validation_size_m,dictionary] = LoadData()
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

fprintf('reading training data ...\n')

train=textread('FGI.txt','','delimiter',',');

fprintf('reading validation data  currently nothing ...\n')

fprintf('reading dictionary data ...\n')
dictionary=textread('dict.txt','%s');
validation_size_m = numel(textread('val.txt','%s'))
validate=textread('vFGI.txt','','delimiter',',');
fprintf('Load data finish ...\n')
end

