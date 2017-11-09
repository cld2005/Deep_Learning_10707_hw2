function [ train,validate,dictionary ] = LoadData()
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

fprintf('reading training data ...\n')

train=textread('FGI.txt','','delimiter',',');

fprintf('reading validation data  currently nothing ...\n')

fprintf('reading dictionary data ...\n')
dictionary=textread('dict.txt','%s');
 
validate=[];
fprintf('Load data finish ...\n')
end

