function [train_data_norm,train_data]=data_process(numdely,a)

numdata = size(a,1);
numsample = numdata - numdely;
train_data_norm = zeros(numdely+1, numsample);
for i = 1 :numsample
    train_data_norm(:,i) = a(i:i+numdely)';
end     
data_num=size(train_data_norm,2);  
train_data=train_data_norm;

