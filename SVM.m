
%% ����֧�����������лع�
close all;
clear;
clc;
format compact;
tic
%% ������ȡ

clear all
close all
clc
format compact
%% ��������
load CCPP_Data.mat
inpu=data(:,2:5)';%����
outpu = data(:,1)';%���
%% ��һ��
[input,~]=mapminmax(inpu,0,1);
[output,outputns]=mapminmax(outpu,0,1);
input=input';
output=output';
%% �������ݼ�
%�������� �����ȡ9500����Ϊѵ��������ʣ��68����Ϊ��������
rand('seed',0)
[m n ]=sort(rand(1,9568));
P_train=input(n(1:9468),:);      %ѵ������
T_train=output(n(1:9468));
P_test=input(n(9469:9568),:);%��������
T_test=output(n(9469:9568));
clear data m n input output

%% ѡ����ѵ�SVM����c&g
%% ������ѵĲ�������SVM����ѵ��
model = svmtrain(T_train,P_train,'-s 3 -t 2 -c 1.2 -g 2.8');
%% SVM����Ԥ��
[Y,accuracy] = svmpredict(T_train,P_train,model);


T=mapminmax('reverse',T_train,outputns);
Y=mapminmax('reverse',Y,outputns);
error=T-Y;
SVM_train_mse=mse(error)
figure

figure
plot(error)
title('ѵ�������')

figure
plot(T,'-*')
hold on
plot(Y,'ko-')
firstline = '���Խ׶�'; 
secondline = 'ʵ���������������Ľ������';
title({firstline;secondline},'Fontsize',12);
xlabel('����������')
ylabel('���ʷ���ֵ')
legend('�������','ʵ�����')

%% SVM����Ԥ��
[Y,accuracy] = svmpredict(T_test,P_test,model);


T_test=mapminmax('reverse',T_test,outputns);
TY=mapminmax('reverse',Y,outputns);



figure
plot(T_test,'-*')
hold on
plot(TY,'ko-')
firstline = '���Խ׶�'; 
secondline = 'ʵ���������������Ľ������';
title({firstline;secondline},'Fontsize',12);
xlabel('����������')
ylabel('���ʷ���ֵ')
legend('�������','ʵ�����')
%% �������
% ���Լ���ʵ�ʺ�Ԥ��ͼ

error=T_test-TY;
fprintf('���Լ�����������\n');
figure
plot(error)
title('���Լ����')

MSE=mse(error)
 MAPE=sum(abs(TY-T_test)./T_test)/length(T_test)


MAE=mean(abs(TY-mean(T_test))) 

N = length(T_test);
R2 = (N*sum(TY.*T_test)-sum(TY)*sum(T_test))^2/((N*sum((TY).^2)-(sum(TY))^2)*(N*sum((T_test).^2)-(sum(T_test))^2))

a=corrcoef(TY,T_test);%Ƥ��ѷ���ϵ��
corrcoeff=a(1,2) 
error_svm=error;
TY_svm=TY;
save svm T_test error_svm TY_svm
toc
