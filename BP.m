tic;%��ʱ��ʼ
%% ����������BP������
clear all
close all
clc
format compact
format long
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
P=input(n(1:9468),:)';      %ѵ������
T=output(n(1:9468))';
P_test=input(n(9469:9568),:)';%��������
T_test=output(n(9469:9568))';
clear data m n input output


%% BP����ѵ��
% %��ʼ������ṹ

disp('ѵ��bp������')
net=newff(P,T,10);
net.trainParam.epochs=100;%ѵ������
net.trainParam.lr=1;%ѧϰ��
net.trainParam.goal=0.001;%ѧϰĿ��
net.trainParam.max_fail = 200;% ��Сȷ��ʧ�ܴ��� 
net.trainParam.showWindow = false; 
net.trainParam.showCommandLine = false; 
%��������
%����ѵ��
net=train(net,P,T);
% ѵ�����
disp('����ѵ��bp������')
Y=sim(net,P);
  
%%%%%%%%%% ����ѵ��������Ҫ������ͼ����
% ����һ��
T=mapminmax('reverse',T,outputns);
Y=mapminmax('reverse',Y,outputns);
error=T-Y;
bp_train_mse=mse(error);;
figure
plot(error)
title('ѵ�������')
%
figure
plot(T,'r-*')
hold on
plot(Y,'bo-')
legend('�������','ʵ�����')
firstline = 'ѵ���׶�'; 
secondline = 'ʵ���������������Ľ������';
title({firstline;secondline},'Fontsize',12);
xlabel('ѵ��������')

%%
%===========================���Թ���=====================================%
%=======================================================================%
TY=sim(net,P_test);
% ����һ��

% T_test=mapminmax('reverse',T_test,outputns)';
% TY=mapminmax('reverse',TY,outputns)';
%%%%%%%%%% ������Խ��
result=[TY;T_test];
error=TY-T_test;
figure
plot(error)
title('���Լ����')
fprintf('���Լ�����������\n');

MSE=mse(error)
%=sum[ |y*-y|*100 / y ] /n 
 MAPE=sum(abs(TY-T_test)./T_test)/length(T_test)

MAE=mean(abs(TY-mean(T_test))) 

N = length(T_test);
R2 = (N*sum(TY.*T_test)-sum(TY)*sum(T_test))^2/((N*sum((TY).^2)-(sum(TY))^2)*(N*sum((T_test).^2)-(sum(T_test))^2))

a=corrcoef(TY,T_test);%Ƥ��ѷ���ϵ��
corrcoeff=a(1,2) 

error_bp=error;
TY_bp=TY;
save bp T_test error_bp TY_bp
t=toc %��ʱ����
%%
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