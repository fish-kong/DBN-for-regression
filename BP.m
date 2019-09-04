tic;%计时开始
%% 本程序用于BP神经网络
clear all
close all
clc
format compact
format long
%% 加载数据
load CCPP_Data.mat
inpu=data(:,2:5)';%输入
outpu = data(:,1)';%输出
%% 归一化
[input,~]=mapminmax(inpu,0,1);
[output,outputns]=mapminmax(outpu,0,1);
input=input';
output=output';
%% 划分数据集
%打乱样本 随机抽取9500组作为训练样本，剩下68组作为测试样本
rand('seed',0)

[m n ]=sort(rand(1,9568));
P=input(n(1:9468),:)';      %训练输入
T=output(n(1:9468))';
P_test=input(n(9469:9568),:)';%测试输入
T_test=output(n(9469:9568))';
clear data m n input output


%% BP网络训练
% %初始化网络结构

disp('训练bp神经网络')
net=newff(P,T,10);
net.trainParam.epochs=100;%训练次数
net.trainParam.lr=1;%学习率
net.trainParam.goal=0.001;%学习目标
net.trainParam.max_fail = 200;% 最小确认失败次数 
net.trainParam.showWindow = false; 
net.trainParam.showCommandLine = false; 
%建立网络
%网络训练
net=train(net,P,T);
% 训练误差
disp('结束训练bp神经网络')
Y=sim(net,P);
  
%%%%%%%%%% 计算训练误差，不重要，看看图就行
% 反归一化
T=mapminmax('reverse',T,outputns);
Y=mapminmax('reverse',Y,outputns);
error=T-Y;
bp_train_mse=mse(error);;
figure
plot(error)
title('训练集误差')
%
figure
plot(T,'r-*')
hold on
plot(Y,'bo-')
legend('期望输出','实际输出')
firstline = '训练阶段'; 
secondline = '实际输出与理想输出的结果对照';
title({firstline;secondline},'Fontsize',12);
xlabel('训练样本数')

%%
%===========================测试过程=====================================%
%=======================================================================%
TY=sim(net,P_test);
% 反归一化

% T_test=mapminmax('reverse',T_test,outputns)';
% TY=mapminmax('reverse',TY,outputns)';
%%%%%%%%%% 计算测试结果
result=[TY;T_test];
error=TY-T_test;
figure
plot(error)
title('测试集误差')
fprintf('测试集输出结果分析\n');

MSE=mse(error)
%=sum[ |y*-y|*100 / y ] /n 
 MAPE=sum(abs(TY-T_test)./T_test)/length(T_test)

MAE=mean(abs(TY-mean(T_test))) 

N = length(T_test);
R2 = (N*sum(TY.*T_test)-sum(TY)*sum(T_test))^2/((N*sum((TY).^2)-(sum(TY))^2)*(N*sum((T_test).^2)-(sum(T_test))^2))

a=corrcoef(TY,T_test);%皮尔逊相关系数
corrcoeff=a(1,2) 

error_bp=error;
TY_bp=TY;
save bp T_test error_bp TY_bp
t=toc %计时结束
%%
figure
plot(T_test,'-*')
hold on
plot(TY,'ko-')
firstline = '测试阶段'; 
secondline = '实际输出与理想输出的结果对照';
title({firstline;secondline},'Fontsize',12);
xlabel('测试样本数')
ylabel('功率幅幅值')
legend('期望输出','实际输出')