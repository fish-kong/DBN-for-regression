
%% 利用支持向量机进行回归
close all;
clear;
clc;
format compact;
tic
%% 数据提取

clear all
close all
clc
format compact
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
P_train=input(n(1:9468),:);      %训练输入
T_train=output(n(1:9468));
P_test=input(n(9469:9568),:);%测试输入
T_test=output(n(9469:9568));
clear data m n input output

%% 选择最佳的SVM参数c&g
%% 利用最佳的参数进行SVM网络训练
model = svmtrain(T_train,P_train,'-s 3 -t 2 -c 1.2 -g 2.8');
%% SVM网络预测
[Y,accuracy] = svmpredict(T_train,P_train,model);


T=mapminmax('reverse',T_train,outputns);
Y=mapminmax('reverse',Y,outputns);
error=T-Y;
SVM_train_mse=mse(error)
figure

figure
plot(error)
title('训练集误差')

figure
plot(T,'-*')
hold on
plot(Y,'ko-')
firstline = '测试阶段'; 
secondline = '实际输出与理想输出的结果对照';
title({firstline;secondline},'Fontsize',12);
xlabel('测试样本数')
ylabel('功率幅幅值')
legend('期望输出','实际输出')

%% SVM网络预测
[Y,accuracy] = svmpredict(T_test,P_test,model);


T_test=mapminmax('reverse',T_test,outputns);
TY=mapminmax('reverse',Y,outputns);



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
%% 结果分析
% 测试集的实际和预测图

error=T_test-TY;
fprintf('测试集输出结果分析\n');
figure
plot(error)
title('测试集误差')

MSE=mse(error)
 MAPE=sum(abs(TY-T_test)./T_test)/length(T_test)


MAE=mean(abs(TY-mean(T_test))) 

N = length(T_test);
R2 = (N*sum(TY.*T_test)-sum(TY)*sum(T_test))^2/((N*sum((TY).^2)-(sum(TY))^2)*(N*sum((T_test).^2)-(sum(T_test))^2))

a=corrcoef(TY,T_test);%皮尔逊相关系数
corrcoeff=a(1,2) 
error_svm=error;
TY_svm=TY;
save svm T_test error_svm TY_svm
toc
