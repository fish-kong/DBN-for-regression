tic;%计时开始
%% 本程序用于DBN+BP预测，所属回归预测类
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
%打乱样本 随机抽取9500组作为训练样本，剩下100组作为测试样本
rand('seed',0)
n=1:1:9568;
% [m n ]=sort(rand(1,9568));
P=input(n(1:9468),:);      %训练输入
T=output(n(1:9468));
P_test=input(n(9469:9568),:);%测试输入
T_test=output(n(9469:9568));
clear data m n input output
%% 训练样本构造，分块，批量
numcases=263;%每块数据集的样本个数
numdims=size(P,2);%单个样本的大小
numbatches=36;%将9500组训练样本，分成95批，每一批100组
% 训练数据
for i=1:numbatches
    train1=P((i-1)*numcases+1:i*numcases,:);
    batchdata(:,:,i)=train1;
end%将分好的10组数据都放在batchdata中



%% 2.训练RBM
%% rbm参数
maxepoch=100;%训练rbm的次数
numhid=45; numpen=44; numpen2=54; numpen3=43;%dbn隐含层的节点数
disp('构建一个4层的深度置信网络DBN用于特征提取');
%% 无监督预训练
fprintf(1,'Pretraining Layer 1 with RBM: %d-%d ',numdims,numhid);
restart=1;
rbm1;%使用cd-k训练rbm，注意此rbm的可视层不是二值的，而隐含层是二值的
vishid1=vishid;hidrecbiases=hidbiases;


fprintf(1,'\nPretraining Layer 2 with RBM: %d-%d ',numhid,numpen);
batchdata=batchposhidprobs;%将第一个RBM的隐含层的输出作为第二个RBM 的输入
numhid=numpen;%将numpen的值赋给numhid，作为第二个rbm隐含层的节点数
restart=1;
rbm1;
hidpen=vishid; penrecbiases=hidbiases; hidgenbiases=visbiases;

fprintf(1,'\nPretraining Layer 3 with RBM: %d-%d ',numpen,numpen2);%200-100
batchdata=batchposhidprobs;%显然，将第二哥RBM的输出作为第三个RBM的输入
numhid=numpen2;%第三个隐含层的节点数
restart=1;
rbm1;
hidpen2=vishid; penrecbiases2=hidbiases; hidgenbiases2=visbiases;

fprintf(1,'\nPretraining Layer 4 with RBM: %d-%d\n ',numpen2,numpen3);%200-100
batchdata=batchposhidprobs;%显然，将第二哥RBM的输出作为第三个RBM的输入
numhid=numpen3;%第三个隐含层的节点数
restart=1;
rbm1;
hidpen3=vishid; penrecbiases3=hidbiases; hidgenbiases3=visbiases;


%%%% 将预训练好的RBM用于初始化DBN权重%%%%%%%%%
w1=[vishid1; hidrecbiases]; 
w2=[hidpen; penrecbiases]; 
w3=[hidpen2; penrecbiases2];
w4=[hidpen3; penrecbiases3];

%% 有监督回归层训练
%===========================训练过程=====================================%
%==========DBN无监督用于提取特征，需要加上有监督的回归层==================%
%由于含有偏执，所以实际数据应该包含一列全为1的数，即w0x0+w1x1+..+wnxn 其中x0为1的向量 w0为偏置b
N1 = size(P,1);
digitdata = [P ones(N1,1)];

w1probs = 1./(1 + exp(-digitdata*w1));
w1probs = [w1probs  ones(N1,1)];

w2probs = 1./(1 + exp(-w1probs*w2));
w2probs = [w2probs ones(N1,1)];

w3probs = 1./(1 + exp(-w2probs*w3)); 
w3probs = [w3probs ones(N1,1)];

w4probs = 1./(1 + exp(-w3probs*w4)); 
H = w4probs'; %DBN的输出  也是BP的输入


%% BP网络训练
inputn=H;
outputn=T';
% %初始化网络结构
s1=100;%隐含层节点
disp('训练bp神经网络')
net=newff(inputn,outputn,s1);
net.trainParam.epochs=100;%训练次数
net.trainParam.lr=1;%学习率
net.trainParam.goal=0.0001;%学习目标
net.trainParam.max_fail = 200;% 最小确认失败次数 
net.trainParam.showWindow = false; 
net.trainParam.showCommandLine = false; 
%建立网络
%网络训练
net=train(net,inputn,outputn);
% 训练误差
disp('结束训练bp神经网络')
Y=sim(net,inputn);
  
%%%%%%%%%% 计算训练误差，不重要，看看图就行
% 反归一化
T=mapminmax('reverse',T,outputns);
Y=mapminmax('reverse',Y,outputns)';
error=T-Y;
bp_train_mse=mse(error);
figure
plot(error)
title('训练集误差')
%
figure
plot(T,'k-*')
hold on
plot(Y,'o-')
legend('期望输出','实际输出')
firstline = '训练阶段'; 
secondline = '实际输出与理想输出的结果对照';
title({firstline;secondline},'Fontsize',12);
xlabel('训练样本数')

%%
%===========================测试过程=====================================%
%=======================================================================%
N2 = size(P_test,1);
digitdata = [P_test ones(N2,1)];
w1probs = 1./(1 + exp(-digitdata*w1));
w1probs = [w1probs  ones(N2,1)];
w2probs = 1./(1 + exp(-w1probs*w2));
w2probs = [w2probs ones(N2,1)];
w3probs = 1./(1 + exp(-w2probs*w3)); 
w3probs = [w3probs ones(N2,1)];
w4probs = 1./(1 + exp(-w3probs*w4)); 
H = w4probs'; %DBN的输出  也是BP的输入

TY=sim(net,H);
% 反归一化

T_test=mapminmax('reverse',T_test,outputns);
TY=mapminmax('reverse',TY,outputns)';
%%%%%%%%%% 计算测试结果
result=[TY;T_test];
error=TY-T_test;
figure
plot(error)
title('测试集误差')
fprintf('测试集输出结果分析\n');


MSE=mse(error)
 
MAPE=sum(abs(TY-T_test)./T_test)/length(T_test)
MAE=mean(abs(TY-mean(T_test))) 
N = length(T_test);
R2 = (N*sum(TY.*T_test)-sum(TY)*sum(T_test))^2/((N*sum((TY).^2)-(sum(TY))^2)*(N*sum((T_test).^2)-(sum(T_test))^2))

a=corrcoef(TY,T_test);%皮尔逊相关系数
corrcoeff=a(1,2) 
error_dbnbp=error;
TY_dbnbp=TY;
save dbnbp T_test error_dbnbp TY_dbnbp
t=toc %计时结束
%%
figure
plot(T_test,'k-*')
hold on
plot(TY,'o-')
firstline = '测试阶段'; 
secondline = '实际输出与理想输出的结果对照';
title({firstline;secondline},'Fontsize',12);
xlabel('测试样本数')
ylabel('功率幅幅值')
legend('期望输出','实际输出')