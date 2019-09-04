function y= fitnessnew(xx)
load shuju
maxepoch=2;%训练rbm的次数
%% 训练第1层RBM
numhid=xx(1,1);
restart=1;
rbm1;%使用cd-k训练rbm，注意此rbm的可视层不是二值的，而隐含层是二值的
vishid1=vishid;hidrecbiases=hidbiases;
%% 训练第2层RBM

batchdata=batchposhidprobs;%将第一个RBM的隐含层的输出作为第二个RBM 的输入
numhid=xx(1,2);%将numpen的值赋给numhid，作为第二个rbm隐含层的节点数
restart=1;
rbm1;
hidpen=vishid; penrecbiases=hidbiases; hidgenbiases=visbiases;
%% 训练第3层RBM

batchdata=batchposhidprobs;%显然，将第二个RBM的输出作为第三个RBM的输入
numhid=xx(1,3);%第三个隐含层的节点数
restart=1;
rbm1;
hidpen2=vishid; penrecbiases2=hidbiases; hidgenbiases2=visbiases;
%% 训练第4层RBM

batchdata=batchposhidprobs;%显然，将第二哥RBM的输出作为第三个RBM的输入
numhid=xx(1,4);%第三个隐含层的节点数
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
s1=10;%隐含层节点
net=newff(inputn,outputn,s1);
net.trainParam.epochs=10;%训练次数
net.trainParam.lr=1;%学习率
net.trainParam.goal=0.0001;%学习目标
net.trainParam.max_fail = 200;% 最小确认失败次数 
net.trainParam.showWindow = false; 
net.trainParam.showCommandLine = false; 
%建立网络
%网络训练
net=train(net,inputn,outputn);
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

%%%%%%%%%% 计算测试结果

error=TY'-T_test;

% fprintf('均方误差MSE\n');
y=1/mse(error);



%%

end

