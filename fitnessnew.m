function y= fitnessnew(xx)
load shuju
maxepoch=2;%ѵ��rbm�Ĵ���
%% ѵ����1��RBM
numhid=xx(1,1);
restart=1;
rbm1;%ʹ��cd-kѵ��rbm��ע���rbm�Ŀ��Ӳ㲻�Ƕ�ֵ�ģ����������Ƕ�ֵ��
vishid1=vishid;hidrecbiases=hidbiases;
%% ѵ����2��RBM

batchdata=batchposhidprobs;%����һ��RBM��������������Ϊ�ڶ���RBM ������
numhid=xx(1,2);%��numpen��ֵ����numhid����Ϊ�ڶ���rbm������Ľڵ���
restart=1;
rbm1;
hidpen=vishid; penrecbiases=hidbiases; hidgenbiases=visbiases;
%% ѵ����3��RBM

batchdata=batchposhidprobs;%��Ȼ�����ڶ���RBM�������Ϊ������RBM������
numhid=xx(1,3);%������������Ľڵ���
restart=1;
rbm1;
hidpen2=vishid; penrecbiases2=hidbiases; hidgenbiases2=visbiases;
%% ѵ����4��RBM

batchdata=batchposhidprobs;%��Ȼ�����ڶ���RBM�������Ϊ������RBM������
numhid=xx(1,4);%������������Ľڵ���
restart=1;
rbm1;
hidpen3=vishid; penrecbiases3=hidbiases; hidgenbiases3=visbiases;

%%%% ��Ԥѵ���õ�RBM���ڳ�ʼ��DBNȨ��%%%%%%%%%
w1=[vishid1; hidrecbiases]; 
w2=[hidpen; penrecbiases]; 
w3=[hidpen2; penrecbiases2];
w4=[hidpen3; penrecbiases3];

%% �мල�ع��ѵ��
%===========================ѵ������=====================================%
%==========DBN�޼ල������ȡ��������Ҫ�����мල�Ļع��==================%
%���ں���ƫִ������ʵ������Ӧ�ð���һ��ȫΪ1��������w0x0+w1x1+..+wnxn ����x0Ϊ1������ w0Ϊƫ��b
N1 = size(P,1);
digitdata = [P ones(N1,1)];

w1probs = 1./(1 + exp(-digitdata*w1));
w1probs = [w1probs  ones(N1,1)];

w2probs = 1./(1 + exp(-w1probs*w2));
w2probs = [w2probs ones(N1,1)];

w3probs = 1./(1 + exp(-w2probs*w3)); 
w3probs = [w3probs ones(N1,1)];

w4probs = 1./(1 + exp(-w3probs*w4)); 
H = w4probs'; %DBN�����  Ҳ��BP������


%% BP����ѵ��
inputn=H;
outputn=T';
% %��ʼ������ṹ
s1=10;%������ڵ�
net=newff(inputn,outputn,s1);
net.trainParam.epochs=10;%ѵ������
net.trainParam.lr=1;%ѧϰ��
net.trainParam.goal=0.0001;%ѧϰĿ��
net.trainParam.max_fail = 200;% ��Сȷ��ʧ�ܴ��� 
net.trainParam.showWindow = false; 
net.trainParam.showCommandLine = false; 
%��������
%����ѵ��
net=train(net,inputn,outputn);
%%
%===========================���Թ���=====================================%
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
H = w4probs'; %DBN�����  Ҳ��BP������

TY=sim(net,H);
% ����һ��

%%%%%%%%%% ������Խ��

error=TY'-T_test;

% fprintf('�������MSE\n');
y=1/mse(error);



%%

end

