tic;%��ʱ��ʼ
%% ����������DBN+BPԤ�⣬�����ع�Ԥ����
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
%�������� �����ȡ9500����Ϊѵ��������ʣ��100����Ϊ��������
rand('seed',0)
n=1:1:9568;
% [m n ]=sort(rand(1,9568));
P=input(n(1:9468),:);      %ѵ������
T=output(n(1:9468));
P_test=input(n(9469:9568),:);%��������
T_test=output(n(9469:9568));
clear data m n input output
%% ѵ���������죬�ֿ飬����
numcases=263;%ÿ�����ݼ�����������
numdims=size(P,2);%���������Ĵ�С
numbatches=36;%��9500��ѵ���������ֳ�95����ÿһ��100��
% ѵ������
for i=1:numbatches
    train1=P((i-1)*numcases+1:i*numcases,:);
    batchdata(:,:,i)=train1;
end%���ֺõ�10�����ݶ�����batchdata��



%% 2.ѵ��RBM
%% rbm����
maxepoch=100;%ѵ��rbm�Ĵ���
numhid=45; numpen=44; numpen2=54; numpen3=43;%dbn������Ľڵ���
disp('����һ��4��������������DBN����������ȡ');
%% �޼ලԤѵ��
fprintf(1,'Pretraining Layer 1 with RBM: %d-%d ',numdims,numhid);
restart=1;
rbm1;%ʹ��cd-kѵ��rbm��ע���rbm�Ŀ��Ӳ㲻�Ƕ�ֵ�ģ����������Ƕ�ֵ��
vishid1=vishid;hidrecbiases=hidbiases;


fprintf(1,'\nPretraining Layer 2 with RBM: %d-%d ',numhid,numpen);
batchdata=batchposhidprobs;%����һ��RBM��������������Ϊ�ڶ���RBM ������
numhid=numpen;%��numpen��ֵ����numhid����Ϊ�ڶ���rbm������Ľڵ���
restart=1;
rbm1;
hidpen=vishid; penrecbiases=hidbiases; hidgenbiases=visbiases;

fprintf(1,'\nPretraining Layer 3 with RBM: %d-%d ',numpen,numpen2);%200-100
batchdata=batchposhidprobs;%��Ȼ�����ڶ���RBM�������Ϊ������RBM������
numhid=numpen2;%������������Ľڵ���
restart=1;
rbm1;
hidpen2=vishid; penrecbiases2=hidbiases; hidgenbiases2=visbiases;

fprintf(1,'\nPretraining Layer 4 with RBM: %d-%d\n ',numpen2,numpen3);%200-100
batchdata=batchposhidprobs;%��Ȼ�����ڶ���RBM�������Ϊ������RBM������
numhid=numpen3;%������������Ľڵ���
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
s1=100;%������ڵ�
disp('ѵ��bp������')
net=newff(inputn,outputn,s1);
net.trainParam.epochs=100;%ѵ������
net.trainParam.lr=1;%ѧϰ��
net.trainParam.goal=0.0001;%ѧϰĿ��
net.trainParam.max_fail = 200;% ��Сȷ��ʧ�ܴ��� 
net.trainParam.showWindow = false; 
net.trainParam.showCommandLine = false; 
%��������
%����ѵ��
net=train(net,inputn,outputn);
% ѵ�����
disp('����ѵ��bp������')
Y=sim(net,inputn);
  
%%%%%%%%%% ����ѵ��������Ҫ������ͼ����
% ����һ��
T=mapminmax('reverse',T,outputns);
Y=mapminmax('reverse',Y,outputns)';
error=T-Y;
bp_train_mse=mse(error);
figure
plot(error)
title('ѵ�������')
%
figure
plot(T,'k-*')
hold on
plot(Y,'o-')
legend('�������','ʵ�����')
firstline = 'ѵ���׶�'; 
secondline = 'ʵ���������������Ľ������';
title({firstline;secondline},'Fontsize',12);
xlabel('ѵ��������')

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

T_test=mapminmax('reverse',T_test,outputns);
TY=mapminmax('reverse',TY,outputns)';
%%%%%%%%%% ������Խ��
result=[TY;T_test];
error=TY-T_test;
figure
plot(error)
title('���Լ����')
fprintf('���Լ�����������\n');


MSE=mse(error)
 
MAPE=sum(abs(TY-T_test)./T_test)/length(T_test)
MAE=mean(abs(TY-mean(T_test))) 
N = length(T_test);
R2 = (N*sum(TY.*T_test)-sum(TY)*sum(T_test))^2/((N*sum((TY).^2)-(sum(TY))^2)*(N*sum((T_test).^2)-(sum(T_test))^2))

a=corrcoef(TY,T_test);%Ƥ��ѷ���ϵ��
corrcoeff=a(1,2) 
error_dbnbp=error;
TY_dbnbp=TY;
save dbnbp T_test error_dbnbp TY_dbnbp
t=toc %��ʱ����
%%
figure
plot(T_test,'k-*')
hold on
plot(TY,'o-')
firstline = '���Խ׶�'; 
secondline = 'ʵ���������������Ľ������';
title({firstline;secondline},'Fontsize',12);
xlabel('����������')
ylabel('���ʷ���ֵ')
legend('�������','ʵ�����')