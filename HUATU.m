%% ��ͼ����
clear
clc
close all
%%
load bp
load dbnbp
load svm
% 1��������Ԥ��ֵ����ֵ
figure
plot(T_test,'*-')
hold on
plot(TY_dbnbp,'ko-')
hold on
plot(TY_bp,'o-')
hold on
plot(TY_svm,'o-')
legend('��ʵֵ','DBN+BP','BP','SVM')
title('������Ԥ��ֵ����ʵֵ�Ƚ�')
% 2.���
figure
plot(error_dbnbp,'k*-')
hold on
plot(error_bp,'o-')
hold on
plot(error_svm,'o-')
legend('DBN+BP','BP','SVM')
title('���������')