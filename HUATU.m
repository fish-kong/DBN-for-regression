%% 画图分析
clear
clc
close all
%%
load bp
load dbnbp
load svm
% 1，各方法预测值与真值
figure
plot(T_test,'*-')
hold on
plot(TY_dbnbp,'ko-')
hold on
plot(TY_bp,'o-')
hold on
plot(TY_svm,'o-')
legend('真实值','DBN+BP','BP','SVM')
title('各方法预测值与真实值比较')
% 2.误差
figure
plot(error_dbnbp,'k*-')
hold on
plot(error_bp,'o-')
hold on
plot(error_svm,'o-')
legend('DBN+BP','BP','SVM')
title('各方法误差')