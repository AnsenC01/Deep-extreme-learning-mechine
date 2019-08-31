%%  
%%没优化的DELM  3个隐含层
%% 清空环境
tic
close all
clear 
format compact
format short
clc
%%
in=xlsread('数据2.xlsx','input');

out=xlsread('数据2.xlsx','output');
% 归一化
input=mapminmax(in',0,1);
[output,TSps]=mapminmax(out',0,1);
input=input';
output=output';

m=2552;
P_train=input(1:m,:);      %训练输入
T_train=output(1:m,:);
P_test=input(m+1:end,:);%测试输入
T_test=output(m+1:end,:);

%% 参数设置
hidden=[10;20;500];%n个隐含层就是[n1;n2;n3;n4;....nn]
lambda=inf;
TF='sig';
%% 训练
delm=delmtrain(P_train,T_train,hidden,lambda,TF);
T1=delmpredict(delm,P_train);
% 反归一化
T11=mapminmax('reverse',T1',TSps);
TTR=mapminmax('reverse',T_train',TSps);

figure
plot(T11)
hold on
plot(TTR)
legend('实际输出','期望输出')
title('训练集')

% 测试集;
T2=delmpredict(delm,P_test);
T22=mapminmax('reverse',T2',TSps);
TTe=mapminmax('reverse',T_test',TSps);
figure
plot(T22)
hold on
plot(TTe)
legend('实际输出','期望输出')
test_mse=mse(T22-TTe)
T_test=TTe;

TY=T22;
N = length(T_test);
R2 = (N*sum(TY.*T_test)-sum(TY)*sum(T_test))^2/((N*sum((TY).^2)-(sum(TY))^2)*(N*sum((T_test).^2)-(sum(T_test))^2))
title('测试集')