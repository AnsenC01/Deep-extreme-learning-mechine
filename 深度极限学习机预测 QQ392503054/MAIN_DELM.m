%%  
%%û�Ż���DELM  3��������
%% ��ջ���
tic
close all
clear 
format compact
format short
clc
%%
in=xlsread('����2.xlsx','input');

out=xlsread('����2.xlsx','output');
% ��һ��
input=mapminmax(in',0,1);
[output,TSps]=mapminmax(out',0,1);
input=input';
output=output';

m=2552;
P_train=input(1:m,:);      %ѵ������
T_train=output(1:m,:);
P_test=input(m+1:end,:);%��������
T_test=output(m+1:end,:);

%% ��������
hidden=[10;20;500];%n�����������[n1;n2;n3;n4;....nn]
lambda=inf;
TF='sig';
%% ѵ��
delm=delmtrain(P_train,T_train,hidden,lambda,TF);
T1=delmpredict(delm,P_train);
% ����һ��
T11=mapminmax('reverse',T1',TSps);
TTR=mapminmax('reverse',T_train',TSps);

figure
plot(T11)
hold on
plot(TTR)
legend('ʵ�����','�������')
title('ѵ����')

% ���Լ�;
T2=delmpredict(delm,P_test);
T22=mapminmax('reverse',T2',TSps);
TTe=mapminmax('reverse',T_test',TSps);
figure
plot(T22)
hold on
plot(TTe)
legend('ʵ�����','�������')
test_mse=mse(T22-TTe)
T_test=TTe;

TY=T22;
N = length(T_test);
R2 = (N*sum(TY.*T_test)-sum(TY)*sum(T_test))^2/((N*sum((TY).^2)-(sum(TY))^2)*(N*sum((T_test).^2)-(sum(T_test))^2))
title('���Լ�')