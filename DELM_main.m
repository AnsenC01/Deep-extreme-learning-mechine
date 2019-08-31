%% ����������ڷ�������-
%% ����DELM


%%  ��������  ����600�飬ά��6400������ѵ����480�飬���Լ�120��
tic
close all
clear all
clc
%%

fprintf(1,'�������� \n');
load('drivface4444');%����1-173Ϊ1�࣬174-343Ϊ2�� 344-510Ϊ3�� 511-600Ϊ4�࣬��ѡ��20%��Ϊ���Լ�
%��һ��173��
[i1 i2]=sort(rand(173,1)); 
train(1:139,:)=input(i2(1:139),:);     train_label(1:139,:)=output(i2(1:139),:);
test(1:34,:)=input(i2(140:173),:);     test_label(1:34,:)=output(i2(140:173),:);
%�ڶ�����170��
[i1 i2]=sort(rand(170,1));
train(140:275,:)=input(173+i2(1:136),:);    train_label(140:275,:)=output(173+i2(1:136),:);
test(35:68,:)=input(173+i2(137:170),:);     test_label(35:68,:)=output(173+i2(137:170),:);
%��������167
[i1 i2]=sort(rand(167,1));
train(276:408,:)=input(343+i2(1:133),:);    train_label(276:408,:)=output(343+i2(1:133),:);
test(69:102,:)=input(343+i2(134:167),:);     test_label(69:102,:)=output(343+i2(134:167),:);
%��4����90
[i1 i2]=sort(rand(90,1));
train(409:480,:)=input(510+i2(1:72),:);    train_label(409:480,:)=output(510+i2(1:72),:);
test(103:120,:)=input(510+i2(73:90),:);     test_label(103:120,:)=output(510+i2(73:90),:); 
clear i1 i2 input output 
%%����˳��
k=rand(480,1);[m n]=sort(k);
train_x=train(n(1:480),:);train_y=train_label(n(1:480),:);
k=rand(120,1);[m n]=sort(k);
test_x=test(n(1:120),:);test_y=test_label(n(1:120),:);
P_train=train_x;
T_train=train_y;
P_test=test_x;  
T_test=test_y;
clear k m n train train_label test test_label train_x train_y test_x test_y
%% 
TF='sig';
h=[100;500;20];
%% ѵ��ML-ELM����ѵ�������
%��������
lambda=inf;
delm=delmtrain(P_train,T_train,h,lambda,TF);
T1=delmpredict(delm,P_train);

[I J]=max(T1,[],2);
[I1 J1]=max(T_train,[],2);
train_accuracy=sum(J==J1)/length(J)
figure
stem(J,'bo');hold on 
plot(J1,'r*');
title('ѵ����������')
legend('ʵ�����','�������')
% ���Լ�;
T2=delmpredict(delm,P_test);
[I J2]=max(T2,[],2);
[I1 J3]=max(T_test,[],2);
test_accuracy=sum(J2==J3)/length(J2)
figure
stem(J2,'bo');hold on 
plot(J3,'r*');
title('���Լ���������')
legend('ʵ�����','�������')
toc