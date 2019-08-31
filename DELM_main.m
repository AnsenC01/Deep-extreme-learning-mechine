%% 这个程序用于分类任务-
%% 基础DELM


%%  加载数据  数据600组，维度6400；划分训练集480组，测试集120组
tic
close all
clear all
clc
%%

fprintf(1,'加载数据 \n');
load('drivface4444');%其中1-173为1类，174-343为2类 344-510为3类 511-600为4类，各选择20%作为测试集
%第一类173组
[i1 i2]=sort(rand(173,1)); 
train(1:139,:)=input(i2(1:139),:);     train_label(1:139,:)=output(i2(1:139),:);
test(1:34,:)=input(i2(140:173),:);     test_label(1:34,:)=output(i2(140:173),:);
%第二类有170组
[i1 i2]=sort(rand(170,1));
train(140:275,:)=input(173+i2(1:136),:);    train_label(140:275,:)=output(173+i2(1:136),:);
test(35:68,:)=input(173+i2(137:170),:);     test_label(35:68,:)=output(173+i2(137:170),:);
%第三类有167
[i1 i2]=sort(rand(167,1));
train(276:408,:)=input(343+i2(1:133),:);    train_label(276:408,:)=output(343+i2(1:133),:);
test(69:102,:)=input(343+i2(134:167),:);     test_label(69:102,:)=output(343+i2(134:167),:);
%第4类有90
[i1 i2]=sort(rand(90,1));
train(409:480,:)=input(510+i2(1:72),:);    train_label(409:480,:)=output(510+i2(1:72),:);
test(103:120,:)=input(510+i2(73:90),:);     test_label(103:120,:)=output(510+i2(73:90),:); 
clear i1 i2 input output 
%%打乱顺序
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
%% 训练ML-ELM，看训练集误差
%加载数据
lambda=inf;
delm=delmtrain(P_train,T_train,h,lambda,TF);
T1=delmpredict(delm,P_train);

[I J]=max(T1,[],2);
[I1 J1]=max(T_train,[],2);
train_accuracy=sum(J==J1)/length(J)
figure
stem(J,'bo');hold on 
plot(J1,'r*');
title('训练集分类结果')
legend('实际输出','期望输出')
% 测试集;
T2=delmpredict(delm,P_test);
[I J2]=max(T2,[],2);
[I1 J3]=max(T_test,[],2);
test_accuracy=sum(J2==J3)/length(J2)
figure
stem(J2,'bo');hold on 
plot(J3,'r*');
title('测试集集分类结果')
legend('实际输出','期望输出')
toc