function delm=delmtrain(P_train,T_train,hidden,lambda,TF)
disp('ELM-AE�޼ලѵ��')
% ELM-AE�޼ලѵ��
input_num=size(P_train,2);
for u = 1 : numel(hidden)
    
    fprintf(1,'Pretraining Layer %d with ELM-AE: %d-%d \n',u,input_num,hidden(u));

    weight = ELMAEtrain(hidden(u),P_train,lambda,TF);
    
    delm.elmae{u}=pinv(weight);
    
    input_num=hidden(u);
    P_train=P_train*delm.elmae{u};

end


% �����ع��ļලѵ��
disp('����ع��ļලѵ��')

weight=top_ELMtrain(P_train,T_train,lambda);
delm.output=weight;


