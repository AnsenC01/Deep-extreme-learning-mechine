function delm=delmtrain(P_train,T_train,hidden,lambda,TF)
disp('ELM-AEÎÞ¼à¶½ÑµÁ·')
% ELM-AEÎÞ¼à¶½ÑµÁ·
input_num=size(P_train,1);
for u = 1 : numel(hidden)
    
    fprintf(1,'Pretraining Layer %d with ELM-AE: %d-%d \n',u,input_num,hidden(u));

    weight = ELMAEtrain(hidden(u),P_train,lambda,TF);
    delm.elmae{u}=pinv(weight);
    
    input_num=hidden(u);
    P_train=P_train*delm.elmae{u};

end


% ·ÖÀà»ò»Ø¹é²ãµÄ¼à¶½ÑµÁ·
disp('·ÖÀà»Ø¹é²ãµÄ¼à¶½ÑµÁ·')

weight=top_ELMtrain(P_train,T_train,lambda);
delm.output=weight;


