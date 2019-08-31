function output=delmpredict(delm,P)

num_hidden=numel(delm.elmae);
for i=1:num_hidden
    P=P*delm.elmae{i};
end

output=P*delm.output;

