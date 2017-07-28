dat=csvread('iris.data.csv');
data=autoencoderelm(dat);

len=size(data,1);
%data
 %ix = randperm(len);
 %data = data(ix,:); %t shuffle the data
 %data
 ln=3;  %no of unique label
 nof=size(data,2)-1;%no. of features
features=data(:,1:nof);
label=data(:,nof+1);
 labels=zeros(len,ln);
 for x=1:len
     labels(x,label(x))=1;
 end
ntrain=100   %no. of training eg.
ntest=len-ntrain %no. of test eg.
feature_train=features(1:ntrain,:);
label_train=labels(1:ntrain,:);
feature_test=features(ntrain+1:len,:);
label_test=labels(ntrain+1:len,:);

L=35;  %no. of nodes

w=rand(L,nof);  %random weight matrix

b=rand(L,1);
h=[];
for x=1:ntrain
    Y=w*feature_train(x,:)'+b;
    Y=sigmf(Y,[1 0]);
    h(:,x)=Y;
end
h=h';
h;
beta=pinv(h)*label_train;

feature_test(1,:);

htest=[];

for x=1:ntest
    Y=w*feature_test(x,:)'+b;
    Y=sigmf(Y,[1 0]);
    htest(:,x)=Y;
end
htest=htest';

final=htest*beta;
for x=1:ntest
    ma=max(final(x,:));
    for j=1:ln
        if(final(x,j)==ma)
            final(x,j)=1;
        else
            final(x,j)=0;
        end
    end
end
final
c=0;
for i=1:ntest
    if(final(i,:)==label_test(i,:))
        c=c+1;
    end
end
accu=c/ntest



        
        
    
    




  







