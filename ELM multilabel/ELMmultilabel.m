%multilabel classification using ELM
data=csvread('scene-trainnew.csv');%change data to dat1 for autoencoder
%data=multiautoencoderelm(dat1);%uncomment to use autoencoder
size(data,2);

len=size(data,1);
% %data
%  %ix = randperm(len);
%  %data = data(ix,:); %t shuffle the data
%  %data
  ln=6;  %no of unique label
 nof=size(data,2)-6 ;%no. of features
feature_train=data(:,1:nof);
%size(features);
nofcol=size(data,2);
label_train=data(:,nof+1:nofcol);

 ntrain=len;   %no. of training eg.
% ntest=len-ntrain %no. of test eg.
% feature_train=features(1:ntrain,:);
% label_train=labels(1:ntrain,:);
% feature_test=features(ntrain+1:len,:);
% label_test=labels(ntrain+1:len,:);
% 
 L=30;  %no. of nodes
% 
w=rand(L,nof)/100;  %random weight matrix
% 
b=rand(L,1)/100;
h=[];
for x=1:ntrain
   Y=w*feature_train(x,:)'+b;
   
   Y=sigmf(Y,[1 0]);
  
   h(:,x)=Y;
end
 h=h';
 h
 beta=pinv(h)*label_train;
 
 label_got=h*beta;
 size(label_got);
 
 
 
 
 
 datatest=csvread('scene-testnew.csv');%change datatest to datatest1 for autoencoder
% datatest=multiautoencoderelm(datatest1);%uncomment to use autoencoder
 ntest=size(datatest,1);
 
 feature_test=datatest(:,1:nof);
 nofcol2=size(datatest,2);
 label_test=datatest(:,nof+1:nofcol2);
 htest=[];
 
 for x=1:ntest
     Y=w*feature_test(x,:)'+b;
     Y=sigmf(Y,[1 0]);
     htest(:,x)=Y;
 end
 htest=htest';
 
 final=htest*beta;
 final1=MLPupdate(label_got,label_train,final);
 final1=round(final1')
%  for x=1:ntest
%     ma=mean(final1(x,:));
%      for j=1:ln
%         if(final1(x,j)>ma)
%            final1(x,j)=1;
%         else
%             final1(x,j)=0;
%        end
%     end
%  end
 
 

c=0;
for i=1:ntest
    s=0;
    for j=1:ln
        s=s+xor(final1(i,j),label_test(i,j));
    end
    c=c+(s/ln);
end
hloss=c/ntest
% for i=1:ntest
%     if(final(i,:)==label_test(i,:))
%         c=c+1;
%     end
%  end
 