%clustering on multilabel data usin Kmeans
data=csvread('scene-trainnew.csv');
size(data,2);

len=size(data,1);
ln=6;  %no of unique label
nof=size(data,2)-6 ;%no. of features
feature_train=data(:,1:nof);
nofcol=size(data,2);
label_train=data(:,nof+1:nofcol);
ntrain=len;
idx=kmeans(feature_train,12);

final=[label_train,idx];

%sorting
for x=1:len
    for y=x+1:len
        if final(y,7)<final(x,7)
            temp=final(x,:);
            final(x,:)=final(y,:);
            final(y,:)=temp;
        end
    end
end
final

            
