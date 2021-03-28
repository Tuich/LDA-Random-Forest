%% Load the Training Data
fprintf('Loading the Training Set:\n\n');
% if you want to know how to load training images one by one, read 'load_image.m'

% or you can directly load .mat file
% TrainData denotes training data; 
% trainLabelVec includes the label of each image.
load TrainDATA.mat;   
train2=fliplr(TrainData);
TrainData=cat(3,TrainData,train2);
trainLabelVec=repmat(trainLabelVec,2,1);

%% IMPORTANT: TUNE THE FOLLOWING TWO PARAMETERS BY YOURSELF
% try to specify a PCA dimension by yourself
para.pcaDim = 130;   
% you also need to specify a threshold for the LDA classifier
threshold = 360;
hpara=[];
score=[];
%for threshold=300:30:810
    %for tmp=60:10:200
       %para.pcaDim = tmp; 
        %% Train an LDA Classifier
fprintf('Learning the LDA Subspace:\n\n');
% specify the dimension of the LDA subspace
para.ldaDim = 1;
% reshape 2D image data to 1D vector
HiData = reshape(TrainData, size(TrainData,1)*size(TrainData,2), size(TrainData,3));
[LowData, w, pcaProj, meanData] = lda(HiData, trainLabelVec, para);


%% Make Predictions
% TestData denotes training data; 
% testLabelVec includes the label of each image.
load TestDATA.mat;
HiData = reshape(TestData, size(TestData,1)*size(TestData,2), size(TestData,3));
% apply the LDA classifier
output = w'*(double(HiData)-repmat(meanData,1,size(HiData,2)));


prediction = output>threshold;
% find the data that have been correctly classified
[a, ~] = find(prediction==testLabelVec');
% calculate face recognition accuracy
accuracy01 = length(a)/length(testLabelVec);

% in case that the direction of the classifier is reversed
prediction = output<-threshold;
[a, ~] = find(prediction==testLabelVec');
accuracy02 = length(a)/length(testLabelVec);

hpara=[hpara;[threshold,para.pcaDim]];
score=[score;max(accuracy01, accuracy02)];
    %end
%end;

[acc,idx]=max(score);
bestpara=hpara(idx,:);
para.ldaDim = 1;
para.pcaDim=bestpara(1,2);
threshold=bestpara(1,1);
% reshape 2D image data to 1D vector
HiData = reshape(TrainData, size(TrainData,1)*size(TrainData,2), size(TrainData,3));
[LowData, w, pcaProj, meanData] = lda(HiData, trainLabelVec, para);
HiData = reshape(TestData, size(TestData,1)*size(TestData,2), size(TestData,3));
output = w'*(double(HiData)-repmat(meanData,1,size(HiData,2)));
prediction = output>threshold;
% find the data that have been correctly classified
[a, ~] = find(prediction==testLabelVec');
% calculate face recognition accuracy
accuracy01 = length(a)/length(testLabelVec);

% in case that the direction of the classifier is reversed
prediction = [prediction;output<-threshold];
[a, ~] = find(prediction(2,:)==testLabelVec');
accuracy02 = length(a)/length(testLabelVec);

if accuracy02>accuracy01
    prediction=prediction(2,:);
    lda_score=(output+threshold)./(threshold);
    fore=-1;
else
    prediction=prediction(1,:);
    lda_score=(output-threshold)./(threshold);
    fore=1;
end;

% find the data that have been correctly classified
[~, bidx] = find(prediction~=testLabelVec');
%% Task #3: try to draw eigenfaces by yourself
num_eigenfaces_show=9;
for i = 1:num_eigenfaces_show
	subplot(3, 3, i)
    eigenface=40*reshape(pcaProj(:,i),[100,100]);
	imshow(eigenface);
	title(['Eigenfaces ' num2str(i)]);
    
end


%% Extensions
% try to promote the face recognition accrucy by any ideas. 
% (except for changing the training data) 
HiData2 = reshape(TrainData, size(TrainData,1)*size(TrainData,2), size(TrainData,3));
svm=fitcsvm(double(HiData2)',trainLabelVec','Standardize',true);
[svm_pred,svm_score]=predict(svm,double(HiData)');
[a, ~] = find(svm_pred==testLabelVec);
acc_svm = length(a)/length(testLabelVec);
[~, svmbidx] = find(svm_pred'~=testLabelVec');

min1=std(lda_score);
min2=std(svm_score(:,2));

score_all=lda_score'+0.3*fore*svm_score(:,2);
pred_all=(fore*score_all)>0;
[a, ~] = find(pred_all==testLabelVec);
acc_both=length(a)/length(testLabelVec);
[~, svmcidx] = find(pred_all'~=testLabelVec');

wrongs=union(bidx,svmbidx);
both_wrong=intersect(bidx,svmbidx);
bw_score=zeros(2,length(both_wrong));
bw_score(1,:)=lda_score(both_wrong);
bw_score(2,:)=svm_score(both_wrong);
aw_score=zeros(2,length(wrongs));
aw_score(1,:)=lda_score(wrongs);
aw_score(2,:)=svm_score(wrongs);

