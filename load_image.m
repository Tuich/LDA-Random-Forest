%% Load the Training Data
trainDataRoute = '.\data\train\';
l_p = list_files([trainDataRoute 'positive']);
l_n = list_files([trainDataRoute 'negative']);
TrainData = zeros(100, 100, length(l_p)+length(l_n), 'uint8');
trainLabelVec = zeros(length(l_p)+length(l_n), 1);

% read in positive training samples
for i = 1:length(l_p)
    im = imread([trainDataRoute 'positive\' l_p{i}]);
    im = rgb2gray(im);
    TrainData(:,:,i) = imresize(im, [100 100]);
    trainLabelVec(i) = 1;
end
% read in negative training samples
for i = 1:length(l_n)
    im = imread([trainDataRoute 'negative\' l_n{i}]);
    im = rgb2gray(im);
    TrainData(:,:,length(l_p)+i) = imresize(im, [100 100]);
    trainLabelVec(length(l_p)+i) = 0;
end
save TrainDATA.mat TrainData trainLabelVec

%% Load the Testing Data
TestDataRoute = '.\data\test\';
l_p = list_files([TestDataRoute 'positive']);
l_n = list_files([TestDataRoute 'negative']);
TestData = zeros(100, 100, length(l_p)+length(l_n), 'uint8');
testLabelVec = zeros(length(l_p)+length(l_n), 1);

% read in positive training samples
for i = 1:length(l_p)
    im = imread([TestDataRoute 'positive\' l_p{i}]);
    im = rgb2gray(im);
    TestData(:,:,i) = imresize(im, [100 100]);
    testLabelVec(i) = 1;
end
% read in negative training samples
for i = 1:length(l_n)
    im = imread([TestDataRoute 'negative\' l_n{i}]);
    im = rgb2gray(im);
    TestData(:,:,length(l_p)+i) = imresize(im, [100 100]);
    testLabelVec(length(l_p)+i) = 0;
end
save TestDATA.mat TestData testLabelVec

