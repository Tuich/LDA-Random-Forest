function [LowData, projMat, pcaProj, meanData] = lda(HiData, labelVec, para)
%% DESCRIPTION
%       The function is to do dimension reduction for the HiData by LDA.
%
%   INPUT:  
%       HiData:           the data matrix whose each column stands for one
%                         image data;
%       labelVec:         the subject label for each image in HiData;
%       para:             the parameters required for the dimension
%                         reduction method.
%
%   OUTPUT:
%       LowData:          the low-dimensional Data after LDA projection;
%       projMat:          the projection matrix for dimension reduction;
%       meanData:         the mean vector of the vectors in HiData.
%
%   VERSION:
%       0.1 - 01/01/2013  First implementation,
%       Matlab 2012a



%% PCA pre-processing
[pcaHiData, pcaProj, meanData] = pca(HiData, para);          

%% Apply LDA
[~, ldaProj] = lda_core(pcaHiData, labelVec, para.ldaDim);
projMat = pcaProj*ldaProj;
LowData = ldaProj'*pcaHiData;
end