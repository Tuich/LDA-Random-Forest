function [LowData, projMat, meanData] = pca(DATA, para)
%% DESCRIPTION
%       This program is to calculate the PCA projection matrix.
%
%   REFERENCE:
%       [1] M Turk & A Pentland, 'Eigenfaces for recognition',
%       Journal of cognitive neuroscience 1991.
%
%   INPUT:
%       DATA:           the matrix whose columns stands for observations;
%       para:           struct containing parameters for PCA dimension reduction.
%
%   OUTPUT:
%       LowData:        the low-dimensional Data after PCA projection;
%       projMat:        projection matrix;
%       meanData:       the mean of all observations in DATA;
%
%   VERSION:
%       0.1 - 18/10/2012  First implementation,
%       Matlab 2012a


        
%% Calculate the mean face and reshape it
DATA = double(DATA);
meanData = mean(DATA, 2);
% subtract the mean
DATA = DATA - repmat(meanData, 1, size(DATA, 2));


%% Calculate the principal eigenvectors
% calcuate the scatter matrix (applying a trick if the number of samples is less
% than the dimension of the image vectors)
if size(DATA, 1) > size(DATA, 2)
    %% Task 1: Try to write this piece of code (the trick on page570 in Pattern.Recognition.and.Machine.Learning)
    %task1
    scatterMat=DATA'*DATA;
    [V,D]=eig(scatterMat);
    trans=DATA*V;
    l2=(sum(trans.^2)).^0.5;
    %a=l2./size(DATA,2);就是D且wi模为1
    %[~,IX] = sort(a,'descend'); 
    [~,IX] = sort(diag(D),'descend');
    projMat=trans./repmat(l2,size(DATA,1),1);
    projMat=projMat(:,IX(1:para.pcaDim));
    %%
else
    scatterMat = DATA*DATA';
    [V, D] = eig(scatterMat);
    [~,IX] = sort(diag(D),'descend');   % get the label of the maximum eigenvalue
    projMat = V(:,IX(1:para.pcaDim));   % get the eigenvector corresponding to the top eigenvalues
    projMat = normc(projMat);
end
LowData = projMat'*DATA;  % dimension reduction






