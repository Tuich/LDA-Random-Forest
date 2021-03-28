function [meanData, projMat] = lda_core(DATA, labelVec, subSpaceDim)
%   REFERENCE:
%       [1] PN Belhumeur, JP Hespanha, 
%       'Eigenfaces vs. fisherfaces: Recognition using class specific linear 
%       projection', PAMI 1998.
%
%   INPUT:
%       DATA:              the input high-dimensional data;
%       labelVec:          the vector that indicates which subject each 
%                          column of DATA belongs to;
%       subSpaceDim:       the subspace dimension learnt by LDA.
%
%   OUTPUT:
%       meanData:          the mean data of the input matrix DATA;
%       projMat:           the learnt lda projection matrix.
%
%   VERSION:
%       0.1 - 18/10/2012  First implementation,
%       Matlab 2012a


% obtain the labels of subjects
subjectVec = unique(labelVec);

%% Calculate Sw and Sb

Sw = zeros(size(DATA, 1), 'double');  % within class scatter matrix
Sb = zeros(size(DATA, 1), 'double');  % between class scatter matrix
meanData=mean(DATA,2);
for i=1:length(subjectVec)
    %% Task 2: Try to obtain Sw and Sb by yourself (on page62 of the Machine Learning textbook)
    label=subjectVec(i);
    idx=find(labelVec==label);
    n_sub=length(idx);
    DATA_sub=DATA(:,idx);
    mf_sub=mean(DATA_sub,2);
    Sb=Sb+n_sub*(mf_sub-meanData)*(mf_sub-meanData)';
    tmp_mat=DATA_sub-repmat(mf_sub,1,n_sub);
    Sw=Sw+tmp_mat*tmp_mat';
end

%% Calculate LDA projection matrix
% different from the eig function, eigs would not normalize the eigen
% vectors automatically
[projMat, ~] = eigs(Sb, Sw, subSpaceDim, 'LM');

% normalization
projMat = projMat./repmat(sqrt(sum(projMat.*projMat)), size(projMat,1), 1);

