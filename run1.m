% This Section of program, contains the code to implement the RUN1 of the
% coursework. These following tasks are carried out 1. Image Cropping about
% the center 2. Resize the image to a fixed resolution 16x16 pixels 3. The
% resultant image are taken as a vector and concatenated into Row 4.
% Scaling to unit length 5. Implementing KNN Classification
% 
clc;
clear all;
%Loading of Image dataset into matlab
trainImg=imageSet('.\training','recursive');
testImg=imageSet('.\testing','recursive');
%Section where the variables are initialised and 
%to implement Image cropping and resizing are done to extract tiny
%features
imgCell={trainImg};
imgCellLen=length(trainImg);
totalImginEachfolder=trainImg.Count;
resizeImageDim=16;
imgVector1=zeros(100,resizeImageDim*resizeImageDim);
tinyFeature=double.empty;
for ii=1:imgCellLen
    for jj=1:totalImginEachfolder
        imgRead=read(trainImg(ii),jj);
        ig=imgRead;
        [xDim,yDim]=size(imgRead);
        if xDim~=yDim
            squareDim=min(xDim,yDim);
            x_topLeftCorner=xDim-squareDim-floor((xDim-squareDim)/2);
            y_topLeftCorner=yDim-squareDim-floor((yDim-squareDim)/2);
            imgRead=imcrop(imgRead, [y_topLeftCorner,x_topLeftCorner,squareDim,squareDim]);
        end
        imgResize=imresize(imgRead,[resizeImageDim resizeImageDim]);
        imgResize=im2double(imgResize);
        imgVector = reshape(imgResize,[resizeImageDim*resizeImageDim,1]);
        imgVector1(jj,:) = imgVector;
    end
    tinyFeature=vertcat(tinyFeature,imgVector1);
end
%%
%normalization of the the vector to unit length
meanX=mean(tinyFeature);
stdX=std(tinyFeature);
normX=norm(tinyFeature);
unitLen=(tinyFeature)/normX;
%%
% Creating the X and Y Parameters where the X represents the Normalized
% unit vectors of size [1500x256] Y represents the Labels of the classes
% that are used to classify the images from X. It is of the size [1500x1]
% which means 15 classes for 100 images from each classes.
n=100;
X=unitLen;
classLabel={'bedroom','coast','Forest','HighW','industrial','Insidecity','Kitchen','LivingR','Mountain','Office','OpenCont','Store','Street','Suburb','TallB'}';
Y=[[repmat(classLabel(1),n,1);repmat(classLabel(2),n,1);repmat(classLabel(3),n,1);
    repmat(classLabel(4),n,1);repmat(classLabel(5),n,1);repmat(classLabel(6),n,1);
    repmat(classLabel(7),n,1);repmat(classLabel(8),n,1);repmat(classLabel(9),n,1);
    repmat(classLabel(10),n,1);repmat(classLabel(11),n,1);repmat(classLabel(12),n,1);
    repmat(classLabel(13),n,1);repmat(classLabel(14),n,1);repmat(classLabel(15),n,1)]];


%% Evaluating the Performance of KNN with Validation Set
% This section is Just to check the performance of the KNN. On this
% following section the images are randomised and later split into training
% and validation set. The training set and validation set are taken in the
% ratio of 80% and 20% These values are used in the function fitcknn to do
% the KNN classification where the X,Y and Number of Neighbours are given
% as input parmeters.

%%On the Training Images
[N, p1] = size(X);

ii = randperm(N);
%%Training set----> 80%
imageTrain = X(ii(1:(N*8/10)),:);
featureTrain = Y(ii(1:(N*8/10)),:);
%%Validation set-------> 20%
imageValidation=X(ii(N*8/10+1:N),:);
featureValidation=Y(ii(N*8/10+1:N),:);

%%Knn Classifier
Mdltrx1 = fitcknn(imageTrain,featureTrain,'NumNeighbors',10);

%%Output Prediction in Validation Set
labelvalidation = predict(Mdltrx1, imageValidation);
% The classifier predicts incorrectly classified form training data Loss of
% the KNN Claasifier
L = loss(Mdltrx1,imageTrain,featureTrain)

% To evaluate the performance of the trainined information on the
% Validation set. As the information in the array are matrix string 
% comparison is used which gives a logical output for the matching 
% Labels on the validation set. By taking the sum of all true values 
% and the total number of data accuracy of the KNN classifiaction
% is predicted.
lengthLabelValidation=length(labelvalidation);
stringComparison=strcmp(featureValidation,labelvalidation);
totalMatch=sum(stringComparison);
accuracyValidation=totalMatch/lengthLabelValidation
%%
%Loading of Image dataset into matlab
testImg=imageSet('.\testing','recursive');
%Section where the variables are initialised and 
%to implement Image cropping and resizing are done to extract tiny
%features
imgTCell={testImg};
imgTCellLen=length(testImg);
totalImginEachfolderT=testImg.Count;
resizeImageDim=16;
imgVector2=zeros(100,resizeImageDim*resizeImageDim);
tinyFeatureT=double.empty;
for ii=1:imgTCellLen
    for jj=1:totalImginEachfolderT
        imgReadT=read(testImg(ii),jj);
        igT=imgReadT;
        [xDimT,yDimT]=size(imgReadT);
        if xDimT~=yDimT
            squareDimT=min(xDimT,yDimT);
            x_topLeftCornerT=xDimT-squareDimT-floor((xDimT-squareDimT)/2);
            y_topLeftCornerT=yDimT-squareDimT-floor((yDimT-squareDimT)/2);
            imgReadT=imcrop(imgReadT, [y_topLeftCornerT,x_topLeftCornerT,squareDimT,squareDimT]);
        end
        imgResizeT=imresize(imgReadT,[resizeImageDim resizeImageDim]);
        imgResizeT=im2double(imgResizeT);
        imgVectorT = reshape(imgResizeT,[resizeImageDim*resizeImageDim,1]);
        imgVector2(jj,:) = imgVectorT;
    end
    tinyFeatureT=vertcat(tinyFeatureT,imgVector2);
end
%%
%normalization of the the vector to unit length
meanTX=mean(tinyFeatureT);
stdTX=std(tinyFeatureT);
normTX=norm(tinyFeatureT);
unitLenT=(tinyFeatureT)/normTX;

%%Knn Classifier
Mdltrx1 = fitcknn(imageTrain,featureTrain,'NumNeighbors',10);

TestImagesKNN=unitLenT;
%%Output Prediction on test Images
labelTst = predict(Mdltrx1, TestImagesKNN);


%%
testDir=fullfile('./testing');
%exts = {'.jpg','.png','.tif'};
%testImds = imageDatastore(testDir,'FileExtensions',exts);
testImages=dir(testDir);
testImagesCell=struct2cell(testImages);
imageNames=testImagesCell(1,:);
addpath('./3rdPartyPackages/natsort');
natSortedImages=natsortfiles(imageNames)';

%%
fileID = fopen('run1.txt','w');

for i=3:totalImginEachfolderT
    fprintf(fileID,'%s %s\n',natSortedImages{i+0,:},labelTst{i,:});
end
fclose(fileID);

