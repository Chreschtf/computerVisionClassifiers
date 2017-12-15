trainingDir  = fullfile('./training');
images = imageDatastore(trainingDir,'IncludeSubfolders',true,'LabelSource','foldernames');
images.ReadFcn = @(loc)imresize(cat(3,imread(loc),imread(loc),imread(loc)),[227,227]);
[trainingSet,validationSet] = splitEachLabel(images,.8,'randomized');



%%
net = alexnet;

%%
layer = 'fc7';
trainingFeatures = activations(net,trainingSet,layer);
validationFeatures = activations(net,validationSet,layer);
trainingLabels = trainingSet.Labels;
testLabels = validationSet.Labels;
classifier = fitcecoc(trainingFeatures,trainingLabels);
predictedLabels = predict(classifier,validationFeatures);

%%
accuracy = mean(predictedLabels == testLabels)


%%
% testDir=fullfile('./testing');
% %exts = {'.jpg','.png','.tif'};
% %testImds = imageDatastore(testDir,'FileExtensions',exts);
% testImages=dir(testDir);
% testImagesCell=struct2cell(testImages);
% imageNames=testImagesCell(1,:);
% addpath('./3rdPartyPackages/natsort');
% natSortedImages=natsortfiles(imageNames)';
% %%
% imagePredictions = strings(length(natSortedImages),2);
% for i=3:length(natSortedImages)
%     %1 and 2  = . and ..
%     testImageName = char(natSortedImages(i));%.name;
%     img= imread(fullfile(testDir,testImageName));
%     img = imresize(cat(3,img,img,img),[227,227]);
%     imgFeature = activations(net,img,layer);
%     assignedLabel = predict(classifier,imgFeature);
%     
%     imagePredictions(i-2,2)=assignedLabel;
%     imagePredictions(i-2,1)=testImageName;
% end
% 
% %%
% 
% fileID = fopen('run3.txt','w');
% for i=1:length(imagePredictions)
%     fprintf(fileID,'%s %s\n',imagePredictions(i,1),imagePredictions(i,2));
% end
% fclose(fileID);