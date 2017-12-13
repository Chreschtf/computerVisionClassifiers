setDir  = fullfile('./training');
imds = imageDatastore(setDir,'IncludeSubfolders',true,'LabelSource','foldernames');
[trainingSet,testSet] = splitEachLabel(imds,0.3,'randomize');
%%
bag = bagOfFeatures(trainingSet,'Gridstep',[4 4],'BlockWidth',[32]);%,'Verbose',false);   
%%
categoryClassifier = trainImageCategoryClassifier(trainingSet,bag);
%%
confMatrix = evaluate(categoryClassifier,testSet);
%%
testDir=fullfile('./testing');
img = imread(fullfile(testDir,'2.jpg'));
[labelIdx, score] = predict(categoryClassifier,img);
categoryClassifier.Labels(labelIdx);

%%
testDir=fullfile('./testing');
%exts = {'.jpg','.png','.tif'};
%testImds = imageDatastore(testDir,'FileExtensions',exts);
testImages=dir(testDir);
testImagesCell=struct2cell(testImages);
imageNames=testImagesCell(1,:);
natSortedImages=natsortfiles(imageNames)';
%%
imagePredictions = strings(length(natSortedImages),2);
for i=3:length(natSortedImages)
    %1 and 2  = . and ..
    testImageName = char(natSortedImages(i));%.name;
    img= imread(fullfile(testDir,testImageName));
    [labelIdx, score] = predict(categoryClassifier,img);
    assignedLabel = categoryClassifier.Labels(labelIdx);
    
    imagePredictions(i-2,2)=assignedLabel;
    imagePredictions(i-2,1)=testImageName;
end

%%

fileID = fopen('run2.txt','w');
for i=1:length(imagePredictions)
    fprintf(fileID,'%s %s\n',imagePredictions(i,1),imagePredictions(i,2));
end
fclose(fileID);