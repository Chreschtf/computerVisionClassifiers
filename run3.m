trainingDir  = fullfile('./training');
images = imageDatastore(trainingDir,'IncludeSubfolders',true,'LabelSource','foldernames');
addpath('./3rdPartyPackages/grs2rgb');
images.ReadFcn = @(loc)imresize(cat(3,imread(loc),imread(loc),imread(loc)),[227,227]);
[trainingSet,validationSet] = splitEachLabel(images,0.7,'randomized');

%%
numTrainImages = numel(trainingSet.Labels);
idx = randperm(numTrainImages,16);
figure
for i = 1:16
    subplot(4,4,i)
    I = readimage(trainingSet,idx(i));
    imshow(I)
end
%%
net = alexnet;

%%
layersTransfer = net.Layers(1:end-3);
numClasses = numel(categories(trainingSet.Labels));
layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];
%%
miniBatchSize = 10;
numIterationsPerEpoch = floor(numel(trainingSet.Labels)/miniBatchSize);
options = trainingOptions('sgdm',...
    'MiniBatchSize',miniBatchSize,...
    'MaxEpochs',4,...
    'InitialLearnRate',1e-4 );
%%
netTransfer = trainNetwork(trainingSet,layers,options);

%%
correctPredict=0;
for i=1:length(validationSet.Files)
    image = readimage(validationSet,i);
    label = classify(net,image);
    if label == validationSet.Labels(i)
        correctPredict=correctPredict+1;
    end
    disp(label);
end

