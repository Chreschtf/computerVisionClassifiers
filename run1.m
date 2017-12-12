clc;
clear all;
trainImg=imageSet('C:\Users\aswin\computerVisionClassifiers\training\training','recursive');
imgCell={trainImg};
imgCellLen=length(imgCell);
imgVector1=zeros(100,256);
tinyFeature=zeros(1500,3840);
for ii=1:15
    for jj=1:100
    imgRead=read(trainImg(ii),jj);
    imgResize=imresize(imgRead,[16 16]);
    imgVector = imgResize(:)'; 
    imgVector1(jj,:) = imgVector;
    end
    tinyFeature=vertcat(tinyFeature,imgVector1);
    jj=0;
end


