clc;
clear all;
trainImg=imageSet('.\training','recursive');
imgCell={trainImg};
imgCellLen=length(imgCell);
imgVector1=zeros(100,256);
tinyFeature=zeros(1500,256);
for ii=1:15
    for jj=1:100
        imgRead=read(trainImg(ii),jj);
        ig=imgRead;
        [xDim,yDim]=size(imgRead);
        if xDim~=yDim
            squareDim=min(xDim,yDim);
            x_topLeftCorner=xDim-squareDim-floor((xDim-squareDim)/2);
            y_topLeftCorner=yDim-squareDim-floor((yDim-squareDim)/2);
            imgRead=imcrop(imgRead, [y_topLeftCorner,x_topLeftCorner,squareDim,squareDim]);
        end
        imgResize=imresize(imgRead,[16 16]);
        imgResize=im2double(imgResize);
        imgVector = reshape(imgResize,[256,1]);
        imgVector=imgVector-mean(imgVector);
        imgVector = imgVector/std(imgVector);
        imgVector1(jj,:) = imgVector;
    end
    tinyFeature=vertcat(tinyFeature,imgVector1);
end


