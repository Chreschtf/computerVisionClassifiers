clc;
clear all;
trainImg=imageSet('.\training','recursive');
imgCell={trainImg};
imgCellLen=length(imgCell);
resizeImageDim=16;
imgVector1=zeros(100,resizeImageDim*resizeImageDim);
tinyFeature=double.empty;
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
        imgResize=imresize(imgRead,[resizeImageDim resizeImageDim]);
        imgResize=im2double(imgResize);
        imgVector = reshape(imgResize,[resizeImageDim*resizeImageDim,1]);
        %imgVector=imgVector-mean(imgVector);
        %imgVector = imgVector/std(imgVector);
        imgVector1(jj,:) = imgVector;
    end
    tinyFeature=vertcat(tinyFeature,imgVector1);
end
%%
% minX=min(tinyFeature);
% maxX=max(tinyFeature);
meanX=mean(tinyFeature);
stdX=std(tinyFeature);
normX=norm(tinyFeature);
% normImg=(tinyFeature-meanX)./(max(tinyFeature)-min(tinyFeature));
unitLen=(tinyFeature)/normX;
%%
n=100;
X=unitLen;
classLabel={'bedroom','coast','Forest','HighW','industrial','Insidecity','Kitchen','LivingR','Mountain','Office','OpenCont','Store','Street','Suburb','TallB'}';
Y=[[repmat(classLabel(1),n,1);repmat(classLabel(2),n,1);repmat(classLabel(3),n,1);repmat(classLabel(4),n,1);repmat(classLabel(5),n,1);repmat(classLabel(6),n,1);repmat(classLabel(7),n,1);repmat(classLabel(8),n,1);repmat(classLabel(9),n,1);
             repmat(classLabel(10),n,1);repmat(classLabel(11),n,1);repmat(classLabel(12),n,1);repmat(classLabel(13),n,1);repmat(classLabel(14),n,1);repmat(classLabel(15),n,1)]];
Mdl = fitcknn(X,Y,'NumNeighbors',5,'Standardize',1)

