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

% %%
% IM=imgResize;
% IM=IM-mean(IM);
% IM=IM/norm(IM);
% %%
% IM=IM.*IM;
% s=sum(sum(IM));
% disp(sqrt(s));
