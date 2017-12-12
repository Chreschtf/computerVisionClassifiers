clc;
clear all;
trainImg=imageSet('.\training','recursive');
imgCell={trainImg};
imgCellLen=length(imgCell);
imgVector1=zeros(100,256);
tinyFeature=zeros(1500,256);
for ii=1:15
    for jj=1:100
        if jj == 1
            disp(ii);
        end
        imgRead=read(trainImg(ii),jj);
        imgRead=im2double(imgRead);
        [xDim,yDim]=size(imgRead);
        if xDim~=yDim
            %disp(ii+", "+jj);
            xCenter=floor(xDim/2);
            yCenter=floor(yDim/2);
            squareDim=yCenter;
            if yCenter>xCenter
                squareDim = xCenter;
            else
                squareDim = yCenter;
            end
            xLeft=xCenter-squareDim;
            width=squareDim*2;%xLeft+2*xLim;
            yTop=yCenter-squareDim;
            height=squareDim*2%yTop+2*yLim;
            imgRead2=imcrop(imgRead, [xLeft, yTop, width, height]);
            imgRead=imcrop(imgRead, [xLeft, yTop, width, height]);
        end
        imgResize=imresize(imgRead,[16 16]);
        imgVector = reshape(imgResize,[256,1]);
        imgVector=imgVector-mean(imgVector);
        imgVector = imgVector/std(imgVector);
        imgVector1(jj,:) = imgVector;
    end
    tinyFeature=vertcat(tinyFeature,imgVector1);
    %jj=0;
end


