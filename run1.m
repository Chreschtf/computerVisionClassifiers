trainImg=imageSet('C:\Users\aswin\computerVisionClassifiers\training\training','recursive');
trainsetLeng=length(trainImg);

for ii=1:trainsetLeng
    for jj=1:100
    imgRead=read(trainImg(ii),jj);
    imgResize=imresize(imgRead,[16 16]);
    end
end


