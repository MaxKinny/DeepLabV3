files = dir('*.JPG');
picNum=length(files);
for i = 1:picNum
    I = imread(files(i).name);
    imwrite(I, ['../', files(i).name(1:end-4),'.png'])
end