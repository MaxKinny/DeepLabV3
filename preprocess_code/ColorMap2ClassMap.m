files = dir('*.png');
picNum=length(files);
for i = 1:picNum
    I = imread(files(i).name);
    I_tmp = uint8(zeros(size(I,1),size(I,2)));
    I_tmp = I_tmp + uint8(I(:,:,2)==255);
    I = I_tmp + 2*uint8(I(:,:,3)==255);
    imwrite(I, ['../', files(i).name])
end