files = dir('*.jpg');
picNum=length(files);
fid=fopen('./train.txt','wt');
for i = 1:picNum
    fprintf(fid,'%s\n',files(i).name(1:end-4));
end
fclose(fid); 