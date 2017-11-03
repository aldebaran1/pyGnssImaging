clear all;
close all;
clc;

savemovie = 'movie/';     
imagefolder = 'new1/';       
imagefiles = dir('new1/*.png');      
nfiles = length(imagefiles);    % Number of files found

for ii=1:nfiles
   currentfilename = imagefiles(ii).name;
   currentimage = imread(strcat(imagefolder, currentfilename));
   images{ii} = currentimage;
end

for i=1:ii
    f(i) = im2frame(images{i});
end

v = VideoWriter(strcat(savemovie,'Eclipse_mov3.avi'));
v.FrameRate = 10;
open(v)
writeVideo(v,f);
close(v);