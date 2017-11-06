clear all;
close all;
clc;

savemovie = 'movie/';     
imagefolder = 'day233/';       
imagefiles = dir(strcat(imagefolder, '*.png'));      
nfiles = length(imagefiles);    % Number of files found

for ii=1:nfiles
   currentfilename = imagefiles(ii).name;
   currentimage = imread(strcat(imagefolder, currentfilename));
   images{ii} = currentimage;
end

for i=1:ii
    f(i) = im2frame(images{i});
end

v = VideoWriter(strcat(savemovie,'day233.avi'));
v.FrameRate = 5;
open(v)
writeVideo(v,f);
close(v);