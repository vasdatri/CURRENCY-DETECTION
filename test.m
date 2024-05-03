% input - give the image file name as input. eg :- car3.jpg
clc;
clear all;
close all;
% k=input('Enter the file name','s'); % input image; color image
im=imread('Images/2000_fake.jpeg');
figure; imshow(im);
im1=rgb2gray(im);
im1=medfilt2(im1,[3 3]); %Median filtering the image to remove noise%
BW = edge(im1,'sobel'); %finding edges
figure; imshow(im1);
[imx,imy]=size(BW);
msk1=[0 0 0 0 0;
     0 1 1 1 0;
     0 1 1 1 0;
     0 1 1 1 0;
     0 0 0 0 0;];

B=conv2(double(BW),double(msk1)); %Smoothing  image to reduce the number of connected components
msk2=[1 1 1 1 1;
     1 0 0 0 1;
     1 0 0 0 1;
     1 0 0 0 1;
     1 1 1 1 1;];
B=conv2(double(B),double(msk2));
L = bwlabel(B,8);% Calculating connected components
values=unique(L).';
mx=max(max(L))
% There will be mx connected components.Here U can give a value between 1 and mx for L or in a loop you can extract all connected components
% If you are using the attached car image, by giving 17,18,19,22,27,28 to L you can extract the number plate completely.
figure,imshow(B);
for j=1:mx
    [r,c] = find(L==j);  
    rc = [r c];
    [sx sy]=size(rc);
    n = zeros(imx,imy);
    for i=1:sx
        x1=rc(i,1);
        y1=rc(i,2);
        n(x1,y1)=255;
    end % Storing the extracted image in an array
%     figure,imshow(n,[]); title(j)
    name=sprintf('segim%d.png',j);
    imwrite(n,name,'PNG');
end


%------------ Image Reading ------------------------------------------
[FILENAME,PATHNAME]=uigetfile('*.jpg','Select the Image');
FilePath=strcat(PATHNAME,FILENAME);
disp('The Image File Location is');
disp(FilePath);
[DataArray,map]=imresize(imread(FilePath),[300,650]);
figure,imshow(DataArray,map);
title('Input Image');
% Seperate Channel
r_channel=DataArray(:,:,1);
b_channel=DataArray(:,:,2);
g_channel=DataArray(:,:,3);
% Noise Removal
r_channel=medfilt2(r_channel);
g_channel=medfilt2(g_channel);
b_channel=medfilt2(b_channel);
%restore channels
rgbim(:,:,1)=r_channel;
rgbim(:,:,2)=g_channel;
rgbim(:,:,3)=b_channel;
figure,imshow(uint8(rgbim));
title('Denoised Image');
% RGB to Gray
Igray = 0.30*r_channel + 0.59*g_channel + 0.11*b_channel;
figure,imshow(uint8(Igray));
title('Gray Image');
% Edge Detection
y=double(Igray);
f1 = zeros(3,3,5);
f1(:,:,1) = [1 2 1;0 0 0;-1 -2 -1]; %vertical
f1(:,:,2) = [-1 0 1;-2 0 2;-1 0 1];   %horizontal
f1(:,:,3) = [2 2 -1;2 -1 -1; -1 -1 -1];% 45 diagonal
f1(:,:,4) = [-1 2 2; -1 -1 2; -1 -1 -1];%135 diagonal
f1(:,:,5) = [-1 0 1;0 0 0;1 0 -1]; % non directional
for i = 1:5
g_im(:,:,i) = filter2(f1(:,:,i),y);
end
[m, p] = max(g_im,[],3);
edim = edge(y, 'canny');
im2 =(p.*edim);
edhist=im2;
figure, imshow(edhist)
title('Edge Detection');
Avg=mean2(Igray);
if(Avg>202 && Avg<207)
    load 100.mat
elseif(Avg>175 && Avg<180)
    load 200.mat
elseif(Avg>190 && Avg<195)   
    load 500.mat
elseif(Avg>209 && Avg<214) 
    load 2000.mat
end
figure,imshow(uint8(DataArray));
title('ROI-Extract Texture & Statistical Features');
hold on
for n=1:size(A,1)
rectangle('Position',A(n,:),'EdgeColor','r','LineWidth',4)
end
pause(1)
figure,imshow(edhist);
title('ROI-Extract Edge & Shape Features');
hold on
for n=1:size(A,1)
rectangle('Position',A(n,:),'EdgeColor','r','LineWidth',4)
end
pause(1)
SFL_Data=zeros(size(A,1),6);
SSL_Data=zeros(size(A,1),12);
for n=1:size(A,1)
    imcropgray = imcrop(Igray,A(n,:));
    Img_data=imcropgray;
    % STATISTICAL FEATURES 
    % First Level Feature
    Mean = mean2(Img_data);
    Variance  = mean2(var(double(Img_data)));
    Kurtosis = kurtosis(double(Img_data(:)));
    stats = graycoprops(Img_data,'Contrast Correlation Energy Homogeneity');
    Energy = stats.Energy;
    Contrast = stats.Contrast;
    Entropy = entropy(Img_data);
    FL_Feat=[Mean Variance Kurtosis Energy Contrast Entropy];
    FL_Feat(isnan(FL_Feat))=0; 
%     disp('First Level Feature');
%     disp(FL_Feat)
    SFL_Data(n,:)=FL_Feat;
    % Second Level Feature
    offsets = [0 1; -1 1; -1 0; -1 -1];  %0°, 45°, 90°, 135°
    GLCM1 = graycomatrix(Img_data,'NumLevels',8,'Offset',offsets);
    GLCM2 = graycomatrix(Img_data,'NumLevels',32,'Offset',offsets);
    stats = graycoprops(GLCM1,'Contrast Correlation Energy Homogeneity');
    stats1 = graycoprops(GLCM2,'Contrast Correlation Energy Homogeneity');
    Correlation=[mean(stats.Correlation) mean(stats1.Correlation)];
    ASM=[mean(stats.Energy) mean(stats1.Energy)];
    Homogeneity=[mean(stats.Homogeneity) mean(stats1.Homogeneity)];
    IDM=[Inverse_Diff(GLCM1) Inverse_Diff(GLCM2)];
    Max_prob=[Maximium_Prob(GLCM1) Maximium_Prob(GLCM2)];
    Entropy = [entropy(GLCM1) entropy(GLCM2)];
    SL_Feat=[ASM Correlation Homogeneity IDM Max_prob Entropy];
    SL_Feat(isnan(SL_Feat))=0;
%     disp('Second Level Feature')
%     disp(SL_Feat)
    SSL_Data(n,:)=SL_Feat;
end
ST_feat=[mean(SFL_Data) mean(SSL_Data)];
disp('Statistical Features')
disp(ST_feat);
EF_Data=zeros(size(A,1),7);
for n=1:size(A,1)
    imcropedge = imcrop(edhist,A(n,:));
    % Edge Features
    results=regionprops(imcropedge,'Area','EulerNumber','Orientation','BoundingBox','Extent',...
        'Perimeter','Centroid','Extrema','PixelIdxList','ConvexArea',...
        'FilledArea','PixelList','ConvexHull','FilledImage','Solidity',...
        'ConvexImage','Image','SubarrayIdx','Eccentricity','MajorAxisLength',...
        'EquivDiameter','MinorAxisLength','EulerNumber');
    NR=vertcat(results.BoundingBox);
    Circularity=zeros(size(NR,1));
    Eccentricity=zeros(size(NR,1));
    Convexity=zeros(size(NR,1));
    Area=zeros(size(NR,1));
    Compactness=zeros(size(NR,1));
    Extent=zeros(size(NR,1));
    Solidity=zeros(size(NR,1));
    for ii=1:size(NR,1)
    Circularity(ii) = ((results(ii).Perimeter) .^2 )./ (4*(pi*(results(ii).Area)));
    Circularity(isnan(Circularity))=0;
    Circularity(isinf(Circularity)) = 0;
    Compactness(ii)=(4*results(ii).Area*pi)/(results(ii).Perimeter).^2;
    Compactness(isnan(Compactness))=0;
    Compactness(isinf(Compactness)) = 0;
    Convexity(ii)=results(ii).ConvexArea;
    Convexity(isnan(Convexity))=0;
    Convexity(isinf(Convexity)) = 0;
    Area(ii)=results(ii).Area;
    Area(isnan(Area))=0;
    Area(isinf(Area)) = 0;
    Eccentricity(ii)=results(ii).Eccentricity;
    Eccentricity(isnan(Eccentricity))=0;
    Eccentricity(isinf(Eccentricity)) = 0;
    Extent(ii)=results(ii).Extent;
    Extent(isnan(Extent))=0;
    Extent(isinf(Extent)) = 0;
    Solidity(ii)=results(ii).Solidity;
    Solidity(isnan(Solidity))=0;
    Solidity(isinf(Solidity)) = 0;
    end
    SF=[mean2(Area) mean2(Solidity) mean2(Convexity) mean2(Circularity) mean2(Eccentricity) mean2(Compactness) mean2(Extent)];
    EF_Data(n,:)=SF;
end
EDF_feat=mean(EF_Data);
disp('Edge Features')
Tfeat=[ST_feat EDF_feat];
load Pdata.mat
load Ndata.mat
xdata = [Train_dataP;Train_dataN];
group = [Train_LabP;Train_LabN];
svmTrain = svmtrain(xdata,group,'kernel_function','rbf');
Classfy_Result = svmclassify(svmTrain,Tfeat);
if(Classfy_Result==1)
    figure,imshow(DataArray,map);
    title('Currency Type: Real');
     msgbox('Currency Type: Real');
else
    figure,imshow(DataArray,map);
    title('Currency Type: Fake');
    msgbox('Currency Type: Fake');
end
% pause(2)
% perfdata
