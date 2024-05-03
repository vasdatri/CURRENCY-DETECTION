
clc;
close all;
clear all;
%------------ Image Reading ------------------------------------------
delete Pdata.mat;
FilePathStorage=pwd;
FilePathname1=strcat(FilePathStorage,'/Real_Notes/');
FiledirName1=dir([FilePathname1 '*.jpg']);
TotalImages1=size(FiledirName1,1);
PFeatures = double(zeros(TotalImages1));
Train_dataP=zeros(TotalImages1,25);
Train_LabP=zeros(TotalImages1,1);
for PPP=1:TotalImages1
Filename1=FiledirName1(PPP).name;
FilePathDatabase1=[FilePathname1,Filename1];
[DataArray,map]=imresize(imread(FilePathDatabase1),[300,650]);
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
% A=imrect;
% xy = A.getPosition
% % 100 Notes
% A(1,:)=[459,199,106,45];
% A(2,:)=[590,154,27,20];
% A(3,:)=[634,97,17,48];
% A(4,:)=[583,177,43,75];
% A(5,:)=[384.0000,196.0000,51.0000,54.0000];
% A(6,:)=[369.0000,3.0000,17.0000,298.0000];
% A(7,:)=[104.0000,122.0000,42.0000,88.0000];
% A(8,:)=[62,239,86,35];
% A(9,:)=[58.0000,196.0000,15.0000,34.0000];
% A(10,:)=[8,88,17,54];
% A(11,:)=[184,151,18,44];
% % 200 Notes
% A(1,:)=[472,200,100,37];
% A(2,:)=[605,144,17,22];
% A(3,:)=[637,74,15,68];
% A(4,:)=[590.0000,168.0000,45.0000,77.0000];
% A(5,:)=[391.0000,196.0000,51.0000,51.0000];
% A(6,:)=[366,3,17,292];
% A(7,:)=[114,124,35,80];
% A(8,:)=[60,239,89,29];
% A(9,:)=[61.0000,187.0000,22.0000,42.0000];
% A(10,:)=[3.0000,64.0000,19.0000,75.0000];
% A(11,:)=[181,150,19,42];
% % 500 Notes
% A(1,:)=[475,198,102,42];
% A(2,:)=[606,151,21,22];
% A(3,:)=[634,83,16,55];
% A(4,:)=[591,174,48,76];
% A(5,:)=[398.0000,199.0000,50.0000,50.0000];
% A(6,:)=[365,3,17,297];
% A(7,:)=[112,123,34,88];
% A(8,:)=[53,240,96,31];
% A(9,:)=[56.0000,192.0000,13.0000,42.0000];
% A(10,:)=[2.0000,69.0000,20.0000,64.0000];
% A(11,:)=[188.0000,147.0000,17.0000,42.0000];
% 2000 Notes
% A(1,:)=[482,198,111,42];
% A(2,:)=[602,148,27,20];
% A(3,:)=[633,64,21,87];
% A(4,:)=[597,169,43,81];
% A(5,:)=[413.0000,195.0000,45.0000,56.0000];
% A(6,:)=[380.0000,3.0000,17.0000,297.0000];
% A(7,:)=[122.0000,117.0000,28.0000,97.0000];
% A(8,:)=[64,234,101,39];
% A(9,:)=[50,173,25,56];
% A(10,:)=[2,61,21,85];
% A(11,:)=[200,140,15,50];
% save 2000.mat A
% RGB to Gray
Igray = 0.30*r_channel + 0.59*g_channel + 0.11*b_channel;
% figure,imshow(uint8(Igray));
% title('Gray Image');
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
% figure, imshow(edhist)
% title('Edge Detection');
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
title('Candidate Region');
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
disp('Total Features');
disp(Tfeat)
Train_dataP(PPP,:)=Tfeat;
Train_LabP(PPP,1)=1;
end
save Pdata.mat Train_dataP Train_LabP
