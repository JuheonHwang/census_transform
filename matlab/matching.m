% matching.m
% Batch driver script for the MATLAB reference: loops over image pairs, calls
% imstereo_sch_stereo, applies the left-right consistency check (interp2 +
% threshold), displays the maps and saves them as .mat files.
% NOTE: paths (input folder, Windows-style output path) are hard-coded and
% refer to the original author's local dataset; edit them before running.
% Requires the Image Processing Toolbox (and a GPU when use_gpu = true).

clear; close all; fclose all;
global use_gpu;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PARAMETERS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
census_filter_size = 11;
aggreate_filter_size = 11;
min_disp = 0; max_disp = 300;
thres_LR_disp = 10;
sub_pixel_refine = true;
nCam = 16;
resize_scale = 1;
rotate_angle = 0;
use_gpu = true;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for i = 1:nCam
    % tic;
    fileList1 = dir(fullfile('imgout_virtual_4m_hjh', sprintf('*%02d.png', (i-1)*2)));
    fileList2 = dir(fullfile('imgout_virtual_4m_hjh', sprintf('*%02d.png', (i-1)*2+1)));

    for j = 1:length(fileList1)
        tic;
        % L = G(double(imread(fullfile(fileList1(j).folder, fileList1(j).name))));
        % R = G(double(imread(fullfile(fileList2(j).folder, fileList2(j).name))));
        L = G(imread(fullfile(fileList1(j).folder, fileList1(j).name)));
        R = G(imread(fullfile(fileList2(j).folder, fileList2(j).name)));

        % L = imrotate(L, rotate_angle);
        % R = imrotate(R, rotate_angle);

        L = imresize(L, resize_scale);
        R = imresize(R, resize_scale);

        L_names = split(fileList1(j).name, '.');

%% CENSUS 
        % RIGHT
        [dmapL, dmapR, costL, costR] = imstereo_sch_stereo(L, R, [min_disp max_disp], aggreate_filter_size, census_filter_size, sub_pixel_refine);
        % LEFT
        %[dmapL, cost] = imstereo_sch(R, L, [-max_disp min_disp], aggreate_filter_size, census_filter_size);
        
        
        [H, W] = size(dmapL);
        [X,Y] = meshgrid(G(1:W), G(1:H));
        dmapRtoL = interp2(dmapR, X-dmapL, Y, 'linear', 0);
        dmapLtoR = interp2(dmapL, X+dmapR, Y, 'linear', 0);
        
        dmapL(abs(dmapRtoL - dmapL) > thres_LR_disp) = 0;
        dmapR(abs(dmapLtoR - dmapR) > thres_LR_disp) = 0;
        
        toc;
        % figure('Name','Left'); imshow(L./255.0)
        figure('Name','Left'); imshow(dmapL./max(dmapL(:)))
        % figure('Name','Right'); imshow(R./255.0)
        figure('Name','Right'); imshow(dmapR./max(dmapR(:)))
        dmapL = imrotate(gather(dmapL), -1 * rotate_angle); dmapR = imrotate(gather(dmapR), -1 * rotate_angle);
        save(['imgout_virtual_4m_hjh\disparity\' L_names{1} '_' num2str(census_filter_size) '_' num2str(aggreate_filter_size) '_' num2str(size(L, 1)) 'x' num2str(size(L, 2))], 'dmapL', 'dmapR');
    end
    % toc;
end


function A = G(A)
    global use_gpu; %#ok<GVMIS> 
    if use_gpu
        A = gpuArray(A);
    end
end