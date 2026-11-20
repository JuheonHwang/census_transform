% imstereo_sch_stereo.m
% Original MATLAB reference implementation of census-transform stereo matching.
% Pipeline: census transform -> XOR + popcount cost -> box-filter aggregation
%           -> winner-take-all (+ optional parabolic sub-pixel) -> returns
%           left and right disparity maps (dmapL, dmapR). The left-right
%           consistency check is done by the caller (see matching.m).
% Usage: [dmapL, dmapR] = imstereo_sch_stereo(imL, imR, [dmin dmax], ws, wc, true)
%        ws = aggregation window size, wc = census window size.
% NOTE: logic is kept exactly as in the original; do not edit when porting.

function [dmapL, dmapR, costL, costR] = imstereo_sch_stereo(imL, imR, drange, ws, wc, sub_pixel_refine)
    %IMSTEREO_SCH Stereo block-matching with SCH (Census)
    
    imRC = cenesus(imR, wc);
    imLC = cenesus(imL, wc);
    
    dmin=drange(1); dmax=drange(2);
    sch=ones([size(imR),dmax-dmin+1], class(imRC))*Inf;
    for d = 0:dmax-dmin
        imL_d = circshift(imLC,[0, -(dmin+d)]); % shift
        imL_d(:, end-(dmin+d):end, :) = 0;
        z = xor(imRC, imL_d);
        sch(:,:,d+1)= imfilter(sum(z, 3), ones(ws));
    end
    [costR,dmapR]=min(sch,[],3);
    
    % SUBPIXEL
    if sub_pixel_refine
        cost = reshape(sch, [], diff(drange)+1);
        tar_idx = (dmapR(:)>1 & dmapR(:)<=diff(drange));
        cost = cost(tar_idx, :);
        dP = dmapR(tar_idx) + 1; dM = dmapR(tar_idx) - 1; dD = dmapR(tar_idx);

        yP = sub2ind(size(cost), 1:size(cost, 1), dP')';
        yM = sub2ind(size(cost), 1:size(cost, 1), dM')';
        yD = sub2ind(size(cost), 1:size(cost, 1), dD')';

        dmapR(tar_idx) = dmapR(tar_idx) + ((cost(yP) - cost(yM)) ./ (2.*( 2.*cost(yD) - cost(yM) - cost(yP))));
    end
    %%%%%%%%%%
    
    dmapR=dmapR+dmin-1;
    
    
    dmin=-drange(2); dmax=drange(1);
    sch=ones([size(imL),dmax-dmin+1], class(imLC))*Inf;
    for d = 0:dmax-dmin
        imR_d = circshift(imRC,[0, -(dmin+d)]); % shift
        imR_d(:, end-(dmin+d):end, :) = 0;
        z = xor(imLC, imR_d);
        sch(:,:,d+1)= imfilter(sum(z, 3), ones(ws));%fspecial('gaussian',ws,ws/2));%
    end
    [costL,dmapL]=min(sch,[],3);
    
    % SUBPIXEL
    if sub_pixel_refine
        cost = reshape(sch, [], diff(drange)+1);
        tar_idx = (dmapL(:)>1 & dmapL(:)<=diff(drange));
        cost = cost(tar_idx, :);
        dP = dmapL(tar_idx) + 1; dM = dmapL(tar_idx) - 1; dD = dmapL(tar_idx);

        yP = sub2ind(size(cost), 1:size(cost, 1), dP')';
        yM = sub2ind(size(cost), 1:size(cost, 1), dM')';
        yD = sub2ind(size(cost), 1:size(cost, 1), dD')';

        dmapL(tar_idx) = dmapL(tar_idx) + ((cost(yP) - cost(yM)) ./ (2.*( 2.*cost(yD) - cost(yM) - cost(yP))));
    end
    %%%%%%%%%%
    
    dmapL=dmapL+dmin-1;
    dmapL = -dmapL;
end

function imLCensus = cenesus(imL, wc)
    imLmargin = zeros(size(imL)+[wc wc]-1, class(imL));
    imLmargin(floor((wc-1)/2)+(1:size(imL, 1)),floor((wc-1)/2)+(1:size(imL, 2))) = imL;
    imLBlock = im2col(imLmargin, [wc wc], 'sliding');
    imLBlock = imLBlock > imLBlock(ceil((wc*wc)/2), :);
    %imLBlock(ceil((wc*wc)/2), :) = [];
    imLCensus = permute(reshape(imLBlock, [wc*wc size(imL)]), [2 3 1]);
end

%% Warning: SCH needs 'nlfilter' from the Image Processing Toolbox