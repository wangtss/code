clear all;

% define database info, remember to change the name and path to your own
load LIVEVIDEOData.mat;
root = 'E:\DATABASE\videodatabase\live\videos\';
dims = [432, 768];
video_num = length(dmos_all);

% task specific info
addpath('ms_ssim');
block_size = [64, 64, 4];
stride = [48, 48, 3];

for index_video = 1:video_num
    disp(index_video);
    
    % load reference and distorted video name
    disName = [root file_name{index_video}];
    refName = [root ref_name{ref_no(index_video)}];
    
    % run function
    output_pattern = ['data\', file_name{index_video}(1:size(file_name{index_video}, 2)-4)];
    generate_block(refName, disName, dims, block_size, stride, output_pattern);
end
