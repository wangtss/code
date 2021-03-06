function generate_block(ref_name, dis_name, dims, block_size, stride, output_pattern)
    % load reference and distorted video frames
    ref_frames = Yuv2Frame(ref_name, dims(1), dims(2));
    dis_frames = Yuv2Frame(dis_name, dims(1), dims(2));
    frame_num = min(size(ref_frames, 3), size(dis_frames, 3));
    block_size = block_size - 1;
    % generate block
    for h = 1:stride(1):dims(1) - block_size(1)
        for w = 1:stride(2):dims(2) - block_size(2)
            for t = 1:stride(3):frame_num - block_size(3)
                ref_video_block = ref_frames(h:h+block_size(1), w:w+block_size(2), t:t+block_size(3));
                dis_video_block = dis_frames(h:h+block_size(1), w:w+block_size(2), t:t+block_size(3));
                data.ac = sliding_window_dct(dis_video_block);
                data.label = compute_label(ref_video_block, dis_video_block);
                save_name = [output_pattern, '_', int2str(h), int2str(w), int2str(t)];
                save(save_name, 'data');
            end
        end
    end
     
end

function frame_num = getFrameNum(filename, height, width)
    fid = fopen(filename, 'r');
    fseek(fid, 0, 'eof');
    size = ftell(fid);
    frame_num = size / (height * width * 1.5);
    fclose(fid);
end

function frame = Yuv2Frame(filename, height, width, frameNum)

    if (nargin == 3)
        frameNum = getFrameNum(filename, height, width);
    end

    fid = fopen(filename, 'r');
    fseek(fid, 0, 'bof');
    Y = zeros(height, width, frameNum);

    for index_frame = 1:frameNum
        temp = fread(fid, width * height,'uchar')';
        temp = reshape(temp, [width height]);
        Y(:, :, index_frame) = temp';
        Cb = fread(fid, width * height / 4, 'uchar');
        Cr = fread(fid, width * height / 4, 'uchar');
    end

    frame = Y;
    fclose(fid);    
end

function ac = sliding_window_dct(block)
    window_size = [4 - 1, 4 - 1, 4 - 1];
    window_stride = [1, 1, 4];
    block_size = size(block);
    raw_ac = [];
    
    for u = 1:window_stride(1):block_size(1) - window_size(1)
        for v = 1:window_stride(2):block_size(2) - window_size(2)
            for w = 1:window_stride(3):block_size(3) - window_size(3)
                sub_block = block(u:u + window_size(1), v:v + window_size(2), w:w + window_size(3));
                res = reshape(mirt_dctn(sub_block), [1, 64]);
                res(1) = 0;
                raw_ac = [raw_ac, sum(res)];
            end
        end
    end
    ac = zeros(block_size(1), block_size(2));
    raw_ac = reshape(raw_ac, [61, 61]);
    ac(1:61, 1:61) = raw_ac;
end

function score = compute_label(ref_block, dis_block)
    block_size = size(ref_block);
    spatial_score = zeros(1, block_size(3));
    temporal_score = zeros(1, block_size(3) - 1);
    
    for i = 1:block_size(3)
        ref_frame = ref_block(:, :, i);
        dis_frame = dis_block(:, :, i);
        spatial_score(i) = ssim_mscale_new(ref_frame, dis_frame);
        if i > 1
            ref_flow = optical_flow(ref_block(:, :, i), ref_block(:, :, i - 1));
            dis_flow = optical_flow(dis_block(:, :, i), dis_block(:, :, i - 1));
            temporal_score(i) = ssim_mscale_new(ref_flow, dis_flow);
        end
    end
    temporal_score(1) = 0;
    score = sum([spatial_score, temporal_score]) / (2 * block_size(3) - 1);
end

function flow = optical_flow(pre, cur)
    i1 = zeros(128, 128);
    i2 = zeros(128, 128);
    i1(1:64, 1:64) = pre;
    i2(1:64, 1:64) = cur;
    [vx, vy, re] = opticalFlow(i1, i2);
    vx = vx(1:64, 1:64);
    vy = vy(1:64, 1:64);
    flow = real(sqrt(vx^2 + vy^2));
end
