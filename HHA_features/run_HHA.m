%%Root directory for dataset
data_root = '/home/aeroscout2/data/NYUv2_HHA';

for i = 0:1448 
    % getting directory number
    dir_num = sprintf('%05d',i); 
    
    % current directory path
    dir_path = fullfile(data_root,dir_num);
    imgname = 'HHA' ;
    
    % depth and raw depth image path
    d_path = fullfile(dir_path,'depth.png');
    rd_path = fullfile(dir_path,'raw_depth.png');
    D = imread(d_path);
    RD = imread(rd_path); 
    
    %Calculating camera matrix
    C = cropCamera(getCameraParam('depth'));
    % Compute the HHA Features
    HHA = saveHHA(imgname, C, dir_path, D, RD);

end


