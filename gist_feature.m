clear param
param.orientationsPerScale = [8 8 8 8]; % number of orientations per scale (from HF to LF)
param.numberBlocks = 1;
param.fc_prefilt = 4;
%load_path = 'rlcc/Windows/Downloads/SSRL_SKY/';
load_path = '/home/lcc/code/data/SSRL_SKY_CAM_IMAGE'
save_path = '/home/lcc/code/python/SolarPrediction/dataset/NREL_SSRL_BMS_SKY_CAM/inputdata/'
file_list = [];
feat_list = [];
for y = 1999:2016
    for m = 1:12
        if m < 10
            month_path = strcat(load_path, int2str(y), '/', '0', int2str(m),'/');
        else
            month_path = strcat(load_path, int2str(y), '/', int2str(m),'/');
        end
        files = dir(month_path);
        imgDataPath = month_path;
        imgDataDir  = dir(imgDataPath);             % 遍历所有文件
        for i = 1:length(imgDataDir)
            if(isequal(imgDataDir(i).name,'.')||... % 去除系统自带的两个隐文件夹
                isequal(imgDataDir(i).name,'..')||...
                ~imgDataDir(i).isdir)                % 去除遍历中不是文件夹的
            continue;
            end
            imgDir = dir([imgDataPath imgDataDir(i).name '/*jpg']); 
            for j =1:length(imgDir)                 % 遍历所有图片
                str = [imgDataPath imgDataDir(i).name '/' imgDir(j).name]
                img = imread(str);
                [gist, param] = LMgist(img, '', param);
                file_list = [file_list;imgDir(j).name(1:12)];
                feat_list = [feat_list;gist];
%                 size(file_list)
%                 size(feat_list)
            end
        end
    end
end
csv.write('exist_file_list.csv', file_list);
csv.write('GIST.csv', feat_list);
% 
% % 计算GIST
% 
% imgDataPath = load_path;
% imgDataDir  = dir(imgDataPath);             % 遍历所有文件
% for i = 1:length(imgDataDir)
%     if(isequal(imgDataDir(i).name,'.')||... % 去除系统自带的两个隐文件夹
%        isequal(imgDataDir(i).name,'..')||...
%        ~imgDataDir(i).isdir)                % 去除遍历中不是文件夹的
%            continue;
%     end
%     imgDir = dir([imgDataPath imgDataDir(i).name '/*jpg']); 
%     for j =1:length(imgDir)                 % 遍历所有图片
%         str = [imgDataPath imgDataDir(i).name '/' imgDir(j).name];
%         disp(str);
%     end
% end
