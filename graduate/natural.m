jsonFilePath = 'D:\Program\Matlab_natural\griddata_input.json';

% 读取 JSON 文件
try
  jsonData = jsondecode(fileread(jsonFilePath));
catch ME
  error('无法读取 JSON 文件: %s', ME.message);
end

% 从JSON数据中提取xi,yi,zi 这里的xi,yi其实就是网格数据了，与下面的[Xq,Yq]是一致的
xi_data = jsonData.xi;
yi_data = jsonData.yi;
zi_data = jsonData.zi;

% 输出数据维度验证
disp(['x_data 维度: ',num2str(size(x_data))]);
disp(['y_data 维度: ',num2str(size(y_data))]);
disp(['z_data 维度: ',num2str(size(z_data))]);

% 创建网格数据
[Xq, Yq] = meshgrid(-0.2:0.001:0.2, -0.2:0.001:0.2);

disp(['Xq 维度: ', num2str(size(Xq))]);
disp(['Yq 维度: ', num2str(size(Yq))]);

% 使用自然临近插值
Zq_na = griddata(x_data, y_data, z_data, xi_data, yi_data, 'natural');
disp(['Zq_na 维度: ',num2str(size(Zq_na))]);

jsonData_zq_na = jsondecode(jsonencode(jsonData));

jsonData_zq_na.zi = Zq_na;

% 输出文件路径
newJsonPath = 'D:\Program\Matlab_natural\natural_matlab.json';

% 将修改后的数据写入新 JSON 文件
jsonStr = jsonencode(jsonData_zq_na);
fid = fopen(newJsonPath, 'w');
fwrite(fid, jsonStr, 'char');
fclose(fid);

disp(['新 JSON 文件已保存至: ',newJsonPath]);
