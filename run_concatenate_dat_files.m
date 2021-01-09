datapath = 'D:\IntanData\Peter_MS12';
recordings = {'Peter_MS12_170713_093400','Peter_MS12_170713_141709','Peter_MS12_170713_145428'};
combine_amp_analog = 0;
add_empty_space = 0;
concatenate_dat_files(datapath,recordings,combine_amp_analog,add_empty_space);

%% % Concat digital channels
datapath = 'D:\IntanData\Peter_MS10\';
recordings = {'Peter_MS12_170709_201401','Peter_MS10_170320_212321','Peter_MS10_170320_220510'};

fname_concat = fullfile(datapath, recordings{1}, 'digitalin_concat.dat');
h = fopen(fname_concat,'w+');
disp('Concatenating digital channels...')
for i = 1:length(recordings)
    disp(['Loading digital channels from ' recordings{i}])
    m = memmapfile(fullfile(datapath, recordings{i}, 'digitalin.dat'),'Format','uint16','writable',false);
    fwrite(h,m.Data,'uint16');
end
fclose(h);
disp('Finished concatenating digital channels')
