% function shorten_dat_files(datapath,recording,recording_window)
% Creates a concatenated dat file, optionally with analog channels as well
%
% Inputs:
% datapath: path to the files (e.g.: 'G:\IntanData\' )
% recordings: name of recordings (e.g.: {'Peter_160831_182631'} )
% combine_amp_analog: further combines amplifier.dat with analogin.dat
% add_empty_space : Adds empty space between recordings (NOT implementet!)
%
% By Peter Petersen
% petersen.peter@gmail.com

datapath = 'G:\IntanData\';
recording = 'Peter_MS7_161008_134424';
recording_length_shortened = 190*60; % length in seconds
% Opening the output .dat file
recording_dir_path = [datapath, recording,'\'];
fname_concat = [recording_dir_path, 'amplifier_shortened.dat'];
h = fopen(fname_concat,'w+');

Intan_rec_info = read_Intan_RHD2000_file_Peter(recording_dir_path);
sr = Intan_rec_info.frequency_parameters.amplifier_sample_rate;
nbChan = size(Intan_rec_info.amplifier_channels,2);
durationPerChunk = 10; % In second

disp(['Shortening ' recording]);
recording_dir_path = [datapath, recording,'\'];

% Amplifier channels
% MyFileInfo = dir([recording_dir_path, 'amplifier.dat']);
% nb_chunks = ceil(MyFileInfo.bytes/20000/nbChan/2);
nb_chunks = ceil(recording_length_shortened/durationPerChunk); 
f = fopen([recording_dir_path, 'amplifier.dat'],'r');

for i = 1:nb_chunks
    if mod(i,100)==0
        % disp([num2str(round(100*i/nb_chunks)) ' percent']);
        if i~=100
            fprintf(repmat('\b',[1 length([num2str(round(100*(i-100)/nb_chunks)), ' percent'])]))
        end
        fprintf('%d percent', round(100*i/nb_chunks));
    end
    fwrite(h,LoadBinaryChunk(f,'frequency',sr,'nChannels',nbChan,'channels',1:nbChan,'duration',durationPerChunk,'skip',0)','int16');
end
fclose(f);
fclose(h);

fprintf('\n dat file shortened successfully!');
