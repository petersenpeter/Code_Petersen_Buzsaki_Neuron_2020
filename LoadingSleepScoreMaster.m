% Loading file info
clear all
datapath = 'G:\IntanData\';
Recordings_MedialSeptum
id = 63;
recording = recordings(id).name;
cd([datapath, recording])

% % Create eeg file
disp('Creating EEG file')
downsample_dat_to_eeg([datapath, recording,'\']);
copyfile('amplifier.xml',[recording '.xml'])
movefile('amplifier.eeg',[recording '.eeg'])

disp('EEG file created successfully')
% % Automatic sleep scoring
disp('Performing automatic sleep scoring')
SleepScoreMaster(datapath,recording,'overwrite',true)
disp('clustering complete')
