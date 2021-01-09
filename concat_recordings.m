% Concat recordings within every animal

%
clear all
% datapath = '/Volumes/P/IntanData/'; % Location of the recording computer
% datapath = '/Volumes/TurtlePower/DataBank/Buzsakilab/';
datapath = 'G:\IntanData\';
Recordings_MedialSeptum
rat_id = 'MS8';
arena = 'wheelbox';
% arena = 'homecage';
cooling_area = 'Medial Septum';
% cooling_area = 'Supramammillary nucleus';
concat_recordings = [];
concat_datasets = []
ids_accepted = 1;
for i = 1:size(recordings,2)
    if strcmp(recordings(i).rat_id, rat_id) & strcmp(recordings(i).arena, arena) & strcmp(recordings(i).cooling_area, cooling_area);
        concat_recordings(ids_accepted) = i;
        concat_datasets{ids_accepted} = recordings(i).name;
        ids_accepted = ids_accepted + 1;
    end
end
disp('Concatenating recordings...')
concat_datasets
concatenate_dat_files(datapath,concat_datasets,0,0);
disp('Complete!')
