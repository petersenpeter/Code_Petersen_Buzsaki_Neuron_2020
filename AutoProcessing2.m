% function Timer = AutoProcessing(ids)
% Script for processing daily recordings
% clear all

MedialSeptum_Recordings
datapath = 'Z:\peterp03\IntanData';
AquisitionDataPath = 'Z:\peterp03\IntanData';
SSDPath = 'G:\Kilosort';
ids = [173,174,175,176,177,178,179,180,181,182];
ids = [173];
t_auto = tic;
Timer = [];
% TODO: 173,174,175,176,177,178,179,180,181,182
% Complete: 

StepsToPerform = [5,7,8]; % [1,2,3,4,6,7,8];%1:10
% Steps:
% Step 1: Copying recording folders and renaming files
% Step 2: Concatenating recording files
% Step 3.1: Copying video file
% Step 3.2: Copying OptiTrack file
% Step 4: Creating lfp file
% Step 5: Loading temperature data
% Step 6: Calculating instantaneous theta
% Step 7: Loading Optitrack tracking data
% Step 8: Loading digital channels
% Step 9: Running Kilosort
% Step 10: Copying Kilosort output to SSD

for i = 1:length(ids)
    id = ids(i);
    recording = recordings(id);
    disp(['Processing recording ' num2str(id)])
%     if ~isempty(recording.dataroot)
%         datapath = recording.dataroot;
%     end
    AnimalPath = recording.animal_id;
    
    % Copying recording folders from PeterSetup if local folder is not existing already
    if any(StepsToPerform == 1)
        disp('Step 1: Copying recording folders from the share and renaming files')
        for j = 1:length(recording.concat_recordings)
            if ~exist(fullfile(datapath,AnimalPath,recording.concat_recordings{j}))
                disp(['copying recording folder: ' recording.concat_recordings{j}]);
                copyfile(fullfile(AquisitionDataPath,AnimalPath,recording.concat_recordings{j}) , fullfile(datapath,AnimalPath,recording.concat_recordings{j}));
            end
            % Renaming amplifier.* files
            if ~exist(fullfile(datapath,AnimalPath,recording.concat_recordings{j},[recording.concat_recordings{j},'.dat']))
                disp('Renaming .dat file')
                FileRename(fullfile(datapath,AnimalPath,recording.concat_recordings{j},'amplifier.dat') , fullfile(datapath,AnimalPath,recording.concat_recordings{j},[recording.concat_recordings{j},'.dat']))
            end
            if ~exist(fullfile(datapath,AnimalPath,recording.concat_recordings{j},[recording.concat_recordings{j},'.xml']))
                disp('Renaming .xml file')
                if exist(fullfile(datapath,AnimalPath,recording.concat_recordings{j},'amplifier.xml'))
                    FileRename(fullfile(datapath,AnimalPath,recording.concat_recordings{j},'amplifier.xml') , fullfile(datapath,AnimalPath,recording.concat_recordings{j},[recording.concat_recordings{j},'.xml']))
                else
                    copyfile(fullfile(datapath,AnimalPath,'amplifier.xml') , fullfile(datapath,AnimalPath,recording.concat_recordings{j},[recording.concat_recordings{j},'.xml']))
                end
            end
            if ~exist(fullfile(datapath,AnimalPath,recording.concat_recordings{j},[recording.concat_recordings{j},'.nrs']))
                if exist(fullfile(datapath,AnimalPath,recording.concat_recordings{j},'amplifier.nrs'))
                    disp('Renaming .nrs file')
                    FileRename(fullfile(datapath,AnimalPath,recording.concat_recordings{j},'amplifier.nrs') , fullfile(datapath,AnimalPath,recording.concat_recordings{j},[recording.concat_recordings{j},'.nrs']))
                end
            end
        end
        Timer.duration(i,1) = toc(t_auto);
        Timer.event{1} = 'Copying recording folders';
    end
    
    % Concatenating recording files
    if any(StepsToPerform == 2)
        disp('Step 2: Concatenating recording files')
        if exist(fullfile(datapath,AnimalPath,recording.name))~=7 | (exist(fullfile(datapath,AnimalPath,recording.name))==7 && exist(fullfile(datapath,AnimalPath,recording.name,[recording.name,'.dat']))~=2)
            combine_amp_analog = 0; add_empty_space = 0;
            concatenate_dat_files_v2(fullfile(datapath,AnimalPath),recording.concat_recordings);
            % concatenate_dat_files(fullfile(datapath,AnimalPath),recording.concat_recordings,combine_amp_analog,add_empty_space);
        end
        Timer.duration(i,2) = toc(t_auto);
        Timer.event{2} = 'Concatenating recording files';
    end
    cd(fullfile(datapath,AnimalPath,recording.name))
    % Copying video file to concat directory
    if any(StepsToPerform == 3)
        disp('Step 3.1: Copying video file')
        if iscell(recording.Cameratracking.Behavior)
            for j = 1:size(recording.Cameratracking.Behavior,2)
                if ~exist(fullfile(datapath,AnimalPath,recording.name,recording.Cameratracking.Behavior{j}))
                    disp(['Copying video file to concat directory: ' fullfile(datapath,AnimalPath,recording.name,recording.Cameratracking.Behavior{j})])
                    if exist(fullfile(datapath,AnimalPath,recording.concat_recordings{recording.concat_behavior_nb}))
                        copyfile( fullfile(datapath,AnimalPath,recording.concat_recordings{recording.concat_behavior_nb} , recording.Cameratracking.Behavior{j}), fullfile(datapath,AnimalPath,recording.name,recording.Cameratracking.Behavior{j}) )
                    else
                        disp('Video file does not exist')
                    end
                end
            end
        else
            if exist(fullfile(datapath,AnimalPath,recording.concat_recordings{recording.concat_behavior_nb}))
                if ~exist(fullfile(datapath,AnimalPath,recording.name,recording.Cameratracking.Behavior))
                    disp('Copying video file to concat directory')
                    copyfile( fullfile(datapath,AnimalPath,recording.concat_recordings{recording.concat_behavior_nb} , recording.Cameratracking.Behavior), fullfile(datapath,AnimalPath,recording.name,recording.Cameratracking.Behavior) )
                else
                    disp('Video file already exist')
                end
            else
                disp('Video file does not exist')
            end
        end
        
        % Copying OptiTrack file to concat directory
        disp('Step 3.2: Copying OptiTrack file')
        if iscell(recording.OptiTracktracking)
            for j = 1:size(recording.OptiTracktracking,2)
                if ~exist(fullfile(datapath,AnimalPath,recording.name,recording.OptiTracktracking{j}))
                    disp('Copying OptiTrack file to concat directory')
                    if exist(fullfile(datapath,AnimalPath,recording.concat_recordings{recording.concat_behavior_nb} , recording.OptiTracktracking{j}))
                        copyfile( fullfile(datapath,AnimalPath,recording.concat_recordings{recording.concat_behavior_nb} , recording.OptiTracktracking{j}), fullfile(datapath,AnimalPath,recording.name,recording.OptiTracktracking{j}) )
                    end
                end
            end
        else
            if ~exist(fullfile(datapath,AnimalPath,recording.name,recording.OptiTracktracking))
                disp('Copying OptiTrack file to concat directory')
                if exist(fullfile(datapath,AnimalPath,recording.concat_recordings{recording.concat_behavior_nb} , recording.OptiTracktracking))
                    copyfile( fullfile(datapath,AnimalPath,recording.concat_recordings{recording.concat_behavior_nb} , recording.OptiTracktracking), fullfile(datapath,AnimalPath,recording.name,recording.OptiTracktracking) )
                end
            end
        end
        Timer.duration(i,3) = toc(t_auto);
        Timer.event{3} = 'Copying tracking files';
    end
    
    % Creating .lfp file
    if any(StepsToPerform == 4)
        disp('Step 4: Creating lfp file')
        if ~exist(fullfile(datapath,AnimalPath,recording.name,[recording.name,'.lfp']))
            disp('Creating lfp file')
%             downsample_dat_to_eeg(recording.name,fullfile(datapath,AnimalPath,recording.name));
            bz_LFPfromDat(fullfile(datapath,AnimalPath,recording.name),'noPrompts',true)
        end
        Timer.duration(i,4) = toc(t_auto);
        Timer.event{4} = 'Creating lfp file';
    end
    
    % Exporting Temperature for SleepScoring
    if sum(StepsToPerform == 5)
        disp('Step 5: Loading temperature data')
        cd(fullfile(datapath,AnimalPath,recording.name))
        if ~exist('temperature.mat')
            if isempty(recording.ch_temp_type)
                recording.ch_temp_type = 'analog';
            end
            disp('Loading temperature data')
            LoadTemperature(recording.ch_temp,recording.ch_temp_type,pwd);
        end
        if exist(fullfile('SleepScore_Temperature.mat')) ~=2
            disp('Exporting Temperature reading for SleepScoring')
            SleepScoringScript(pwd)
        end
        Timer.duration(i,5) = toc(t_auto);
        Timer.event{5} = 'Exporting Temperature for SleepScoring';
    end
    
    % Calculating instantaneous theta
    if sum(StepsToPerform == 6)
        disp('Step 6: Calculating instantaneous theta')
        calcInstantaneousTheta(recording);
        Timer.duration(i,6) = toc(t_auto);
        Timer.event{6} = 'Calculating instantaneous theta';
    end
    % Optitrack: Loading position data
    if sum(StepsToPerform == 7)
        disp('Step 7: Loading Optitrack tracking data')
        if recording.ch_OptiTrack_sync
            if ~exist('Optitrack.mat')
                disp('Loading Optitrack tracking data')
                if  exist(recording.OptiTracktracking)
                    Optitrack = LoadOptitrack(recording.OptiTracktracking,1,recording.arena, 0,0);
                    save('Optitrack.mat','Optitrack')
                    clear Optitrack;
                else
                    disp('Optitrack tracking file do not exist')
                end
            end
        end
        Timer.duration(i,7) = toc(t_auto);
        Timer.event{7} = 'Optitrack: Loading position data';
    end
    
    % Loading digital channels
    if sum(StepsToPerform == 8)
        disp('Step 8: Loading digital channels')
        if ~exist('digitalchannels.mat')
            [digital_on,digital_off] = Process_IntanDigitalChannels('digitalin.dat');
            save('digitalchannels.mat','digital_on','digital_off');
            clear digital_on,digital_off;
        end
        Timer.duration(i,8) = toc(t_auto);
        Timer.event{8} = 'Loading digital channels';
    end
    
    % Running Kilosort
    if sum(StepsToPerform == 9)
        disp('Step 9: Running Kilosort')
        if exist(fullfile(SSDPath,['temp_wh.dat']))
            delete(fullfile(SSDPath,['temp_wh.dat']))
        end
        cd(fullfile(datapath,AnimalPath,recording.name))
        KiloSortSavePath = KiloSortWrapper;
        if ispc
            copyfile('C:\Users\peter\Dropbox\Buzsakilab Postdoc\Phy\Launch Phy.lnk', fullfile(KiloSortSavePath, 'Launch Phy.lnk'))
        end
        % Cleaning units for phy
        % disp('Cleaning phy units')
        % Kilosort_PostProcessing(KiloSortSavePath,5)
        Timer.duration(i,9) = toc(t_auto);
        Timer.event{9} = 'Running Kilosort';
    end
    
    % Copying Kilosort output to SSD together with the .dat file
    if sum(StepsToPerform == 10)
        disp('Step 10: Copying Kilosort output to SSD')
        tem = split(KiloSortSavePath,'\');
        KiloSortSaveFolder = tem{end};
        if exist(fullfile(SSDPath,recording.name))~=7
            mkdir(fullfile(SSDPath,recording.name))
        end
        if ~exist(fullfile(SSDPath,recording.name,[recording.name,'.dat']))
            copyfile(fullfile(datapath,AnimalPath,recording.name,[recording.name, '.dat']),fullfile(SSDPath,recording.name,[recording.name,'.dat']))
        end
        if ~exist(fullfile(SSDPath,recording.name,KiloSortSaveFolder))
            copyfile(KiloSortSavePath, fullfile(SSDPath,recording.name,KiloSortSaveFolder))
        end
        Timer.duration(i,10) = toc(t_auto);
        Timer.event{10} = 'Copying Kilosort output to SSD';
    end
    save('AutoProcessing.mat','Timer')
    disp(['Processing complete for ' recording.name])
    disp('')
end
Timer.duration(i,11) = toc(t_auto);
Timer.event{11} = 'Auto processing Complete';
save('AutoProcessing.mat','Timer')
disp('Auto processing Complete')
