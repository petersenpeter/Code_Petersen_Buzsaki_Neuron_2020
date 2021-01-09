% Path to data files
[~, name] = system('hostname');
name = cellstr(name);
name = name{1};
switch name
    case 'Peters-MacBook-Pro.local'
        disp([name, ' detected. Loading local paths...'])
        datapath = '/Users/peterpetersen/IntanData/';
        % \\peter\PeterP1\IntanData
    case 'PetersiMac.local'
        disp([name, ' detected. Loading local paths...'])
        datapath = '/Volumes/TurtlePower/DataBank/Buzsakilab/';
    case 'Peter'
        disp([name, ' detected. Loading local paths...']) 
        datapath = 'Z:\peterp03\IntanData\';
%     case 'PetersMacbook.local'
%         disp([name, ' detected. Loading local paths...'])
%         data_root_path = '/Users/peterpetersen/AmplipexDataProcessing/';
%         data_local_path = '/Users/peterpetersen/Google Drev/PhD/Projekter/LognormalDistribution';
%     case 'PetersMacMini.local'
%         disp([name, ' detected. Loading local paths...'])
%         data_root_path = '/Volumes/TurtlePower/DataBank/AmplipexDataProcessing/';
%         data_local_path = '/Volumes/TurtlePower/Google Drive/PhD/Projekter/LognormalDistribution';
    otherwise
        disp([cellstr(name) ' detected. No local paths found...'])
end
clear name

recordingID = 1;
recordings(recordingID).name = 'Peter_MS10_170310_175025_concat';
recordings(recordingID).concat_recordings = {'Peter_MS10_170310_175025','Peter_MS10_170310_183612','Peter_MS10_170310_184551','Peter_MS10_170310_190739'};
recordings(recordingID).concat_behavior_nb = [1];
recordings(recordingID).concat_sleep_nb = [2];
recordings(recordingID).concat_types = [2,1]; % 1: Sleep, 2: Circular Track, 3: Linear Track, 4: Wheelbox
recordings(recordingID).animal_id = 'MS10';
recordings(recordingID).arena = 'Homecage';
recordings(recordingID).manipulation = 'none';
recordings(recordingID).cooling_area = 'Medial Septum';
recordings(recordingID).notes = 'Cooling in homecage';
recordings(recordingID).time_frame = [0,28*60+10];
recordings(recordingID).nChannels = 64;
recordings(recordingID).sr = 20000;
recordings(recordingID).ch_theta = 53; % Base 1
recordings(recordingID).ch_medialseptum = []; % Base 1
recordings(recordingID).ch_hippocampus = [1:64]; % Base 1
recordings(recordingID).accelerometer = 0; % Accelerometer on preamplifier
recordings(recordingID).ch_wheel_pos = 0; % Analog Wheel (base 1)
recordings(recordingID).ch_temp = 1; % Analog Temperature data included (base 1)
recordings(recordingID).ch_peltier = 0; % Analog - Not applicaple (base 1)
recordings(recordingID).ch_fan_speed = 0; % Analog - Not applicaple (base 1)
recordings(recordingID).ch_camera_sync = 3; % Digital pulses (base 1)
recordings(recordingID).ch_OptiTrack_sync = 0; % Digital pulses (base 1)
recordings(recordingID).ch_CoolingPulses = 0; % Digital pulses (base 1)
recordings(recordingID).Cameratracking.Behavior = {'Basler acA1300-200uc (21965891)_20170310_175031417.avi','Basler acA1300-200uc (21965891)_20170310_183618980.avi','Basler acA1300-200uc (21965891)_20170310_184555309.avi'};
recordings(recordingID).Cameratracking.PostBehavior = 'Basler acA1300-200uc (21965891)_20170310_190750660.avi';
recordings(recordingID).OptiTracktracking = 'Take 2017-03-10 05.50.36 PM.csv';
recordings(recordingID).coolterm = '';
recordings(recordingID).cooling_onsets = []; % Timestamp noted manually'
recordings(recordingID).cooling_offsets = []; % Sessions removed: none
recordings(recordingID).SpikeSorting.completed = 0;
recordings(recordingID).SpikeSorting.method = 'Phy'; % Phy (.npy files) e.g.: SpikingCircus, Kilosort. Klustakwik (.clu,.res): , KlustaViewer ()
recordings(recordingID).SpikeSorting.path = '';
recordings(recordingID).SpikeSorting.shanks = 1;
recordings(recordingID).maze.radius_in = 96.5/2;
recordings(recordingID).maze.radius_out =  116.5/2;
recordings(recordingID).maze.arm_half_width = 4;
recordings(recordingID).maze.cross_radii = 47.9;