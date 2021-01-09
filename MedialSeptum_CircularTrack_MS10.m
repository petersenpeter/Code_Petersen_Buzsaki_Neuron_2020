% DOCID = '1WBqEo0OM5qdqmAD_7cGsJe0iyGf6hjgPXy-V2VlVY9c'
% result = GetGoogleSpreadsheet(DOCID); 
% Medial Septum Circular Track
clear all
Recordings_MedialSeptum
id = 58; % 63 % Processed datasets in MS12: 78,79,80,81
recording = recordings(id);
cd([datapath, recording.name(1:6) recording.rat_id '\' recording.name, '\'])
Intan_rec_info = read_Intan_RHD2000_file_Peter([datapath, recording.name(1:6) recording.rat_id '\' recording.name, '\']);
fname = [recording.name '.dat'];
nbChan = size(Intan_rec_info.amplifier_channels,2);
cooling.cooling = recording.cooling;
cooling.onsets = recording.cooling_onsets;
cooling.offsets = recording.cooling_offsets;

sr = Intan_rec_info.frequency_parameters.amplifier_sample_rate;
time_frame = recording.time_frame;
lfp_periods = 30*60; % in seconds
ch_lfp = recording.ch_lfp;
ch_medialseptum = recording.ch_medialseptum;
ch_hippocampus = recording.ch_hippocampus;
inputs.ch_wheel_pos = recording.ch_wheel_pos; % Wheel channel (base 1)
inputs.ch_temp = recording.ch_temp; % Temperature data included (base 1)
inputs.ch_peltier_voltage = recording.ch_peltier; % Peltier channel (base 1)
inputs.ch_fan_speed = recording.ch_fan_speed; % Fan speed channel (base 1)
inputs.ch_camera_sync = recording.ch_camera_sync;
inputs.ch_OptiTrack_sync = recording.ch_OptiTrack_sync;
inputs.ch_CoolingPulses = recording.ch_CoolingPulses;
maze = recording.maze;
recording.sr_lfp = sr/16;
% track_boundaries = recording.track_boundaries;
arena = recording.arena;
% nb_lfp_periods = ceil((time_frame(2)-time_frame(1))/lfp_periods);
temp_ = dir(fname);
recording_length = round(temp_.bytes/sr/nbChan/2)-1;
animal = [];

% Camera tracking: Loading position data
disp('1. Loading Camera tracking data')
if inputs.ch_camera_sync
    if ~exist('Camera.mat')
        colors = [1,3]; % RGB based
        Camera = ImportCameraData(recording.Cameratracking,pwd,colors,arena);
    else
        load('Camera.mat');
    end
end
% Optitrack: Loading position data
disp('2. Loading Optitrack tracking data')
if inputs.ch_OptiTrack_sync
    if ~exist('Optitrack.mat')
        Optitrack = LoadOptitrack(recording.OptiTracktracking,1,arena,0,0)
        save('Optitrack.mat','Optitrack')
    else load('Optitrack.mat'); end
end
if ~isempty(recording.OptiTracktracking_offset)
    Optitrack.position3D = Optitrack.position3D + recording.OptiTracktracking_offset';
end

% Loading digital inputs
disp('3. Loading digital inputs')
if ~exist('digitalchannels.mat')
    [digital_on,digital_off] = Process_IntanDigitalChannels('digitalin.dat');
    save('digitalchannels.mat','digital_on','digital_off');
%     if length(recording.concat_recordings) == 0
%         [digital_on,digital_off] = Process_IntanDigitalChannels('digitalin.dat');
%         save('digitalchannels.mat','digital_on','digital_off');
%     else
%         %fullfile([datapath, recording.name(1:6) recording.rat_id], recording.concat_recordings(recording.concat_behavior_nb), 'digitalin.dat');
%         [digital_on,digital_off] = Process_IntanDigitalChannels('digitalin.dat');
%         save('digitalchannels.mat','digital_on','digital_off');
%     end
else
    load('digitalchannels.mat');
end

prebehaviortime = 0;
if recording.concat_behavior_nb > 0
    prebehaviortime = 0;
    if recording.concat_behavior_nb > 1
    for i = 1:recording.concat_behavior_nb-1
        fullpath = fullfile([datapath, recording.name(1:6) recording.rat_id], recording.concat_recordings{i}, [recording.concat_recordings{i}, '.dat']);
        temp2_ = dir(fullpath);
        prebehaviortime = prebehaviortime + temp2_.bytes/nbChan/2/sr;
    end
    end
    i = recording.concat_behavior_nb;
    fullpath = fullfile([datapath, recording.name(1:6) recording.rat_id], recording.concat_recordings{i}, [recording.concat_recordings{i}, '.dat']);
    temp2_ = dir(fullpath);
    behaviortime = temp2_.bytes/nbChan/2/sr;
else
    temp_ = dir(fname);
    behaviortime = temp_.bytes/nbChan/2/sr;
end

disp('4. Calculating behavior')
if inputs.ch_camera_sync ~= 0
    animal.sr = Camera.framerate;
    if length(recording.concat_recordings) > 0
        camera_pulses = find(digital_on{inputs.ch_camera_sync}/sr > prebehaviortime & digital_on{inputs.ch_camera_sync}/sr < prebehaviortime + recording_length);
        pulse_corrected = camera_pulses;
        pulse_corrected(find(diff(pulse_corrected)<mean(diff(pulse_corrected))/2)) = [];
        disp(['Camera pulses detected: ' num2str(camera_pulses)])
        disp(['Corrected camera pulses detected: ' num2str(pulse_corrected)])
        animal.time = digital_on{inputs.ch_camera_sync}(pulse_corrected(1:Camera.frames))'/sr;
        Camera.onset = min(digital_on{inputs.ch_camera_sync}(pulse_corrected(1)),digital_off{inputs.ch_camera_sync}(pulse_corrected(1)))/sr;
        Camera.offset = (temp_.bytes/sr/nbChan/2)-max(digital_on{inputs.ch_camera_sync}(end),digital_off{inputs.ch_camera_sync}(end))/sr;
    else
        camera_pulses = digital_on{inputs.ch_camera_sync};
        pulse_corrected = camera_pulses;
        pulse_corrected(find(diff(pulse_corrected)<mean(diff(pulse_corrected))/2)) = [];
        animal.time = pulse_corrected(1:Camera.frames)'/sr;
        Camera.onset = min(digital_on{inputs.ch_camera_sync}(1),digital_off{inputs.ch_camera_sync}(1))/sr;
        Camera.offset = (temp_.bytes/sr/nbChan/2)-max(digital_on{inputs.ch_camera_sync}(end),digital_off{inputs.ch_camera_sync}(end))/sr;
    end
    if length(digital_on{inputs.ch_camera_sync})-Camera.frames ~= 0
        disp(['Camera frames and TTL pulses dont add up:'])
        disp(['Number of Camera TTL pulses: ' num2str(length(digital_on{inputs.ch_camera_sync}))])
        disp(['Number of Camera frames: ' num2str(Camera.frames)])
        disp(['Difference: ' num2str(length(digital_on{inputs.ch_camera_sync})-Camera.frames) ' frames'])
    end
    animal.pos  = Optitrack.position3D;
    animal.acceleration = [0,Camera.framerate*sqrt(sum((diff(Camera.pos',2).^2),2))',0];
    animal.pos(:,animal.acceleration>100) = nan;
    animal.speed = [0,Camera.framerate*sqrt(sum((diff(Camera.pos').^2),2))'];
    gausswin_size = animal.sr/4;
    for i = 1:2
        % animal.pos(i,:) = medfilt1(animal.pos(i,:),5,'omitnan');
        animal.pos(i,:) = nanconv(animal.pos(i,:),gausswin(gausswin_size)','edge');
    end
    animal.speed = medfilt1(animal.speed,5,'omitnan');
end
if inputs.ch_OptiTrack_sync ~= 0
    Optitrack.onset = min(digital_on{inputs.ch_OptiTrack_sync}(1),digital_off{inputs.ch_OptiTrack_sync}(1))/sr;
    Optitrack.offset = (temp_.bytes/sr/nbChan/2)-max(digital_on{inputs.ch_OptiTrack_sync}(end),digital_off{inputs.ch_OptiTrack_sync}(end))/sr;
    animal.sr = Optitrack.FrameRate; % 100
    animal.time = digital_on{inputs.ch_OptiTrack_sync}'/sr;
    animal.pos  = Optitrack.position3D;
    animal.pos(:,find(animal.pos(2,:)>70)) = 0;
    gausswin_size = animal.sr/2;
    for i = 1:3
    animal.pos(i,:) = medfilt1(animal.pos(i,:),5);
    animal.pos(i,:) = nanconv(animal.pos(i,:),gausswin(gausswin_size)','edge');
    end
    animal.speed  = Optitrack.animal_speed3D;
end

% if animal.sr ~= sr_lfp
%     animal.sr2 = sr_lfp;
%     % speed = interp1(animal.time,animal.speed,(1:length(signal_filtered))/sr_lfp);
%     % pos = interp1(animal.time,animal.pos',(1:length(signal_filtered))/sr_lfp);
%     animal.time2 = [animal.time(1):1/sr_lfp:animal.time(end)];
%     animal.pos2 = interp1(animal.time,animal.pos',animal.time2)';
%     animal.speed2 = interp1(animal.time,animal.speed',animal.time2);
% else 
%     animal.sr2 = animal.sr;
%     animal.time2 = animal.time;
%     animal.pos2 = animal.pos;
%     animal.speed2 = animal.speed;
% end

[animal.polar_theta,animal.polar_rho] = cart2pol(animal.pos(2,:),animal.pos(1,:));
animal.polar_theta = animal.polar_theta*maze.radius_in;
animal.polar_rho_limits = [44,65];
animal.polar_theta_limits = [-2.8,2.8]*maze.radius_in;
animal.circularpart = find(animal.polar_theta > animal.polar_theta_limits(1) & animal.polar_theta < animal.polar_theta_limits(2) & animal.polar_rho > animal.polar_rho_limits(1) & animal.polar_rho < animal.polar_rho_limits(2));
animal.pos_x_limits = [-10,10];
animal.pos_y_limits = [-40,45];
animal.centralarmpart = find(animal.pos(1,:) > animal.pos_x_limits(1) & animal.pos(1,:) < animal.pos_x_limits(2) & animal.pos(2,:) > animal.pos_y_limits(1) & animal.pos(2,:) < animal.pos_y_limits(2));
animal.arm = double(animal.pos(1,:) > animal.pos_x_limits(1) & animal.pos(1,:) < animal.pos_x_limits(2) & animal.pos(2,:) > animal.pos_y_limits(1) & animal.pos(2,:) < animal.pos_y_limits(2));
animal.rim = double(animal.polar_rho > animal.polar_rho_limits(1) & animal.polar_rho < animal.polar_rho_limits(2) & animal.polar_theta > animal.polar_theta_limits(1) & animal.polar_theta < animal.polar_theta_limits(2));
animal.maze = maze;
animal.state_labels = {'Pre','Cooling','Post'};

% Loading Temperature data
if inputs.ch_temp >0
    disp('5. Loading Temperature data')
    if ~exist('temperature.mat')
        num_channels = length(Intan_rec_info.board_adc_channels); % ADC input info from header file
        fileinfo = dir('analogin.dat');
        num_samples = fileinfo.bytes/(num_channels * 2); % uint16 = 2 bytes
        fid = fopen('analogin.dat', 'r'); v = fread(fid, [num_channels, num_samples], 'uint16'); fclose(fid);
        v = v * 0.000050354; % convert to volts
        v_downsample = mean(reshape(v(inputs.ch_temp,1:end-rem(length(v),16)),16,[]));
        %v_downsample(v_downsample<1.25) = 1.25;
        temperature = [];
        temperature.temp = (v_downsample-1.25)/0.005;
        temperature.temp(temperature.temp > 50 | temperature.temp < 0) = nan;
        temperature.temp = nanconv(temperature.temp,gausswin(200)','edge');
        temperature.sr = recording.sr_lfp;
        temperature.time = [1:length(temperature.temp)]/recording.sr_lfp;
        save('temperature.mat','temperature')
        clear v_downsample v;
        animal.temperature = interp1(temperature.time,temperature.temp,animal.time);
    else
        load('temperature.mat');
        animal.temperature = interp1(temperature.time,temperature.temp,animal.time);
    end
else
    disp('No temperature data available')
    temperature = [];
end
disp('6. Creating Cooling structure')

if inputs.ch_temp ~= 0
    temp_range = [32,34];% temp_1 defines the upper limit on cooling, temp_2 the lower limit on no cooling
    test = find(diff(temperature.temp < temp_range(1),2)== 1);
    test(diff(test)<10*temperature.sr)=[];
    cooling.onsets = temperature.time(test(temperature.temp(test+1)-temperature.temp(test)<0));
    cooling.offsets = temperature.time(test(temperature.temp(test+1)-temperature.temp(test)>0));
    cooling.cooling = [cooling.onsets;cooling.offsets];
    cooling.cooling2 = [cooling.onsets-20;cooling.offsets];
    cooling.nocooling = reshape([prebehaviortime;cooling.cooling2(:);prebehaviortime+behaviortime],[2,size(cooling.cooling2,2)+1]);
elseif inputs.ch_CoolingPulses ~= 0
    cooling.onsets = digital_on{inputs.ch_CoolingPulses}/sr;
    cooling.offsets = cooling.onsets + 12*60;
    cooling.cooling = [cooling.onsets(1)+10;cooling.offsets(1)];
    cooling.nocooling = [[1,cooling.onsets'];[cooling.offsets'+120,behaviortime]]';
else
    cooling.cooling = [cooling.onsets(1)+10;cooling.offsets(1)]+prebehaviortime;
    cooling.nocooling = [[1,cooling.onsets(1)]+prebehaviortime;[cooling.offsets(1)+120,behaviortime]+prebehaviortime]';
end

% Separating left and right trials
disp('7. Defining trials for the behavior')
maze.boundary{1} = [0,40];
maze.boundary{2} = [0,25];
maze.boundary{3} = [-15,40];
maze.boundary{4} = [15,40];
maze.boundary{5} = [maze.radius_in-3.25,150/180*pi*maze.radius_in];
[trials,animal] = trials_thetamaze(animal, maze,[],cooling);
trials.labels = {'Left','Right'};
trials.total = length(trials.start);

temp1 = 0; temp2 = 0;
for j = 1:size(cooling.cooling,2)
    temp1 = temp1 + length(find(animal.time(trials.start(trials.error)) > cooling.cooling(1,j) & animal.time(trials.start(trials.error)) < cooling.cooling(2,j)));
    temp2 = temp2 + length(find(animal.time(trials.start) > cooling.cooling(1,j) & animal.time(trials.start) < cooling.cooling(2,j)));
end
trials.Cooling_error_ratio = temp1/temp2;
trials.NoCooling_error_ratio_before = length(find(animal.time(trials.start(trials.error)) < cooling.cooling(1)))/length(find(animal.time(trials.start) < cooling.cooling(1)));
temp1 = 0; temp2 = 0;
for j = 1:size(cooling.cooling,2)-1
    temp1 = temp1 + length(find(animal.time(trials.start(trials.error)) > cooling.cooling(2,j)+20 &  animal.time(trials.start(trials.error)) < cooling.cooling(1,j+1)));
    temp2 = temp2 + length(find(animal.time(trials.start) > cooling.cooling(2,j)+20 & animal.time(trials.start) < cooling.cooling(1,j+1)+20));
end
temp1 = temp1 + length(find(animal.time(trials.start(trials.error)) > cooling.cooling(2,end)+20));
temp2 = temp2 + length(find(animal.time(trials.start) > cooling.cooling(2,end)+20));
trials.NoCooling_error_ratio_after = temp1/temp2;

for i = 1:trials.total
	trials.temperature(i) = mean(temperature.temp(round(animal.time(trials.start(i))*temperature.sr):round(animal.time(trials.end(i))*temperature.sr)));
end

% Calculating the instantaneous theta frequency
disp('8. Loading instantaneous theta')
theta = [];
theta.sr = recording.sr_lfp;
theta.ch_theta = recording.ch_theta;
[signal_phase,signal_phase2,signal_freq] = calcInstantaneousTheta(recording);
theta.phase = signal_phase;
theta.phase2 = signal_phase2;
theta.freq = signal_freq;
theta.sr_freq = 10;
clear signal_phase signal_phase2

disp('Plotting Maze and behavior')
figure,
subplot(3,2,[1,3])
plot3(animal.pos(1,:),animal.pos(2,:),animal.pos(3,:),'-k','markersize',8), hold on, plot_ThetaMaze(maze)
plot3(animal.pos(1,trials.left),animal.pos(2,trials.left),animal.pos(3,trials.left),'.r','markersize',8), hold on
plot3(animal.pos(1,trials.right),animal.pos(2,trials.right),animal.pos(3,trials.right),'.b','markersize',8), title('Cartesian Coordinates')
xlabel('x (cm)'),ylabel('y (cm)')
axis equal
xlim(maze.radius_out*[-1.2,1.2]),ylim(maze.radius_out*[-1.4,1.2])

subplot(2,2,2)
bins_speed = [0:5:120];
plot(animal.time,animal.speed,'k'), axis tight, title('Speed of the animal'), hold on, 
plot(cooling.cooling,[0,0],'color','blue','linewidth',2)
plot(cooling.nocooling,[0,0],'color','red','linewidth',2)
gridxy(animal.time(trials.start),'color','g')
gridxy(animal.time(trials.start(trials.error)),'color','m','linewidth',2)
legend({'Error trials','All trials','Speed','Cooling','NoCooling'})
xlabel('Time (s)'), ylabel('Speed (cm/s)')

subplot(3,2,5)
cooling.times_cooling = find(animal.time(trials.all)> cooling.cooling(1) & animal.time(trials.all) < cooling.cooling(2));
cooling.times_nocooling = find(animal.time(trials.all)< cooling.cooling(1) | animal.time(trials.all) > cooling.cooling(2));
histogram(animal.speed(trials.all(cooling.times_cooling)),'BinEdges',bins_speed,'Normalization','probability'), hold on
histogram(animal.speed(trials.all(cooling.times_nocooling)),'BinEdges',bins_speed,'Normalization','probability'),
legend({'Cooling','No Cooling'})
xlabel('Speed (cm/s)'), ylabel('Probability'), title(['Speed during the trials (Total: ' num2str(length(trials.start))  ' trials)'])

subplot(4,2,8)
bar(1, trials.NoCooling_error_ratio_before, 'red'), hold on
bar(2, trials.Cooling_error_ratio, 'blue')
bar(3, trials.NoCooling_error_ratio_after, 'red'), hold on
xticks([1, 2, 3]), xticklabels({'Pre Cooling','Cooling','Post cooling'}),ylabel('Percentage of errors'),title('Error trials (%)'),axis tight,
xlim([0,4]),ylim([0,0.3])
subplot(4,2,6)
plot(temperature.time,temperature.temp), xlim([prebehaviortime,prebehaviortime+behaviortime]), title('Temperature')
ylim([20,40])


% % Position on the rim defined in polar coordinates
disp('Plotting polar coordinates of tracking')
figure,
subplot(2,2,1)
plot(animal.pos(1,:),animal.pos(2,:),'.k','markersize',8), hold on
plot(animal.pos(1,trials.left),animal.pos(2,trials.left),'.r','markersize',8)
plot(animal.pos(1,trials.right),animal.pos(2,trials.right),'.b','markersize',8), title('Cartesian Coordinates')
xlabel('x (cm)'),ylabel('y (cm)')
plot([animal.pos_x_limits(1) animal.pos_x_limits(2) animal.pos_x_limits(2) animal.pos_x_limits(1) animal.pos_x_limits(1)],[animal.pos_y_limits(1) animal.pos_y_limits(1) animal.pos_y_limits(2) animal.pos_y_limits(2) animal.pos_y_limits(1)],'c','linewidth',2)
plot_ThetaMaze(maze)
axis equal
xlim([-70,70]),ylim([-70,70]),

subplot(2,2,[3,4])
plot(animal.polar_theta,animal.polar_rho,'.k','markersize',8), hold on
plot(animal.polar_theta(trials.left),animal.polar_rho(trials.left),'.r','markersize',8)
plot(animal.polar_theta(trials.right),animal.polar_rho(trials.right),'.b','markersize',8), title('Polar Coordinates')
plot([animal.polar_theta_limits(1),animal.polar_theta_limits(2) animal.polar_theta_limits(2) animal.polar_theta_limits(1) animal.polar_theta_limits(1)],[animal.polar_rho_limits(1) animal.polar_rho_limits(1) animal.polar_rho_limits(2) animal.polar_rho_limits(2) animal.polar_rho_limits(1)],'g','linewidth',2)
xlim(50*[-pi,pi]),ylim([0,70]),
xlabel('Circular position (cm)'),ylabel('rho (cm)')
subplot(2,2,2)
plot(animal.pos(1,:),animal.pos(2,:),'.k','markersize',8), hold on
plot(animal.pos(1,animal.circularpart),animal.pos(2,animal.circularpart),'.g','markersize',8)
plot(animal.pos(1,animal.centralarmpart),animal.pos(2,animal.centralarmpart),'.c','markersize',8)
xlabel('x (cm)'),ylabel('y (cm)'), title('Rim and Center arm defined')
plot_ThetaMaze(maze)
axis equal
xlim([-70,70]),ylim([-70,70]),
save('animal.mat','animal')

% temperature vs Theta frequency vs temperature
disp('Plotting Theta vs Temperature')
animal.theta = interp1([1:length(theta.freq)]/theta.sr_freq,theta.freq,animal.time);
[hist_data,xlimit,ylimit] = histcounts2(animal.temperature(animal.speed > 5),animal.theta(animal.speed > 5),40);
H = gausswin(19)'/sum(gausswin(11));
hist_data = zscore(filter2(H,hist_data)');
[xcf,lags,bounds] = crosscorr(animal.temperature(animal.speed > 5 & ~isnan(animal.temperature)),animal.theta(animal.speed > 5 & ~isnan(animal.temperature)),10000);
[~,i] = max(xcf);
peak = (i-(length(lags)-1)/2)/animal.sr;
figure, 
subplot(1,3,1:2), imagesc(xlimit,ylimit, hist_data),set(gca,'YDir','normal'),title('Theta vs Temperature'),xlabel('Temperature [Celcius]'),ylabel('Frequency [Hz]')
subplot(1,3,3), plot(lags/animal.sr,xcf,'.-b'), hold on, plot([lags(1),lags(end)]/animal.sr,[bounds;bounds],'r'),title('Offset'),xlabel('Time [s]'),ylabel('Correlation'), hold on
plot(peak,xcf(i),'or','linewidth',2), text(peak,0.9*xcf(i),['Delay: ' num2str(peak,3), 's'],'HorizontalAlignment','Center'), xlim([lags(1),lags(end)]/animal.sr)

disp('9. Finished loading the recording')

%% % Plotting theta frequency vs running speed
plot_ThetaVsSpeed(recording,animal,cooling);

%% % Units
Recordings_MedialSeptum
SpikeSorting_method = recording.SpikeSorting.method; % Phy (.npy files) e.g.: SpikingCircus, Kilosort. Klustakwik (.clu,.res,.spk): , KlustaSuite ()
SpikeSorting_path = recording.SpikeSorting.path;
shanks = recording.SpikeSorting.shanks;
units = loadClusteringData(recording.name,SpikeSorting_method,SpikeSorting_path,shanks);
if ~exist([recording.name, '.eeg'])
    disp('Creating EEG file')
    downsample_dat_to_eeg(recording.name,pwd);
    % movefile('amplifier.eeg',[recording.name '.eeg'])
    % copyfile('amplifier.xml',[recording.name '.xml'])
end

disp('Plotting instantaneous theta frequency for all units')
% figure, plot(signal_phase)
for i = 1:size(units,2)
    % units(i).ts = units(i).ts(units(i).ts/sr < length(signal_phase)/sr_eeg);
    units(i).total = length(units(i).ts);
    units(i).ts_eeg = ceil(units(i).ts/16);
    units(i).theta_phase = theta.phase(units(i).ts_eeg);
    units(i).theta_phase2 = theta.phase2(units(i).ts_eeg);
    units(i).theta_freq = interp1([1:length(theta.freq)]/theta.sr_freq,theta.freq,units(i).ts/sr);
    units(i).speed = interp1(animal.time,animal.speed,units(i).ts/sr);
    units(i).pos = interp1(animal.time,animal.pos',units(i).ts/sr)';
    units(i).polar_theta = interp1(animal.time,animal.polar_theta,units(i).ts/sr);
    units(i).polar_rho = interp1(animal.time,animal.polar_rho,units(i).ts/sr);
    units(i).arm = zeros(1,length(units(i).ts));
    units(i).arm(units(i).pos(1,:) > animal.pos_x_limits(1) & units(i).pos(1,:) < animal.pos_x_limits(2) & units(i).pos(2,:) > animal.pos_y_limits(1) & units(i).pos(2,:) < animal.pos_y_limits(2)) = 1;
    units(i).rim = zeros(1,length(units(i).ts));
    units(i).rim(units(i).polar_rho > animal.polar_rho_limits(1) & units(i).polar_rho < animal.polar_rho_limits(2) & units(i).polar_theta > animal.polar_theta_limits(1) & units(i).polar_theta < animal.polar_theta_limits(2)) = 1;
    %units(i).trials = interp1(animal.time+prebehaviortime/sr,trials.trials,units(i).ts/sr,'nearest');
    units(i).state = interp1(animal.time,trials.state,units(i).ts/sr,'nearest');
    units(i).temperature = temperature.temp(units(i).ts_eeg);
    units(i).trials = interp1(animal.time,trials.trials2,units(i).ts/sr,'nearest');
end
if isfield(recording.SpikeSorting,'polar_theta_placecells')
    for i = 1:size(units,2)  %size(recording.SpikeSorting.polar_theta_placecells,2)
        if units(i).kwik_id <= length(recording.SpikeSorting.polar_theta_placecells)
            if ~isempty(recording.SpikeSorting.polar_theta_placecells{units(i).kwik_id})
                units(i).PhasePrecession.placefields_polar_theta = recording.SpikeSorting.polar_theta_placecells{units(i).kwik_id};
            end
        end
    end
end
if isfield(recording.SpikeSorting,'center_arm_placecells')
    for i = 1:size(units,2)  %size(recording.SpikeSorting.polar_theta_placecells,2)
        if units(i).kwik_id <= length(recording.SpikeSorting.center_arm_placecells)
            if ~isempty(recording.SpikeSorting.center_arm_placecells{units(i).kwik_id})
                units(i).PhasePrecession.placefields_center_arm = recording.SpikeSorting.center_arm_placecells{units(i).kwik_id};
            end
        end
    end
end
disp('done')

disp('Plotting Phase precession')
figure, 
for i = 1:size(units,2)
    subplot(1,3,1)
    plot(units(i).pos(1,:),units(i).pos(2,:),'.','markersize',8), hold on
    legend({'1','2','3','4','5','6','7','8','9'})
	plot_ThetaMaze(maze)
    axis equal
    xlim([-65,65]),ylim([-65,65]),
    subplot(1,3,2)
    histogram(units(i).polar_theta,'Normalization','probability'), hold on
    subplot(1,3,3)
    plot(units(i).polar_theta,units(i).polar_rho,'.','markersize',8), hold on
end
%% % Plotting Phase precession
behavior = animal; % Animal Strucutre
behavior.speed_th = 5;
behavior.rim = animal.rim;
behavior.state = trials.state; 
behavior.pos = animal.polar_theta;
behavior.pos_limits = animal.polar_theta_limits;
% behavior.optogenetics.pos = optogenetics.polar_theta(optogenetics.rim);
% behavior.optogenetics.trial = optogenetics.trials(optogenetics.rim);
units2 = [];
for i = 1:size(units,2)
    units2(i).ts = units(i).ts;
    units2(i).kwik_id = units(i).kwik_id;
    units2(i).total= units(i).total;
end
% Polar plot - PHASE PRECESSION
if isfield(recording.SpikeSorting,'polar_theta_placecells')
    for i = 1:size(units2,2)  %size(recording.SpikeSorting.polar_theta_placecells,2)
        if units2(i).kwik_id <= length(recording.SpikeSorting.polar_theta_placecells)
            if ~isempty(recording.SpikeSorting.polar_theta_placecells{units2(i).kwik_id})
                units2(i).PhasePrecession = recording.SpikeSorting.polar_theta_placecells{units2(i).kwik_id};
            end
        end
    end
end
% units2(1).PhasePrecession = [];
PhasePrecessionSlope1 = plot_FiringRateMap(behavior,units2,trials,theta,sr);

% Firing rate on center arm
behavior = animal;
behavior.speed_th = 5;
behavior.state = trials.state;
behavior.('pos') = animal.pos(2,:);
behavior.rim = animal.arm;
behavior.pos_limits = animal.pos_y_limits;
% behavior.optogenetics.pos = optogenetics.pos(2,optogenetics.arm);
% behavior.optogenetics.trial = optogenetics.trials(optogenetics.arm);
units2 = [];
for i = 1:size(units,2)
    units2(i).ts = units(i).ts;
    units2(i).kwik_id = units(i).kwik_id;
    units2(i).total= units(i).total;
end
% Center arm - PHASE PRECESSION
if isfield(recording.SpikeSorting,'center_arm_placecells')
    for i = 1:size(units2,2)  %size(recording.SpikeSorting.polar_theta_placecells,2)
        if units2(i).kwik_id <= length(recording.SpikeSorting.center_arm_placecells)
            if ~isempty(recording.SpikeSorting.center_arm_placecells{units2(i).kwik_id})
                units2(i).PhasePrecession = recording.SpikeSorting.center_arm_placecells{units2(i).kwik_id};
            end
        end
    end
end
PhasePrecessionSlope2 = plot_FiringRateMap(behavior,units2,trials,theta,sr);
%%
m = 1;
slope1 = [];
mean1 = []; median1 = []; std1 = []; skewness1 = []; % Position
mean2 = []; median2 = []; std2 = []; skewness2 = []; % Theta phase

SpikeCount1 = []; Slope1 = []; R1 = [];

PhasePrecession = PhasePrecessionSlope1;
for i = 1:length(PhasePrecession)
    if ~isempty(PhasePrecession(i).Slope)
        for j = 1:size(PhasePrecession(i).Slope,2)
            elements = length(PhasePrecession(i).Slope{j}.slope1);
            slope1(j,m:m+elements-1) = PhasePrecession(i).Slope{j}.slope1;
            mean1(j,m:m+elements-1) = PhasePrecession(i).Slope{j}.mean;
            std1(j,m:m+elements-1) = PhasePrecession(i).Slope{j}.std;
            skewness1(j,m:m+elements-1) = PhasePrecession(i).Slope{j}.skewness;
            mean2(j,m:m+elements-1) = PhasePrecession(i).Slope{j}.mean_phase;
            std2(j,m:m+elements-1) = PhasePrecession(i).Slope{j}.std_phase;
            skewness2(j,m:m+elements-1) = PhasePrecession(i).Slope{j}.skewness_phase;
        end
        SpikeCount1(m:m+elements-1,1:size(PhasePrecession(i).spikesperTrials,2)) = PhasePrecession(i).spikesperTrials;
        Slope1(m:m+elements-1,1:size(PhasePrecession(i).spikesperTrials,2)) = PhasePrecession(i).PhasePrecessionTrials.slope2;
        R1(m:m+elements-1,1:size(PhasePrecession(i).spikesperTrials,2)) = PhasePrecession(i).PhasePrecessionTrials.R2;
        m = m+elements;
    end
end
if ~isempty(PhasePrecessionSlope2)
PhasePrecession = PhasePrecessionSlope2;
for i = 1:length(PhasePrecession)
    if ~isempty(PhasePrecession(i).Slope)
        for j = 1:size(PhasePrecession(i).Slope,2)
            elements = length(PhasePrecession(i).Slope{j}.slope1);
            slope1(j,m:m+elements-1) = PhasePrecession(i).Slope{j}.slope1;
            mean1(j,m:m+elements-1) = PhasePrecession(i).Slope{j}.mean;
            std1(j,m:m+elements-1) = PhasePrecession(i).Slope{j}.std;
            skewness1(j,m:m+elements-1) = PhasePrecession(i).Slope{j}.skewness;
            mean2(j,m:m+elements-1) = PhasePrecession(i).Slope{j}.mean_phase;
            std2(j,m:m+elements-1) = PhasePrecession(i).Slope{j}.std_phase;
            skewness2(j,m:m+elements-1) = PhasePrecession(i).Slope{j}.skewness_phase;
        end
        SpikeCount1(m:m+elements-1,1:size(PhasePrecession(i).spikesperTrials,2)) = PhasePrecession(i).spikesperTrials;
        Slope1(m:m+elements-1,1:size(PhasePrecession(i).spikesperTrials,2)) = PhasePrecession(i).PhasePrecessionTrials.slope2;
        R1(m:m+elements-1,1:size(PhasePrecession(i).spikesperTrials,2)) = PhasePrecession(i).PhasePrecessionTrials.R2;
        m = m+elements;
    end
end
end
SpikeCount1(~SpikeCount1) = nan;
SpikeCount1 = (SpikeCount1-nanmean(SpikeCount1,2))./nanstd(SpikeCount1,1,2);
Slope1(~Slope1) = nan;
Slope1(abs(Slope1)>0.06) = nan;
Slope1(~R1) = nan;
Slope1(R1<0.4) = nan;
Slope1(find(nanmean(Slope1,2)<0),:) = -Slope1(find(nanmean(Slope1,2)<0),:);
% R1(~R1) = nan;
% R1(R1>0.06) = nan;


figure,
subplot(2,2,1), plot(abs(slope1)-mean(abs(slope1))), 
title('Phase precession'), 
xticks([1:size(PhasePrecession(i).Slope,2)])
xticklabels(animal.state_labels)
subplot(2,2,2),
plot(slope1(1,:)./slope1(2,:),slope1(3,:)./slope1(2,:),'o'), axis equal, hold on, plot([0,2],[0,2]), plot([1,1],[0,2]), plot([0,2],[1,1])
title('Ratio'), xlabel('Pre/Cooling'), ylabel('Post/Cooling')
subplot(2,2,3), 
boxplot((abs(slope1)-mean(abs(slope1)))')
title('Phase precession boxplot'), 
xticks([1:size(PhasePrecession(i).Slope,2)])
xticklabels(animal.state_labels)

subplot(2,2,4), 
boxplot([slope1(1,:)./slope1(2,:);slope1(3,:)./slope1(2,:)]')
title('Ratios'), 
xticks([1,2])
xticklabels({'Pre/Cooling','Post/Cooling'})

% Position
figure
subplot(2,6,1), plot(mean1-mean(mean1)), 
title('Mean'), 
xticks([1:size(PhasePrecession(i).Slope,2)])
xticklabels(animal.state_labels)
subplot(2,6,7), boxplot((mean1-mean(mean1))')
xticks([1:size(PhasePrecession(i).Slope,2)])
xticklabels(animal.state_labels)
title('Mean')
subplot(2,6,2), plot(std1-mean(std1)), 
title('Std'), 
xticks([1:size(PhasePrecession(i).Slope,2)])
xticklabels(animal.state_labels)
subplot(2,6,8), boxplot((std1-mean(std1))')
xticks([1:size(PhasePrecession(i).Slope,2)])
xticklabels(animal.state_labels)
title('Std')
subplot(2,6,3), plot(skewness1-mean(skewness1)), 
title('Skewness'), 
xticks([1:size(PhasePrecession(i).Slope,2)])
xticklabels(animal.state_labels)
subplot(2,6,9), boxplot((skewness1-mean(skewness1))')
xticks([1:size(PhasePrecession(i).Slope,2)])
xticklabels(animal.state_labels)
title('Skewness')

% PHASE 
subplot(2,6,4), plot(mean2-mean(mean2)), 
title('Mean phase'), 
xticks([1:size(PhasePrecession(i).Slope,2)])
xticklabels(animal.state_labels)
subplot(2,6,10), boxplot((mean2-mean(mean2))')
xticks([1:size(PhasePrecession(i).Slope,2)])
xticklabels(animal.state_labels)
title('Mean phase')
subplot(2,6,5), plot(std2-mean(std2)), 
title('Std phase'), 
xticks([1:size(PhasePrecession(i).Slope,2)])
xticklabels(animal.state_labels)
subplot(2,6,11), boxplot((std2-mean(std2))')
xticks([1:size(PhasePrecession(i).Slope,2)])
xticklabels(animal.state_labels)
title('Std phase')
subplot(2,6,6), plot(skewness2-mean(skewness2)), 
title('Skewness phase'), 
xticks([1:size(PhasePrecession(i).Slope,2)])
xticklabels(animal.state_labels)
subplot(2,6,12), boxplot((skewness2-mean(skewness2))')
xticks([1:size(PhasePrecession(i).Slope,2)])
xticklabels(animal.state_labels)
title('Skewness phase')

%%
y11 = SpikeCount1;
plot_title = 'Spike Count';
y11 = Slope1; 
plot_title = 'Phase precession';
figure,
subplot(3,1,1)
plot(y11','.-'), axis tight
title({plot_title ' for place fields'}),ylabel(plot_title)
subplot(3,1,2)
boxplot(y11), axis tight
xticks([1:10:size(y11,2)])
xticklabels([1:10:size(y11,2)])
ylabel('Boxplot')
subplot(3,1,3)
plot(trials.temperature), axis tight
xlabel('Trial'),ylabel('Temperature')
% Linear fit of each field between spikes pr trial vs temperature
figure
r = [];
p = [];
P2 = [];
for i = 1:size(y11,1)
    notnan = ~isnan(y11(i,:));
    x1 = trials.temperature(notnan);
    y1 = y11(i,notnan);
    plot(x1,y1,'o'), hold on
    P = polyfit(x1,y1,1);
    yfit = P(1)*x1+P(2);
    [r(i),p(i)] = corr(x1',y1');
    P2(i,[1,2]) = P;
    P2(i,3) = r(i);
    P2(i,4) = p(i);
    if p(i)<0.05
        plot(x1,yfit,'r');
    else
        plot(x1,yfit,'k');
    end
end
% plot(trials.temperature,SpikeCount1,'.'),
title({'Temperature vs ' plot_title}),xlabel('Temperature'),ylabel({plot_title ' in Place field'})

%% % Stability measure across all place cells
% function out = plotPlaceFieldStability(behavior,units2,trials,theta,sr);
% for i = 1:size(units2,2)
%     unit2.state = interp1(animal.time,animal.state,unit.ts/sr,'nearest');
%     unit2.theta_phase = interp1([1:length(theta.phase)]/theta.sr,theta.phase,unit.ts/sr);
%     unit2.pos = interp1(animal.time,animal.pos,unit.ts/sr);
%     unit2.PhasePrecession = units(i).PhasePrecession;
%     unit2.trials = interp1(animal.time,trials.trials{k},unit.ts(indexes)/sr,'nearest');
% end
jj = 0;
out = [];
for m = 1:2
    for i = 1:size(units,2)
        unit = units(i);
        if ~isempty(unit.PhasePrecession)
            if m == 1
                if isfield(unit.PhasePrecession,'placefields_polar_theta')
                    phasepresession = unit.PhasePrecession.placefields_polar_theta;
                else
                    phasepresession = [];
                end
                trials2average = 25;%25;
                state = unit.rim;
                pos = unit.polar_theta;
            elseif m == 2
                if isfield(unit.PhasePrecession,'placefields_center_arm')
                    phasepresession = unit.PhasePrecession.placefields_center_arm;
                else
                    phasepresession = [];
                end
                trials2average = 15;%15;
                state = unit.arm;
                pos = unit.pos(2,:);
            end
            for j = 1:size(phasepresession,1)
                jj = jj + 1;
                tri = [];
                indexes = find(pos > phasepresession(j,1) & pos < phasepresession(j,2) & state & unit.speed > 10 & ~ismember(unit.trials,trials.error));
                lin1 = pos(indexes);
                circ1 = unit.theta_phase(indexes);
                [slope2,offset2,R2] = CircularLinearRegression(circ1,lin1,phasepresession(j,3));
                for k = 1:trials.total-trials2average
                    
                    indexes_trials = find(unit.trials(indexes) >= k & unit.trials(indexes) <= k+trials2average);
                    
                    [tri.slope1(k),tri.offset1(k),tri.R1(k)] = CircularLinearRegression(circ1(indexes_trials),lin1(indexes_trials),phasepresession(j,3));
                    
                    % Temperature
                    tri.temperature(k) = mean(unit.temperature(indexes(indexes_trials)));
                    
                    % Spike count
                    tri.spikecount(k) = length(indexes_trials)/length(unique(unit.trials(indexes(indexes_trials))));

                    % Animal speed
                    tri.speed(k) = mean(unit.speed(indexes(indexes_trials)));
                    
                    
                    % Theta frequency in field
                    tri.theta(k) = mean(unit.theta_freq(indexes(indexes_trials)));
                    
                    % Number of theta cycles
                    kk2 = 1;
                    theta_cycles = [];
                    for kk = 1:trials2average
                        indexes_trials2 = find(unit.trials(indexes(indexes_trials))==k+kk);
                        if ~isempty(indexes_trials2)
                            temp = unit.theta_phase2(indexes(indexes_trials(indexes_trials2)));
                            theta_cycles(kk) = temp(end)-temp(1);
                            kk2 = kk2 + 1;
                        end
                    end
                    tri.theta_cycles(k) = mean(theta_cycles(theta_cycles>0))/(2*pi);
                    
                    % Oscillation frequency the place field
                    timestamps = unit.ts(indexes(indexes_trials));
                    timestamps = timestamps - timestamps(1);
                    interval = zeros(1,ceil(timestamps(end)/sr*1000+50));
                    interval(1+round(timestamps/sr*1000)) = 1;
                    interval = nanconv(interval,gausswin(80)','edge');
                    
                    xcorr_spikes =xcorr(interval,180);
                    [~,locs] = findpeaks(xcorr_spikes(181+50:181+150),'SortStr','descend');
                        
                    if length(locs)>0
                        tri.oscillation_freq(k) = 1/abs(locs(1)+50)*1000;
                    else
                        tri.oscillation_freq(k) = nan;
                    end
                    % Oscillation freq estimated by cross correlating the
                    % in phase space.
                    phase_stamps = sort(unit.theta_phase2(indexes(indexes_trials)));
                    phase_stamps = phase_stamps - phase_stamps(1);
                    window = 500;
                    interval = zeros(1,ceil(phase_stamps(end)/(2*pi)*window)+window);
                    interval(1+round(phase_stamps/(2*pi)*window)) = 1;
                    interval = nanconv(interval,gausswin(120)','edge');
                    xcorr_spikes2 = xcorr(interval,window);
                    [~,locs2] = findpeaks(xcorr_spikes2(3*window/2:2*window),'SortStr','descend');
                    
                    if length(locs2)>0
                        tri.oscillation_freq2(k) = window / (locs2(1) + window/2);
                    else
                        tri.oscillation_freq2(k) = nan;
                    end
                    debug = 0;
                    if debug
                        figure(100)
                        subplot(2,3,1)
                        plot([-180:180],xcorr_spikes), hold on, plot(locs+50,xcorr_spikes(locs+181+50),'o'), xlim([0,180]),title(['Oscillation frequency, Unit ', num2str(i),])
                        subplot(2,3,2)
                        plot(lin1(indexes_trials),[circ1(indexes_trials);circ1(indexes_trials)+2*pi;circ1(indexes_trials)-2*pi],'.b','markersize',8), hold on
                        x = [phasepresession(j,1),phasepresession(j,2)];
                        offset1 = tri.offset1(k);
                        slope1 = tri.slope1(k);
                        while  2*pi*slope1*x(round(length(x)./2))+ offset1 < 0; offset1 = offset1+2*pi; end
                        while 2*pi*slope1*x(round(length(x)./2))+ offset1 > 2*pi; offset1 = offset1-2*pi;  end
                        plot(x,2*pi*slope1*x + offset1,'k-','linewidth',1.5),xlim(x),title('Phase precession')
                        hold off
                        subplot(2,3,3), plot([-window:window],xcorr_spikes2), hold on, plot(locs2(1)+window/2,xcorr_spikes2(locs2(1)+3*window/2),'o'), xlim([0,window]),title('Oscillation freq from phase')
                        subplot(2,3,4), plot(tri.oscillation_freq,'.-','markersize',8), title('Oscillation freq')
                        subplot(2,3,5), plot(tri.slope1,'.-','markersize',8), title('Phase precession slope')
                        subplot(2,3,6), plot(tri.oscillation_freq2,'.-','markersize',8), title('Oscillation freq from phase')
                        
                        drawnow
                        pause(0.1)
                        if k == trials.total-trials2average
                            subplot(2,2,1)
                            hold off
                        end
                    end

                    %                     % FFT analysis
                    %                     Y = fft(interval);
                    %                     Fs = 1000;
                    %                     L = length(interval);
                    %                     P2 = abs(Y/L);
                    %                     P1 = P2(1:L/2+1);
                    %                     P1(2:end-1) = 2*P1(2:end-1);
                    %                     f = Fs*(0:(L/2))/L;
                    %                     P1 = nanconv(P1,gausswin(1000)','edge');
                    %                     plot(f,P1)
                    %                     [~,locs] = findpeaks(P1,'SortStr','descend');
                    %                     if length(locs)>1
                    %                         tri.oscillation_freq(k) = f(locs(1));
                    %                     else
                    %                         tri.oscillation_freq(k) = nan;
                    %                     end
                end
                % Place field stability
                bins_x =  phasepresession(j,1):3:phasepresession(j,2);
                bin_y = 1:trials.total;
                tri.trial_hist = histcounts2(pos(indexes),unit.trials(indexes),bins_x,bin_y);
                tri.trial_hist = filter2(ones(1,trials2average)/trials2average,tri.trial_hist);
                tri.trial_hist2 = corrcoef(tri.trial_hist);

                figure,
                subplot(4,3,1), plot(pos(indexes),unit.trials(indexes),'.','markersize',8), ylabel('Trials'), axis tight
                subplot(4,3,2), plot(pos(indexes), [unit.theta_phase(indexes);unit.theta_phase(indexes)+2*pi],'.b','markersize',8), ylabel('Theta phase'), axis tight, if m == 1, title(['Unit ', num2str(i), '. Rim placefield. Field ', num2str(j)]), else title(['Unit ', num2str(i) '. Central arm placefield. Field ', num2str(j)]), end
                subplot(4,3,3), imagesc(bins_x,bin_y,tri.trial_hist'), ylabel('Placefield stability'), axis tight, set(gca,'YDir','normal')
                subplot(4,3,4), plot(tri.temperature,tri.slope1,'.-','markersize',8), ylabel('Slope'), axis tight
                subplot(4,3,5), plot(tri.temperature,tri.theta,'.-','markersize',8), ylabel('Theta freq'), axis tight
                subplot(4,3,6), plot(tri.temperature,tri.oscillation_freq,'.-','markersize',8), ylabel('Oscillation freq'), axis tight
                subplot(4,3,7), plot(tri.temperature,tri.theta_cycles,'.-','markersize',8), ylabel('Theta cycles'), axis tight
                subplot(4,3,8), plot(tri.temperature,tri.speed,'.-','markersize',8), ylabel('Speed'), axis tight,
                subplot(4,3,9), plot(tri.temperature,tri.spikecount,'.-','markersize',8), ylabel('Spike Count'), axis tight
                % subplot(4,3,10), plot3(tri.temperature,tri.speed,tri.slope1,'.-','markersize',8), xlabel('Temperature'), ylabel('Speed'), zlabel('Slope'), axis tight
                subplot(4,3,10), imagesc(bin_y,bin_y,tri.trial_hist2'), xlabel('Trial'), ylabel('Trial'), axis tight
                subplot(4,3,11), plot3(tri.temperature,tri.speed,tri.oscillation_freq,'.-','markersize',8), xlabel('Temperature'), ylabel('Speed'), zlabel('Oscillation freq'), axis tight
                %tri.offset1(tri.offset1< 0) = tri.offset1(tri.offset1< 0)+2*pi;
                %subplot(4,3,12), plot(tri.temperature,tri.offset1,'.-'), ylabel('Offset'), axis tight
                subplot(4,3,12), plot(tri.temperature,tri.oscillation_freq./tri.theta,'.-'), ylabel('frequency ratio'), axis tight

                figure,
                subplot(3,3,1), plot(tri.temperature,'.-m','markersize',8), ylabel('Temperature'), axis tight, grid on
                subplot(3,3,2), plot(tri.theta,'.-r','markersize',8), ylabel('LFP theta frequency'), axis tight, grid on, if m == 1, title(['Unit ', num2str(i), '. Rim placefield. Field ', num2str(j)]), else title(['Unit ', num2str(i) '. Central arm placefield. Field ', num2str(j)]), end
                subplot(3,3,3), plot(tri.speed,'.-k','markersize',8), ylabel('Running speed'), axis tight, grid on
                subplot(3,3,4), plot(abs(tri.slope1),'.-','markersize',8), ylabel('Precession slope'), axis tight, grid on, hold on, 
                plot(tri.oscillation_freq./tri.theta./tri.speed,'.-','color',[0.5,0.5,0.5])
                plot([1,trials.total-trials2average],abs([slope2,slope2]),'--b','linewidth',1)
                % plot(abs(tri.slope1)./tri.speed*mean(tri.speed),'.-','markersize',8,'color',[0.5,0.5,0.5])
                subplot(3,3,5), plot(tri.oscillation_freq,'.-','markersize',8), hold on, 
                plot(tri.theta,'.-r','markersize',8),  ylabel('Field oscillation freq'), axis tight, grid on
                plot(tri.oscillation_freq2.*tri.theta,'.-k','markersize',8)
                subplot(3,3,6), plot(tri.theta_cycles,'.-','markersize',8), ylabel('Theta cycles'), axis tight, grid on
                subplot(3,3,7), plot(tri.oscillation_freq./tri.theta,'.-','markersize',8), ylabel('frequency ratio'), axis tight, grid on, %ylim([0,max(tri.oscillation_freq./tri.theta)])
                hold on, plot(tri.oscillation_freq2,'.-k','markersize',8)
                subplot(3,3,9), plot(tri.spikecount,'.-','markersize',8), ylabel('Spike count'), axis tight, xlabel('Trials'), grid on

                if length(tri.temperature) > trials.total/2
                    subplot(3,3,8)
                    lags2 = min(length(tri.temperature),41)-1;
                    [xcf,lags,bounds] = crosscorr(tri.temperature,tri.speed,lags2);
                    plot(lags,xcf,'.-k','markersize',8), hold on
                    
                    [xcf,lags,bounds] = crosscorr(tri.temperature(~isnan(tri.oscillation_freq)),tri.oscillation_freq(~isnan(tri.oscillation_freq)),lags2);
                    plot(lags,xcf,'.-b','markersize',8)
                    
                    [xcf,lags,bounds] = crosscorr(tri.temperature,tri.theta,lags2);
                    plot(lags,xcf,'.-r','markersize',8)
                    
                    [xcf,lags,bounds] = crosscorr(tri.temperature,tri.theta_cycles,lags2);
                    plot(lags,xcf,'.-','markersize',8)
                    
                    [xcf,lags,bounds] = crosscorr(tri.temperature,tri.slope1,lags2);
                    plot(lags,xcf,'.-','markersize',8)
                    
                    ylabel('Correlation'), plot([lags(1),lags(end)],[bounds;bounds],'k'), grid on, axis tight
                    legend({'Running speed','Oscillation frequency','Theta Freq','Theta Cycles','Precession slope'})
                end
                % Temperature
                [out.r_oscillation_freq(jj),out.p_oscillation_freq(jj)] = corr(tri.temperature(~isnan(tri.oscillation_freq))',tri.oscillation_freq(~isnan(tri.oscillation_freq))');
                [out.r_slope1(jj),out.p_slope1(jj)] = corr(tri.temperature',tri.slope1');
                [out.r_theta_cycles(jj),out.p_theta_cycles(jj)] = corr(tri.temperature',tri.theta_cycles');
                [out.r_spikecount(jj),out.p_spikecount(jj)] = corr(tri.temperature',tri.spikecount');
                
                % Theta
                [out.r1_oscillation_freq(jj),out.p1_oscillation_freq(jj)] = corr(tri.theta(~isnan(tri.oscillation_freq))',tri.oscillation_freq(~isnan(tri.oscillation_freq))');
                [out.r1_slope1(jj),out.p1_slope1(jj)] = corr(tri.theta',tri.slope1');
                [out.r1_theta_cycles(jj),out.p1_theta_cycles(jj)] = corr(tri.theta',tri.theta_cycles');
                [out.r1_spikecount(jj),out.p1_spikecount(jj)] = corr(tri.theta',tri.spikecount');
                
                % Speed vs oscillation freq
                [out.r2_oscillation_freq(jj),out.p2_oscillation_freq(jj)] = corr(tri.speed(~isnan(tri.oscillation_freq))',tri.oscillation_freq(~isnan(tri.oscillation_freq))');
                [out.r2_slope1(jj),out.p2_slope1(jj)] = corr(tri.speed',tri.slope1');
                [out.r2_theta_cycles(jj),out.p2_theta_cycles(jj)] = corr(tri.speed',tri.theta_cycles');
                [out.r2_spikecount(jj),out.p2_spikecount(jj)] = corr(tri.speed',tri.spikecount');
                
                % diff in phase vs slope and speed
                [out.r3_slope1(jj),out.p3_slope1(jj)] = corr(abs(tri.slope1(~isnan(tri.oscillation_freq)))', tri.oscillation_freq(~isnan(tri.oscillation_freq))'-tri.theta(~isnan(tri.oscillation_freq))');
                [out.r3_speed(jj),out.p3_speed(jj)] = corr(tri.speed(~isnan(tri.oscillation_freq))', tri.oscillation_freq(~isnan(tri.oscillation_freq))'-tri.theta(~isnan(tri.oscillation_freq))');
                
                % Correlation matrix
                out.trial_hist2(:,:,jj) = tri.trial_hist2;
%                 % GLM fits
%                 X = [tri.temperature;tri.speed]';
%                 y1 = tri.oscillation_freq'; % Frequency
%                 mdl1 = fitglm(X,y1,'linear');
                % Spike Count
                out.spikecount(jj,:) = tri.spikecount;
            end
        end
    end
end

hist_bins = [-1:0.1:1];
figure,
subplot(3,4,1)
histogram(out.r_oscillation_freq,hist_bins), title('oscillation freq'), xlim([-1, 1]), ylabel('Temperature'), hold on
histogram(out.r_oscillation_freq(find(out.p_oscillation_freq > 0.05)),hist_bins)
plot(mean(out.r_oscillation_freq(find(out.p_oscillation_freq < 0.05))),0,'v','linewidth',2)
subplot(3,4,2)
histogram(out.r_slope1,hist_bins), title('slope'), xlim([-1, 1]), hold on
histogram(out.r_slope1(find(out.p_slope1 > 0.05)),hist_bins)
plot(mean(out.r_slope1(find(out.p_slope1 < 0.05))),0,'v','linewidth',2)
subplot(3,4,3)
histogram(out.r_theta_cycles,hist_bins), title('theta cycles'), xlim([-1, 1]), hold on
histogram(out.r_theta_cycles(find(out.p_theta_cycles > 0.05)),hist_bins)
plot(mean(out.r_theta_cycles(find(out.p_theta_cycles < 0.05))),0,'v','linewidth',2)
subplot(3,4,4)
histogram(out.r_spikecount,hist_bins), title('spikecount'), xlim([-1, 1]), hold on
histogram(out.r_spikecount(find(out.p_spikecount > 0.05)),hist_bins)
plot(mean(out.r_spikecount(find(out.p_spikecount < 0.05))),0,'v','linewidth',2)

% Theta
subplot(3,4,5)
histogram(out.r1_oscillation_freq,hist_bins), title('oscillation freq'), xlim([-1, 1]), ylabel('Theta'), hold on
histogram(out.r1_oscillation_freq(find(out.p1_oscillation_freq > 0.05)),hist_bins)
plot(mean(out.r1_oscillation_freq(find(out.p1_oscillation_freq < 0.05))),0,'v','linewidth',2)

subplot(3,4,6)
histogram(out.r1_slope1,hist_bins), title('slope'), xlim([-1, 1]), hold on
histogram(out.r1_slope1(find(out.p1_slope1 > 0.05)),hist_bins)
plot(mean(out.r1_slope1(find(out.p1_slope1 < 0.05))),0,'v','linewidth',2)

subplot(3,4,7)
histogram(out.r1_theta_cycles,hist_bins), title('theta cycles'), xlim([-1, 1]), hold on
histogram(out.r1_theta_cycles(find(out.p1_theta_cycles > 0.05)),hist_bins)
plot(mean(out.r1_theta_cycles(find(out.p1_theta_cycles < 0.05))),0,'v','linewidth',2)

subplot(3,4,8)
histogram(out.r1_spikecount,hist_bins), title('spikecount'), xlim([-1, 1]), hold on
histogram(out.r1_spikecount(find(out.p1_spikecount > 0.05)),hist_bins)
plot(mean(out.r1_spikecount(find(out.p1_spikecount < 0.05))),0,'v','linewidth',2)

% Speed
subplot(3,4,9)
histogram(out.r2_oscillation_freq,hist_bins), title('oscillation freq'), xlim([-1, 1]), ylabel('Speed'), hold on
histogram(out.r2_oscillation_freq(find(out.p2_oscillation_freq > 0.05)),hist_bins)
plot(mean(out.r2_oscillation_freq(find(out.p2_oscillation_freq < 0.05))),0,'v','linewidth',2)

subplot(3,4,10)
histogram(out.r2_slope1,hist_bins), title('slope'), xlim([-1, 1]), hold on
histogram(out.r2_slope1(find(out.p2_slope1 > 0.05)),hist_bins)
plot(mean(out.r2_slope1(find(out.p2_slope1 < 0.05))),0,'v','linewidth',2)

subplot(3,4,11)
histogram(out.r2_theta_cycles,hist_bins), title('theta cycles'), xlim([-1, 1]), hold on
histogram(out.r2_theta_cycles(find(out.p2_theta_cycles > 0.05)),hist_bins)
plot(mean(out.r2_theta_cycles(find(out.p2_theta_cycles < 0.05))),0,'v','linewidth',2)

subplot(3,4,12)
histogram(out.r2_spikecount,hist_bins), title('spikecount'), xlim([-1, 1]), hold on
histogram(out.r2_spikecount(find(out.p2_spikecount > 0.05)),hist_bins)
plot(mean(out.r2_spikecount(find(out.p2_spikecount < 0.05))),0,'v','linewidth',2)

figure, subplot(2,2,1), plot(out.r_oscillation_freq,out.r2_oscillation_freq,'o','linewidth',2), xlim([-1,1]), ylim([-1,1]), hold on, plot([-1,1],[-1,1],'-')
xlabel('Temperature'), ylabel('Running speed'),title('place cells oscillation correlations'), grid on

subplot(2,2,2)
histogram(out.r3_slope1,hist_bins), title('Theta diff vs precession slope'), xlim([-1, 1]), hold on
histogram(out.r3_slope1(find(out.p3_slope1 > 0.05)),hist_bins)
plot(mean(out.r3_slope1(find(out.p3_slope1 < 0.05))),0,'v','linewidth',2)

subplot(2,2,3)
histogram(out.r3_speed,hist_bins), title('Theta ratio vs speed'), xlim([-1, 1]), hold on
histogram(out.r3_speed(find(out.p3_speed > 0.05)),hist_bins)
plot(mean(out.r3_speed(find(out.p3_speed < 0.05))),0,'v','linewidth',2)

subplot(2,2,4), plot(out.r3_slope1,out.r3_speed,'o','linewidth',2), xlim([-1,1]), ylim([-1,1]), hold on, plot([-1,1],[-1,1],'-')
xlabel('Precession slope'), ylabel('Running speed'),title('Oscillation freq/LFP theta correlations'), grid on

figure, subplot(2,1,1)
imagesc(mean(out.trial_hist2,3))
subplot(2,1,2)
plot(zscore(out.spikecount'))

%% Analyzing interneurons
recording.SpikeSorting.interneurons
if isfield(recording.SpikeSorting,'interneurons')
    cells = find(ismember(horzcat(units.kwik_id),recording.SpikeSorting.interneurons));
    for i = 1:length(cells)
        unit = units(cells(i));
        tri = []; 
        for k = 1:trials.total
                indexes = find((unit.arm | unit.rim) & unit.speed > 10 & unit.trials == k);
                tri.temperature(k) = mean(unit.temperature(indexes));
                %tri.speed(k) = mean(unit.speed(indexes));
                indexes2 = find((animal.arm | animal.rim) & animal.speed > 10 & trials.trials2 == k);
                tri.speed(k) = mean(animal.speed(indexes2));
                tri.theta(k) = mean(unit.theta_freq(indexes));
                tri.spikecount(k) = length(indexes)/(sum(trials.trials2==k)/animal.sr);
        end
        figure,
        subplot(4,1,1), plot(tri.temperature), xlabel('Trial'),ylabel('Temperature'), title(['Unit ' num2str(cells(i))]), grid on, axis tight
        subplot(4,1,2), plot(tri.theta), xlabel('Trial'),ylabel('Theta'), grid on, axis tight
        subplot(4,1,3), plot(tri.speed), xlabel('Trial'),ylabel('Animal speed'), grid on, axis tight
        subplot(4,1,4), plot(tri.spikecount), xlabel('Trial'),ylabel('Firing rate'), grid on, axis tight
    end
end
