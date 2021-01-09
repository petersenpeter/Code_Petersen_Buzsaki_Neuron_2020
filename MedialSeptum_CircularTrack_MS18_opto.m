% DOCID = '1WBqEo0OM5qdqmAD_7cGsJe0iyGf6hjgPXy-V2VlVY9c'
% result = GetGoogleSpreadsheet(DOCID);
% Medial Septum Circular Track
clear all
MedialSeptum_Recordings
% Processed datasets in MS12: 78,79,80,81,
% Processed datasets in MS13: 92
id = 141; % 63 

recording = recordings(id);
if ~isempty(recording.dataroot)
    datapath = recording.dataroot;
end

cd(fullfile(datapath, recording.animal_id, recording.name))
Intan_rec_info = read_Intan_RHD2000_file_Peter(fullfile(datapath, recording.animal_id, recording.name));
fname = [recording.name '.dat'];
nChannels = size(Intan_rec_info.amplifier_channels,2);
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
inputs.ch_opto_on = recording.ch_opto_on; % Blue laser diode
inputs.ch_opto_off = recording.ch_opto_off; % Red laser diode
recording.sr_lfp = sr/16;
% track_boundaries = recording.track_boundaries;
arena = recording.arena;
% nb_lfp_periods = ceil((time_frame(2)-time_frame(1))/lfp_periods);
temp_ = dir(fname);
recording_length = round(temp_.bytes/sr/nChannels/2)-1;

maze = recording.maze;
maze.polar_rho_limits = [44,65];
maze.polar_theta_limits = [-2.8,2.8]*maze.radius_in;
maze.pos_x_limits = [-10,10]; % -15
maze.pos_y_limits = [-40,45];

maze.boundary{1} = [0,40];
maze.boundary{2} = [0,25];
maze.boundary{3} = [maze.pos_x_limits(1),40];
maze.boundary{4} = [maze.pos_x_limits(2),40];
maze.boundary{5} = [maze.radius_in-3.25,maze.polar_theta_limits(2)];

animal = [];
for fn = fieldnames(maze)'
   animal.(fn{1}) = maze.(fn{1});
end
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
    else
        load('Optitrack.mat');
    end
end
if ~isempty(recording.OptiTracktracking_offset)
    Optitrack.position3D = Optitrack.position3D + recording.OptiTracktracking_offset';
end
disp('Loading opto data')
if ~exist('opto_data.mat')
    num_channels = length(Intan_rec_info.board_adc_channels); % ADC input info from header file
    fileinfo = dir('analogin.dat');
    num_samples = fileinfo.bytes/(num_channels * 2); % uint16 = 2 bytes
    fid = fopen('analogin.dat', 'r');
    v = fread(fid, [num_channels, num_samples], 'uint16');
    fclose(fid);
    % v = v * 0.000050354; % convert to volts
    opto_signal_stim_on = medfilt1(v(inputs.ch_opto_on,:),200) * 0.000050354;
    opto_signal_stim_off = medfilt1(v(inputs.ch_opto_off,:),200) * 0.000050354;
    opto_stim_on = find(diff(opto_signal_stim_on>1.0)==-1);
    opto_stim_off = find(diff(opto_signal_stim_off<1.0)==-1);
    opto_stim_onset = opto_stim_on;
    opto_stim_offset = opto_stim_onset+sr*30;
    for i = 1:length(opto_stim_onset)
        [~,ia,~] = intersect(opto_stim_off, opto_stim_onset(i):opto_stim_offset(i));
        if ia
            opto_stim_offset(i) = opto_stim_off(ia(1));
        end
    end
    opto.onset = opto_stim_onset/sr;
    opto.offset = opto_stim_offset/sr;
    save('opto_data.mat','opto');
else
    load('opto_data.mat');
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
%         %fullfile([datapath, recording.name(1:6) recording.animal_id], recording.concat_recordings(recording.concat_behavior_nb), 'digitalin.dat');
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
        fullpath = fullfile([datapath, recording.name(1:6) recording.animal_id], recording.concat_recordings{i}, [recording.concat_recordings{i}, '.dat']);
        temp2_ = dir(fullpath);
        prebehaviortime = prebehaviortime + temp2_.bytes/nChannels/2/sr;
    end
    end
    i = recording.concat_behavior_nb;
    fullpath = fullfile([datapath, recording.name(1:6) recording.animal_id], recording.concat_recordings{i}, [recording.concat_recordings{i}, '.dat']);
    temp2_ = dir(fullpath);
    behaviortime = temp2_.bytes/nChannels/2/sr;
else
    temp_ = dir(fname);
    behaviortime = temp_.bytes/nChannels/2/sr;
end

disp('4. Calculating behavior')
if inputs.ch_OptiTrack_sync == 0
if inputs.ch_camera_sync ~= 0
    animal.sr = Camera.framerate;
    if length(recording.concat_recordings) > 0
        camera_pulses = find(digital_on{inputs.ch_camera_sync}/sr > prebehaviortime & digital_on{inputs.ch_camera_sync}/sr < prebehaviortime + behaviortime);
        pulse_corrected = camera_pulses;
        pulse_corrected(find(diff(pulse_corrected)<mean(diff(pulse_corrected))/2)) = [];
        disp(['Camera pulses detected: ' num2str(length(camera_pulses))])
        disp(['Corrected camera pulses detected: ' num2str(length(pulse_corrected))])
        animal.time = digital_on{inputs.ch_camera_sync}(pulse_corrected(1:Camera.frames))'/sr;
        Camera.onset = min(digital_on{inputs.ch_camera_sync}(pulse_corrected(1)),digital_off{inputs.ch_camera_sync}(pulse_corrected(1)))/sr;
        Camera.offset = (temp_.bytes/sr/nChannels/2)-max(digital_on{inputs.ch_camera_sync}(end),digital_off{inputs.ch_camera_sync}(end))/sr;
    else
        camera_pulses = digital_on{inputs.ch_camera_sync};
        pulse_corrected = camera_pulses;
        pulse_corrected(find(diff(pulse_corrected)<mean(diff(pulse_corrected))/2)) = [];
        animal.time = pulse_corrected(1:Camera.frames)'/sr;
        Camera.onset = min(digital_on{inputs.ch_camera_sync}(1),digital_off{inputs.ch_camera_sync}(1))/sr;
        Camera.offset = (temp_.bytes/sr/nChannels/2)-max(digital_on{inputs.ch_camera_sync}(end),digital_off{inputs.ch_camera_sync}(end))/sr;
    end
    if length(digital_on{inputs.ch_camera_sync})-Camera.frames ~= 0
        
        disp(['Camera frames and TTL pulses dont add up:'])
        disp(['Number of Camera TTL pulses: ' num2str(length(digital_on{inputs.ch_camera_sync}))])
        disp(['Number of Camera frames: ' num2str(Camera.frames)])
        disp(['Difference: ' num2str(length(digital_on{inputs.ch_camera_sync})-Camera.frames) ' frames'])
    end
    
    animal.pos  = Camera.pos;
    animal.acceleration = Camera.framerate*[0,sqrt(sum((diff(Camera.pos',2).^2),2))',0];
    animal.pos(:,animal.acceleration>100) = nan;
    animal.speed = [0,Camera.framerate*sqrt(sum((diff(Camera.pos').^2),2))'];
    gausswin_size = animal.sr/4;
    for i = 1:2
        % animal.pos(i,:) = medfilt1(animal.pos(i,:),5,'omitnan');
        animal.pos(i,:) = nanconv(animal.pos(i,:),gausswin(gausswin_size)','edge');
    end
    animal.speed = medfilt1(animal.speed,5,'omitnan');
end
end

if inputs.ch_OptiTrack_sync ~= 0
    Optitrack.onset = min(digital_on{inputs.ch_OptiTrack_sync}(1),digital_off{inputs.ch_OptiTrack_sync}(1))/sr;
    Optitrack.offset = (temp_.bytes/sr/nChannels/2)-max(digital_on{inputs.ch_OptiTrack_sync}(end),digital_off{inputs.ch_OptiTrack_sync}(end))/sr;
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
    animal.acceleration = Optitrack.FrameRate*[0,diff(animal.speed)];
    if size(animal.time,2) < size(animal.pos,2)
        warning('There are fewer Optitrack digital pulses than position points')
        animal.pos = animal.pos(:,1:length(animal.time));
        animal.speed = animal.speed(1:length(animal.time));
        animal.acceleration = animal.acceleration(1:length(animal.time));
    end
    if size(animal.time,2) > size(animal.pos,2)
        warning('There are more Optitrack digital pulses than position points')
        animal.time = animal.time(1:size(animal.pos,2));
    end
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

animal.circularpart = find(animal.polar_theta > animal.polar_theta_limits(1) & animal.polar_theta < animal.polar_theta_limits(2) & animal.polar_rho > animal.polar_rho_limits(1) & animal.polar_rho < animal.polar_rho_limits(2));
animal.centralarmpart = find(animal.pos(1,:) > animal.pos_x_limits(1) & animal.pos(1,:) < animal.pos_x_limits(2) & animal.pos(2,:) > animal.pos_y_limits(1) & animal.pos(2,:) < animal.pos_y_limits(2));
animal.arm = double(animal.pos(1,:) > animal.pos_x_limits(1) & animal.pos(1,:) < animal.pos_x_limits(2) & animal.pos(2,:) > animal.pos_y_limits(1) & animal.pos(2,:) < animal.pos_y_limits(2));
animal.rim = double((animal.polar_rho > animal.polar_rho_limits(1) & animal.polar_rho < animal.polar_rho_limits(2) & animal.polar_theta > animal.polar_theta_limits(1) & animal.polar_theta < animal.polar_theta_limits(2)));
animal.error_trials = zeros(1,size(animal.pos,2));
animal.maze = maze;
animal.state_labels = {'Control','Stim',};
animal.pos_linearized = linearize_pos(animal,recording.arena);
animal.pos_linearized_limits = [0,diff(animal.pos_y_limits) + diff(animal.polar_theta_limits)];

% Loading Temperature data
if inputs.ch_temp >0
    disp('5. Loading Temperature data')
    if ~exist('temperature.mat')
        if isempty(recording.ch_temp_type)
            recording.ch_temp_type = 'analog';
        end
        temperature = LoadTemperature(recording.ch_temp,recording.ch_temp_type,pwd);
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
% if recording.cooling_session == 0
%     cooling.onsets = animal.time(round(length(animal.time)/2));
%     cooling.offsets = animal.time(round(length(animal.time)));
%     cooling.cooling = [cooling.onsets(1)+10;cooling.offsets(1)];
%     cooling.nocooling = [[1,cooling.onsets(1)];[cooling.offsets(1)+120,behaviortime]]';
% else
%     if inputs.ch_temp ~= 0
%         temp_range = [32,34];% temp_1 defines the upper limit on cooling, temp_2 the lower limit on no cooling
%         test = find(diff(temperature.temp < temp_range(1),2)== 1);
%         test(diff(test)<10*temperature.sr)=[];
%         cooling.onsets = temperature.time(test(temperature.temp(test+1)-temperature.temp(test)<0));
%         cooling.offsets = temperature.time(test(temperature.temp(test+1)-temperature.temp(test)>0));
%         if length(cooling.offsets)<length(cooling.onsets)
%             cooling.offsets = [cooling.offsets,temperature.time(end)]
%         end
%         cooling.cooling = [cooling.onsets;cooling.offsets];
%         cooling.cooling2 = [cooling.onsets-20;cooling.offsets];
%         cooling.nocooling = reshape([prebehaviortime;cooling.cooling2(:);prebehaviortime+behaviortime],[2,size(cooling.cooling2,2)+1]);
%     elseif inputs.ch_CoolingPulses ~= 0
%         cooling.onsets = digital_on{inputs.ch_CoolingPulses}/sr;
%         cooling.offsets = cooling.onsets + 12*60;
%         cooling.cooling = [cooling.onsets(1)+10;cooling.offsets(1)];
%         cooling.nocooling = [[1,cooling.onsets'];[cooling.offsets'+120,behaviortime]]';
%     else
%         cooling.onsets = recording.cooling_onsets;
%         cooling.offsets = recording.cooling_offsets;
%         cooling.cooling = [cooling.onsets(1)+10;cooling.offsets(1)]+prebehaviortime;
%         cooling.nocooling = [[1,cooling.onsets(1)]+prebehaviortime;[cooling.offsets(1)+120,behaviortime]+prebehaviortime]';
%     end
% end
% Opto
if ~isempty(opto.onset)
    cooling.onsets = opto.onset;
    cooling.offsets = opto.offset;
    cooling.cooling = [cooling.onsets;cooling.offsets];
    cooling.nocooling = [[1;cooling.onsets(1)],[cooling.offsets(1:end-1);cooling.onsets(2:end)],[cooling.offsets(end);behaviortime]];
else
    cooling.nocooling = [];
end
% Separating left and right trials
disp('7. Defining trials for the behavior')
[trials,animal] = trials_thetamaze(animal, maze,[],cooling);
trials.labels = {'Left','Right'};
trials.total = length(trials.start);
save('animal.mat','animal')

temp1 = 0; temp2 = 0;
for j = 1:size(cooling.cooling,2)
    temp1 = temp1 + length(find(animal.time(trials.start(trials.error)) > cooling.cooling(1,j) & animal.time(trials.start(trials.error)) < cooling.cooling(2,j)));
    temp2 = temp2 + length(find(animal.time(trials.start) > cooling.cooling(1,j) & animal.time(trials.start) < cooling.cooling(2,j)));
end
trials.Cooling_error_ratio = 100*temp1/temp2;
trials.NoCooling_error_ratio_before = 100*length(find(animal.time(trials.start(trials.error)) < cooling.cooling(1)))/length(find(animal.time(trials.start) < cooling.cooling(1)));
temp1 = 0; temp2 = 0;
for j = 1:size(cooling.cooling,2)-1
    temp1 = temp1 + length(find(animal.time(trials.start(trials.error)) > cooling.cooling(2,j)+20 &  animal.time(trials.start(trials.error)) < cooling.cooling(1,j+1)));
    temp2 = temp2 + length(find(animal.time(trials.start) > cooling.cooling(2,j)+20 & animal.time(trials.start) < cooling.cooling(1,j+1)+20));
end
temp1 = temp1 + length(find(animal.time(trials.start(trials.error)) > cooling.cooling(2,end)+20));
temp2 = temp2 + length(find(animal.time(trials.start) > cooling.cooling(2,end)+20));
trials.NoCooling_error_ratio_after = 100*temp1/temp2;

if ~isempty(temperature)
    for i = 1:trials.total
        trials.temperature(i) = mean(temperature.temp(round(animal.time(trials.start(i))*temperature.sr):round(animal.time(trials.end(i))*temperature.sr)));
    end
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

% Redefining temperature effects
if inputs.ch_temp == 0
    temperature.temp = 37* ones(1,length(theta.phase));
    temperature.sr = recording.sr_lfp;
    temperature.time = [1:length(temperature.temp)]/recording.sr_lfp;
    [t,gde] = alphafunction2;
    gde = interp1(t,gde,[t(1):(1/temperature.sr):t(end)]);
    injector = zeros(1,length(theta.phase));
    injector(round(cooling.onsets*temperature.sr)) = 1;
    injector = conv(injector,-gde);
    injector = injector(1:length(theta.phase));
    injector_amplitude = 20;
    temperature.temp = temperature.temp + injector*injector_amplitude;
    animal.temperature = interp1(temperature.time,temperature.temp,animal.time);
end

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
plot(cooling.nocooling',[0,0],'color','red','linewidth',2)
plot(cooling.cooling,[0,0],'color','blue','linewidth',2)
gridxy(animal.time(trials.start),'color','g')
gridxy(animal.time(trials.start(trials.error)),'color','m','linewidth',2)
legend({'Error trials','All trials','Speed','Cooling','NoCooling'})
xlabel('Time (s)'), ylabel('Speed (cm/s)')

subplot(3,2,5)
temp1 = [];
for j = 1:size(cooling.cooling,2)
    temp1 = [temp1, find(animal.time(trials.all) > cooling.cooling(1,j) & animal.time(trials.all) < cooling.cooling(2,j))];
end
cooling.times_cooling = temp1;

temp1 = [];
temp1 = find(animal.time(trials.all) < cooling.cooling(1));
for j = 1:size(cooling.cooling,2)-1
    temp1 = [temp1, find(animal.time(trials.all) > cooling.cooling(2,j)+20 &  animal.time(trials.all) < cooling.cooling(1,j+1))];
end
temp1 = [temp1, find(animal.time(trials.all) > cooling.cooling(2,end)+20)];
cooling.times_nocooling = temp1;

histogram(animal.speed(trials.all(cooling.times_cooling)),'BinEdges',bins_speed,'Normalization','probability'), hold on
histogram(animal.speed(trials.all(cooling.times_nocooling)),'BinEdges',bins_speed,'Normalization','probability'),
legend({'Cooling','No Cooling'})
xlabel('Speed (cm/s)'), ylabel('Probability'), title(['Speed during the trials (Total: ' num2str(length(trials.start))  ' trials)'])

subplot(4,2,8)
bar(1, trials.NoCooling_error_ratio_before, 'red'), hold on
bar(2, trials.Cooling_error_ratio, 'blue')
bar(3, trials.NoCooling_error_ratio_after, 'red'), hold on
xticks([1, 2, 3]), xticklabels({'Pre Cooling','Cooling','Post cooling'}),ylabel('Percentage of errors'),title('Error trials (%)'),axis tight,
xlim([0,4]),ylim([0,30])
subplot(4,2,6)
if ~isempty(temperature)
    plot(temperature.time,temperature.temp), xlim([prebehaviortime,prebehaviortime+behaviortime]), title('Temperature')
    ylim([20,43])
end

% Position on the rim defined in polar coordinates
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

% Temperature vs Theta frequency vs temperature
if ~isempty(temperature)
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
end

% Speed histogram for the maze
disp('Plotting speed histogram for maze')
speed_bins = [30:5:120];
figure,
subplot(2,1,1)
speed_histogram = [];
bins_rim = [animal.polar_theta_limits(1):5:animal.polar_theta_limits(2)];
for j = 1:length(bins_rim)-1
    indexes = find(animal.polar_theta > bins_rim(j) & animal.polar_theta < bins_rim(j+1) & animal.rim & animal.speed > speed_bins(1) & ~ismember(animal.trials,trials.error));
	temp = histogram(animal.speed(indexes),speed_bins,'Normalization','probability');
    speed_histogram(:,j) = temp.Values;
end
imagesc(bins_rim,speed_bins,speed_histogram), set(gca,'YDir','normal'), title('Speed on the rim'), xlabel('Position (cm)'), ylabel('Running speed (cm/s)')

subplot(2,1,2)
speed_histogram = [];
bins_arm = [animal.pos_y_limits(1):5:animal.pos_y_limits(2)];
for j = 1:length(bins_arm)-1
    indexes = find(animal.pos(2,:) > bins_arm(j) & animal.pos(2,:) < bins_arm(j+1) & animal.arm & animal.speed > speed_bins(1) & ~ismember(animal.trials,trials.error));
	temp = histogram(animal.speed(indexes),speed_bins,'Normalization','probability');
    speed_histogram(:,j) = temp.Values;
end
imagesc(bins_arm,speed_bins,speed_histogram), set(gca,'YDir','normal'), title('Speed on the arm'), xlabel('Position (cm)'), ylabel('Running speed (cm/s)')
disp('9. Finished loading the recording')

%% % Plotting theta frequency vs running speed
plot_ThetaVsSpeed(recording,animal,cooling);
plot_ThetaVsAcceleration(recording,animal,cooling);

%% % Gamma vs temperature
stats = [];
for i = [1:recording.nChannels]
    i
    recording.ch_theta = i;
    stats(i) = plot_GammaVsTemperature(recording,animal,cooling);
    figure(1000)
    stairs(stats(i).freqlist,mean(stats(i).freq_cooling),'b'), hold on
    stairs(stats(i).freqlist,mean(stats(i).freq_nocooling),'r')
end
save('GammaStats.mat','stats')
temp = [];
colorid = [];
colorpalet = [0,0,0; 0.5,0.5,0.5; 0.2,0.2,0.2; 1,0,0; 0,1,0; 1,0,1; 0,1,1];
xml = LoadXml(recording.name,'.xml');
figure(1001)
for i = 1:recording.nChannels
    temp(:,i) = mean(stats(i).freq_cooling)./mean(stats(i).freq_nocooling);
    for j = 1:length(xml.SpkGrps)
        if sum(xml.SpkGrps(j).Channels+1 == i)
            colorid(i) = j;
        end
    end
    stairs(stats(i).freqlist,temp(:,i),'color',colorpalet(colorid(i),:)), hold on
end
stairs(stats(i).freqlist,mean(temp,2),'k','linewidth',2), hold on
title('Gamma (Power ratio)'), xlabel('Frequency (Hz)'), ylabel('Ratio')

%% % Units
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
    %units(i).temperature = units(i).theta_freq
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
behavior.speed_th = 10;
behavior.rim = animal.rim;
behavior.state = trials.state;
behavior.pos = animal.polar_theta;
behavior.pos_limits = animal.polar_theta_limits;
%behavior.optogenetics.pos = optogenetics.polar_theta(optogenetics.rim);
%behavior.optogenetics.trial = optogenetics.trials(optogenetics.rim);
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

PhasePrecessionSlope1 = plot_FiringRateMap(behavior,units2,trials,theta,sr,'rim');

% Firing rate on center arm
behavior = animal;
behavior.speed_th = 10;
behavior.state = trials.state;
behavior.('pos') = animal.pos(2,:);
behavior.rim = animal.arm;
behavior.pos_limits = animal.pos_y_limits;
behavior.optogenetics.pos = optogenetics.pos(2,optogenetics.arm);
behavior.optogenetics.trial = optogenetics.trials(optogenetics.arm);
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
PhasePrecessionSlope2 = plot_FiringRateMap(behavior,units2,trials,theta,sr,'arm');

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
            slope1(j,m:m+elements-1) = PhasePrecession(i).placefield(:,3)' .* PhasePrecession(i).Slope{j}.slope1;
            mean1(j,m:m+elements-1) = -PhasePrecession(i).placefield(:,3)' .* PhasePrecession(i).Slope{j}.mean;
            std1(j,m:m+elements-1) = PhasePrecession(i).Slope{j}.std;
            skewness1(j,m:m+elements-1) = -PhasePrecession(i).placefield(:,3)' .* PhasePrecession(i).Slope{j}.skewness;
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
            slope1(j,m:m+elements-1) = PhasePrecession(i).placefield(:,3)' .* PhasePrecession(i).Slope{j}.slope1;
            mean1(j,m:m+elements-1) = -PhasePrecession(i).placefield(:,3)' .* PhasePrecession(i).Slope{j}.mean;
            std1(j,m:m+elements-1) = PhasePrecession(i).Slope{j}.std;
            skewness1(j,m:m+elements-1) = -PhasePrecession(i).placefield(:,3)' .* PhasePrecession(i).Slope{j}.skewness;
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
subplot(2,2,1), plot(slope1-mean(slope1)), 
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
subplot(3,6,1), plot(mean1-mean(mean1)), 
title('Mean'), 
xticks([1:size(PhasePrecession(i).Slope,2)])
xticklabels(animal.state_labels)
subplot(3,6,7), boxplot((mean1-mean(mean1))')
xticks([1:size(PhasePrecession(i).Slope,2)])
xticklabels(animal.state_labels)
title('Mean')
subplot(3,6,2), plot(std1-mean(std1)), 
title('Std'), 
xticks([1:size(PhasePrecession(i).Slope,2)])
xticklabels(animal.state_labels)
subplot(3,6,8), boxplot((std1-mean(std1))')
xticks([1:size(PhasePrecession(i).Slope,2)])
xticklabels(animal.state_labels)
title('Std')
subplot(3,6,3), plot(skewness1-mean(skewness1)), 
title('Skewness'), 
xticks([1:size(PhasePrecession(i).Slope,2)])
xticklabels(animal.state_labels)
subplot(3,6,9), boxplot((skewness1-mean(skewness1))')
xticks([1:size(PhasePrecession(i).Slope,2)])
xticklabels(animal.state_labels)
title('Skewness')

% PHASE 
subplot(3,6,4), plot(mean2-mean(mean2)), 
title('Mean phase'), 
xticks([1:size(PhasePrecession(i).Slope,2)])
xticklabels(animal.state_labels)
subplot(3,6,10), boxplot((mean2-mean(mean2))')
xticks([1:size(PhasePrecession(i).Slope,2)])
xticklabels(animal.state_labels)
title('Mean phase')
subplot(3,6,5), plot(std2-mean(std2)), 
title('Std phase'), 
xticks([1:size(PhasePrecession(i).Slope,2)])
xticklabels(animal.state_labels)
subplot(3,6,11), boxplot((std2-mean(std2))')
xticks([1:size(PhasePrecession(i).Slope,2)])
xticklabels(animal.state_labels)
title('Std phase')
subplot(3,6,6), plot(skewness2-mean(skewness2)), 
title('Skewness phase'), 
xticks([1:size(PhasePrecession(i).Slope,2)])
xticklabels(animal.state_labels)
subplot(3,6,12), boxplot((skewness2-mean(skewness2))')
xticks([1:size(PhasePrecession(i).Slope,2)])
xticklabels(animal.state_labels)
title('Skewness phase')

% Ratios
ratio0 = mean1-mean(mean1);
ratio1 = [ratio0(1,:)-ratio0(2,:);ratio0(3,:)-ratio0(2,:)];
subplot(3,6,13), plot(ratio1(1,:),ratio1(2,:),'o'), 
title('Mean')

ratio0 = std1-mean(std1);
ratio1 = [ratio0(1,:)-ratio0(2,:);ratio0(3,:)-ratio0(2,:)];
subplot(3,6,14), plot(ratio1(1,:),ratio1(2,:),'o'), 
title('Std')

ratio0 = skewness1-mean(skewness1);
ratio1 = [ratio0(1,:)-ratio0(2,:);ratio0(3,:)-ratio0(2,:)];
subplot(3,6,15), plot(ratio1(1,:),ratio1(2,:),'o'), 
title('Skewness')

ratio0 = mean2-mean(mean2);
ratio1 = [ratio0(1,:)-ratio0(2,:);ratio0(3,:)-ratio0(2,:)];
subplot(3,6,16), plot(ratio1(1,:),ratio1(2,:),'o'), 
title('Mean phase')

ratio0 = std2-mean(std2);
ratio1 = [ratio0(1,:)-ratio0(2,:);ratio0(3,:)-ratio0(2,:)];
subplot(3,6,17), plot(ratio1(1,:),ratio1(2,:),'o'), 
title('Std phase')

ratio0 = skewness2-mean(skewness2);
ratio1 = [ratio0(1,:)-ratio0(2,:);ratio0(3,:)-ratio0(2,:)];
subplot(3,6,18), plot(ratio1(1,:),ratio1(2,:),'o'), 
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

%% % Stability measure across place cells
out = plotPlaceFieldStability(units,trials,sr);
% out = plotPlaceFieldStability_v2(units,trials,sr);

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

%% % Ripples
if exist('ripples.mat') ~=2
    detected_swr = detect_swr(recording.name, [8, 3, 7, 12, 2, 9, 13, 1, 6, 14, 0, 10, 15, 4, 5, 11]+1)
    ripples = detected_swr;
    save('ripples.mat','ripples')
else
    load('ripples.mat')
end
load('animal.mat')
ripples.pos = interp1(animal.time,animal.pos', ripples.Ts(:,1));
figure, plot(animal.pos(1,:),animal.pos(2,:),'-'), hold on
plot(ripples.pos(:,1),ripples.pos(:,2),'ok')
draw now

ripple_ave = [];
sr_eeg = recording.sr/16;
signal = 0.000050354 * double(LoadBinary([recording.name '.eeg'],'nChannels',recording.nChannels,'channels',recording.ch_ripple,'precision','int16','frequency',recording.sr/16));
Fpass = [30,300];
Wn_theta = [Fpass(1)/(sr_eeg/2) Fpass(2)/(sr_eeg/2)]; % normalized by the nyquist frequency
[btheta,atheta] = butter(3,Wn_theta);
signal_filtered = filtfilt(btheta,atheta,signal)';
% ripple frequency
freqlist = 10.^(log10(20):log10(24)-log10(20):log10(320));

wt = spectrogram(signal_filtered,sr_eeg/10,sr_eeg/10-1,freqlist,sr_eeg);
clear signal_filtered signal
for i = 1:size(ripples.Ts,1)
    ripple_ave(:,:,i) = wt(:,round(ripples.Ts(i,2)*sr_eeg)-100:round(ripples.Ts(i,2)*sr_eeg)+100);
end
figure, imagesc([-100:100]/sr_eeg,Fpass,mean(abs(ripple_ave),3)),set(gca, 'Ydir', 'normal'),title('Ripples')
set(gca,'yscale','log')
