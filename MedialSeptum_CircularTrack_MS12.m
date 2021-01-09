% Medial Septum Circular Track
clear all, close all
MedialSeptum_Recordings
% MS10: 61 (Peter_MS10_170317_153237_concat) OK
%       62 (Peter_MS10_170314_163038) % gamma id: 62 OK
%       63 (Peter_MS10_170315_123936) 
%       64 (Peter_MS10_170307_154746_concat) OK 
% MS12: 78 (Peter_MS12_170714_122034_concat) OK 
%       79 (Peter_MS12_170715_111545_concat) OK 
%       80 (Peter_MS12_170716_172307_concat) OK 
%       81 (Peter_MS12_170717_111614_concat) OK 
%       83 (Peter_MS12_170719_095305_concat) OK 
% MS13: 92 (Peter_MS13_171129_105507_concat) OK 
%       93 (Peter_MS13_171130_121758_concat) OK 
%       88 (Peter_MS13_171110_163224_concat) No good cooling behavior
%       91 (Peter_MS13_171128_113924_concat) OK 
%       94 (Peter_MS13_171201_130527_concat) OK 
% MS21: 126 (Peter_MS21_180629_110332_concat) OK 
%       140 (Peter_MS21_180627_143449_concat) OK 
%       143 (Peter_MS21_180719_155941_concat, control)
%       149 (Peter_MS21_180625_153927_concat) OK 
%       153 (Peter_MS21_180712_103200_concat) OK 
%       151 (Peter_MS21_180628_155921_concat) OK 
%       154 (Peter_MS21_180719_122733_concat, control, duplicated)
%       159 (Peter_MS21_180807_122213_concat, control)
% MS22: 139 (Peter_MS22_180628_120341_concat) OK 
%       127 (Peter_MS22_180629_110319_concat) OK 
%       144 (Peter_MS22_180719_122813_concat, control)
%       168 (Peter_MS22_180720_110055_concat) OK 
%       166 (Peter_MS22_180711_112912_concat) OK 
% TODO:


idsToLoad = [126,127,140,93,78,81,168,166,151,149]; % 79, 80
id = 140 % 173, 169
% Control sets
% MS13: 73,177
% MS14: 107
% MS18: 116,141,142
% MS21: 143, 154 (duplicated), 159
% MS22: 144


% TODO
% New sessions (11-09-2019): 
% MS13: 173 OK,174 OK,175 OK,176 tracking missing,178,
% MS12: 179 OK, 180 OK,
% MS10: 181 no temperature?,182 no temperature?
% MS14: ids 102 102 103 104 105 106 107 108

recording = recordings(id);
% if ~isempty(recording.dataroot)
%     datapath = recording.dataroot;
% end

[session, basename, basepath, clusteringpath] = db_set_session('sessionName',recording.name);
% if isempty(session.epochs.duration) | session.epochs.duration == 0
%     disp('Updating DB')
% %     session = db_update_session(session);
% end

cd(fullfile(datapath, recording.animal_id, recording.name))
Intan_rec_info = read_Intan_RHD2000_file_from_basepath(fullfile(datapath, recording.animal_id, recording.name));
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
maze = recording.maze;
maze.radius_in = 96.5/2;
maze.radius_out =  116.5/2;
maze.arm_half_width = 4;
maze.cross_radii = 47.9;
maze.polar_rho_limits = [44,65];
maze.polar_theta_limits = [-2.8,2.8]*maze.radius_in;
maze.pos_x_limits = [-10,10]; % -15
maze.pos_y_limits = [-40,45];

maze.boundary{1} = [0,40];
maze.boundary{2} = [0,25];
maze.boundary{3} = [-15,40];
maze.boundary{4} = [15,40];
maze.boundary{5} = [maze.radius_in-3.25,maze.polar_theta_limits(2)];

% maze.boundary{5} = [maze.radius_in-3.25,150/180*pi*maze.radius_in];

recording.sr_lfp = sr/16;
% track_boundaries = recording.track_boundaries;
arena = recording.arena;
% nb_lfp_periods = ceil((time_frame(2)-time_frame(1))/lfp_periods);
temp_ = dir(fname);
recording_length = round(temp_.bytes/sr/nChannels/2)-1;
animal = [];

% Loading digital inputs
disp('1. Loading digital inputs')
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

% Camera tracking: Loading position data
disp('2. Loading Camera tracking data')
if inputs.ch_camera_sync
    if ~exist('Camera.mat')
        colors = [1,3]; % RGB based
        if length(recording.Cameratracking.Behavior)>1
            for i = 1:length(recording.Cameratracking.Behavior)
                Camera{i} = ImportCameraData(recording.Cameratracking.Behavior{i},pwd,colors,arena);
            end
        else
            Camera = ImportCameraData(recording.Cameratracking.Behavior,pwd,colors,arena);
        end
    else
        load('Camera.mat');
    end
end

% Optitrack: Loading position data
disp('3. Loading Optitrack tracking data')
if inputs.ch_OptiTrack_sync
    if ~exist('Optitrack.mat')
        Optitrack = LoadOptitrack(recording.OptiTracktracking,1,arena,0,0);
        save('Optitrack.mat','Optitrack')
    else
        load('Optitrack.mat');
    end
end
if ~isempty(recording.OptiTracktracking_offset)
    Optitrack.position3D = Optitrack.position3D + recording.OptiTracktracking_offset';
end
if ~isempty(recording.OptiTracktracking_scaling)
    Optitrack.position3D = Optitrack.position3D/recording.OptiTracktracking_scaling;
    Optitrack.animal_speed = Optitrack.animal_speed/recording.OptiTracktracking_scaling;
    Optitrack.animal_acceleration = Optitrack.animal_acceleration/recording.OptiTracktracking_scaling;
end

prebehaviortime = 0;
if recording.concat_behavior_nb > 0
    prebehaviortime = 0;
    if all(recording.concat_behavior_nb > 1)
        for i = 1:recording.concat_behavior_nb-1
            fullpath = fullfile([datapath,recording.animal_id], recording.concat_recordings{i}, [recording.concat_recordings{i}, '.dat']);
            temp2_ = dir(fullpath);
            prebehaviortime = prebehaviortime + temp2_.bytes/nChannels/2/sr;
        end
    end
    behaviortime = 0;
    for i = 1:length(recording.concat_behavior_nb)
        i1 = recording.concat_behavior_nb(i);
        fullpath = fullfile([datapath, recording.animal_id], recording.concat_recordings{i1}, [recording.concat_recordings{i1}, '.dat']);
        temp2_ = dir(fullpath);
        behaviortime = behaviortime+temp2_.bytes/nChannels/2/sr;
    end
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
    
    if isfield(recording,'optiTrack_prepulses') && ~isempty(recording.optiTrack_prepulses)
        disp('Removing prepulses')
        animal.time = animal.time(recording.optiTrack_prepulses+1:end);
    end
    animal.pos  = Optitrack.position3D;
    animal.pos(:,find(animal.pos(2,:)>70)) = 0;
    gausswin_size = animal.sr/2;
    for i = 1:3
        animal.pos(i,:) = medfilt1(animal.pos(i,:),5);
        animal.pos(i,:) = nanconv(animal.pos(i,:),gausswin(gausswin_size)','edge');
    end
    animal.speed  = Optitrack.animal_speed;
    animal.acceleration = [0,diff(nanconv(Optitrack.animal_speed,gausswin(1,50)'./sum(gausswin(50,1)),'edge'))];
    if size(animal.time,2) < size(animal.pos,2)
        warning(['There are fewer Optitrack digital pulses than position points: ',num2str(size(animal.time,2) - size(animal.pos,2))])
        animal.pos = animal.pos(:,1:length(animal.time));
        animal.speed = animal.speed(1:length(animal.time));
        animal.acceleration = animal.acceleration(1:length(animal.time));
    end
    if size(animal.time,2) > size(animal.pos,2)
        warning(['There are more Optitrack digital pulses than position points: ', num2str(size(animal.time,2) - size(animal.pos,2))])
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

for fn = fieldnames(maze)'
    animal.(fn{1}) = maze.(fn{1});
end

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
if recording.cooling_session == 0
    cooling.onsets = animal.time(round(length(animal.time)./3));
    cooling.offsets = animal.time(round(2*length(animal.time)./3));
    cooling.cooling = [cooling.onsets(1)+10;cooling.offsets(1)];
    cooling.nocooling = [[1,cooling.onsets(1)];[cooling.offsets(1)+120,behaviortime+prebehaviortime]]';
else
    if inputs.ch_temp ~= 0
%         temperature.temp_smooth = nanconv(temperature.temp',gausswin(100*temperature.sr)./sum(gausswin(100*temperature.sr)),'edge')';
        temp_range = [32,34];% temp_1 defines the upper limit on cooling, temp_2 the lower limit on no cooling
        t_start = find(temperature.time>prebehaviortime,1)
        t_end = find(temperature.time>prebehaviortime+behaviortime,1)
        if isempty(t_end)
            t_end = length(temperature.time)
        end
        test = find(diff(temperature.temp(t_start:t_end) < temp_range(1),2)== 1);
        test = test+t_start;
        test(diff(test)<100*temperature.sr)=[];
        cooling.onsets = temperature.time(test(temperature.temp(test+1)-temperature.temp(test)<0));
        cooling.offsets = temperature.time(test(temperature.temp(test+1)-temperature.temp(test)>0));
        if length(cooling.offsets)<length(cooling.onsets) & cooling.onsets(end) > cooling.offsets(end)
            cooling.offsets = [cooling.offsets,temperature.time(end)];
        elseif length(cooling.offsets)<length(cooling.onsets) & cooling.onsets(end) < cooling.offsets(end)
            cooling.onsets = cooling.onsets(1:end-1);
            warning('one cooling onset timestamp removed');
        end
        cooling.cooling = [cooling.onsets;cooling.offsets];
        cooling.cooling2 = [cooling.onsets-20;cooling.offsets+180];
        cooling.nocooling = reshape([prebehaviortime;cooling.cooling2(:);prebehaviortime+behaviortime],[2,size(cooling.cooling2,2)+1]);
    elseif inputs.ch_CoolingPulses ~= 0
        cooling.onsets = digital_on{inputs.ch_CoolingPulses}/sr;
        cooling.offsets = cooling.onsets + 12*60;
        cooling.cooling = [cooling.onsets(1)+10;cooling.offsets(1)];
        cooling.nocooling = [[1,cooling.onsets'];[cooling.offsets'+120,behaviortime]]';
    else
        disp('Getting cooling intervals from recording metadata')
        cooling.onsets = recording.cooling_onsets;
        cooling.offsets = recording.cooling_offsets;
        cooling.cooling = [];
        cooling.nocooling = [];
        for i = 1:size(cooling.onsets,2)
            cooling.cooling = [cooling.cooling;[cooling.onsets(i)+10,cooling.offsets(i)]+prebehaviortime];
            if i == 1
                cooling.nocooling = [cooling.nocooling;[1,cooling.onsets(1)]+prebehaviortime];
            else
                cooling.nocooling = [cooling.nocooling;[cooling.offsets(i-1),cooling.onsets(i)]+prebehaviortime];
            end
        end
        cooling.nocooling = [cooling.nocooling;[cooling.offsets(end),behaviortime]+prebehaviortime]';
        cooling.cooling = cooling.cooling';
    end
end

% Separating left and right trials
disp('7. Defining trials for the behavior')
[trials,animal] = trials_thetamaze(animal, maze,[],cooling);
animal.state = trials.state;
trials.labels = {'Left','Right'};
trials.total = length(trials.start);
% save('animal.mat','animal')
disp('8. Defining cooling trials')
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

if ~isempty(temperature) & any(temperature.temp<0)
    temperature.temp(find(temperature.temp<0)) = 37;
end

if ~isempty(temperature)
    for i = 1:trials.total
        if round(animal.time(trials.end(i))*temperature.sr)<=length(temperature.temp)
            trials.temperature(i) = mean(temperature.temp(round(animal.time(trials.start(i))*temperature.sr):round(animal.time(trials.end(i))*temperature.sr)));
        else
            trials.temperature(i) = 37;
            warning('No temperature data for select trials')
        end
    end
end
save('trials.mat','trials')

%
disp('8. Loading instantaneous theta')
theta = [];
theta.sr = recording.sr_lfp;
% recording.ch_theta = 80
theta.ch_theta = recording.ch_theta;
theta.sr_freq = 10;
InstantaneousTheta = calcInstantaneousTheta(recording);

theta.phase = InstantaneousTheta.signal_phase{recording.ch_theta};
theta.phase2 = InstantaneousTheta.signal_phase2{recording.ch_theta};
theta.freq = InstantaneousTheta.signal_freq{recording.ch_theta};
theta.power = InstantaneousTheta.signal_power{recording.ch_theta};

% theta18 = InstantaneousTheta;
% theta81 = InstantaneousTheta;

% theta80 = InstantaneousTheta;
%
% %
% figure, plot(theta81.ThetaInstantTime,nanconv(theta81.ThetaInstantFreq,gausswin(800)','edge'),'-b'), hold on, plot(theta18.ThetaInstantTime,nanconv(theta18.ThetaInstantFreq,gausswin(800)','edge'),'-m'), plot(theta61.ThetaInstantTime,nanconv(theta61.ThetaInstantFreq,gausswin(800)','edge'),'-r'), plot(theta80.ThetaInstantTime,nanconv(theta80.ThetaInstantFreq,gausswin(800)','edge'),'-c'), plot(temperature.time,temperature.temp,'-k')
% %
disp('9. Linearizing position')
animal.pos_linearized = linearize_pos(animal,recording.arena);
animal.pos_linearized_limits = [0,diff(animal.pos_y_limits) + diff(animal.polar_theta_limits)-5];

% Redefining temperature effects
if inputs.ch_temp == 0
    temperature.temp = 37 * ones(1,length(theta.phase));
    temperature.sr = recording.sr_lfp;
    temperature.time = [1:length(temperature.temp)]/recording.sr_lfp;
    [t,gde] = alphafunction2(0);
    gde = interp1(t,gde,[t(1):(1/temperature.sr):t(end)]);
    injector = zeros(1,length(theta.phase));
    injector(round(cooling.onsets*temperature.sr)) = 1;
    injector = conv(injector,-gde);
    injector = injector(1:length(theta.phase));
    injector_amplitude = 15;
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
    plot(temperature.time,temperature.temp), xlim([prebehaviortime,prebehaviortime+behaviortime]), title('Temperature'), hold on
    ylim([20,43])
end
plot(cooling.nocooling',[40,40],'color','red','linewidth',2)
plot(cooling.cooling,[40,40],'color','blue','linewidth',2)
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
% save('animal.mat','animal')

% temperature vs Theta frequency vs temperature
if ~isempty(temperature)
    disp('Plotting Theta vs Temperature')
%     animal.theta = interp1(InstantaneousTheta.timestamps,InstantaneousTheta.ThetaInstantFreq{recording.ch_theta},animal.time);
     animal.theta = interp1([1:length(theta.power)]/theta.sr_freq,theta.freq,animal.time);
    animal.thetapower = interp1([1:length(theta.power)]/theta.sr_freq,theta.power,animal.time);
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
% plot_ThetaVsSpeed(recording,animal,cooling);
% saving cooling as manipulation
cooling.timestamps = [cooling.onsets;cooling.offsets];
sucess = saveStruct(cooling,'manipulation','session',session);

% Saving animal tracking data
sucess = saveStruct(animal,'behavior','session',session);
% Saving trials data
sucess = saveStruct(trials,'behavior','session',session);
% Saving temperature data
sucess = saveStruct(temperature,'timeseries','session',session);

% Theta vs speed
interval = [animal.time(1),animal.time(end)];
session.extracellular.srLfp = 1250;
% plot_ThetaVsSpeed3(session,animal,cooling,interval,120); 


%% % Spikes
spikes = loadSpikes('clusteringpath',recording.SpikeSorting.path,'clusteringformat',recording.SpikeSorting.method,'basename',recording.name);
if ~exist([recording.name, '.lfp'])
    disp('Creating lfp file')
    downsample_dat_to_eeg(recording.name,pwd);
end
sr_eeg = recording.sr/16;

for i = 1:size(spikes.ts,2)
    spikes.ts{i} = spikes.ts{i}(spikes.times{i} < length(theta.phase)/sr_eeg);
    spikes.times{i} = spikes.ts{i}/sr;
    spikes.total(i) = length(spikes.ts{i});
    spikes.ts_eeg{i} = ceil(spikes.ts{i}/16);
    spikes.theta_phase{i} = theta.phase(spikes.ts_eeg{i});
    spikes.theta_phase2{i} = theta.phase2(spikes.ts_eeg{i});
    spikes.theta_freq{i} = interp1(InstantaneousTheta.timestamps,InstantaneousTheta.ThetaInstantFreq{recording.ch_theta},spikes.times{i});
    % spikes.theta_freq{i} = interp1([1:length(theta.freq)]/theta.sr_freq,theta.freq,spikes.times{i});
    spikes.speed{i} = interp1(animal.time,animal.speed,spikes.ts{i}/sr);
    spikes.pos{i} = interp1(animal.time,animal.pos',spikes.ts{i}/sr)';
    spikes.pos_linearized{i} = interp1(animal.time,animal.pos_linearized,spikes.times{i});
    spikes.polar_theta{i} = interp1(animal.time,animal.polar_theta,spikes.ts{i}/sr);
    spikes.polar_rho{i} = interp1(animal.time,animal.polar_rho,spikes.ts{i}/sr);
    spikes.arm{i} = zeros(1,length(spikes.ts{i}));
    spikes.arm{i}(spikes.pos{i}(1,:) > animal.pos_x_limits(1) & spikes.pos{i}(1,:) < animal.pos_x_limits(2) & spikes.pos{i}(2,:) > animal.pos_y_limits(1) & spikes.pos{i}(2,:) < animal.pos_y_limits(2)) = 1;
    spikes.rim{i} = zeros(1,length(spikes.ts{i}));
    spikes.rim{i}(spikes.polar_rho{i} > animal.polar_rho_limits(1) & spikes.polar_rho{i} < animal.polar_rho_limits(2) & spikes.polar_theta{i} > animal.polar_theta_limits(1) & spikes.polar_theta{i} < animal.polar_theta_limits(2)) = 1;
    %spikes.trials{i} = interp1(animal.time+prebehaviortime/sr,trials.trials,spikes.ts{i}/sr,'nearest');
    spikes.state{i} = interp1(animal.time,trials.state,spikes.ts{i}/sr,'nearest');
    spikes.temperature{i} = interp1(temperature.time,temperature.temp,spikes.times{i});
    spikes.trials{i} = interp1(animal.time,trials.trials2,spikes.ts{i}/sr,'nearest');
    
    % Linearized and colored and grouped by left/right trials
    
    trials2 = trials;
    trials2.state(find(~isnan(trials2.state))) = trials.stat(trials.trials2(find(~isnan(trials2.state))));
    behavior2 = animal;
    behavior2.state = trials.state;
    behavior2.state(find(~isnan(trials2.state))) = trials.stat(trials.trials2(find(~isnan(trials2.state))));
    behavior2.state_labels = trials.labels;
    spikes.state_leftright{i} = interp1(animal.time,behavior2.state,spikes.ts{i}/sr,'nearest');
end

if isfield(recording.SpikeSorting,'polar_theta_placecells')
    for i = 1:size(spikes.ts,2)  %size(recording.SpikeSorting.polar_theta_placecells,2)
        if spikes.cluID(i) <= length(recording.SpikeSorting.polar_theta_placecells)
            if ~isempty(recording.SpikeSorting.polar_theta_placecells{spikes.cluID(i)})
                spikes.PhasePrecession{i}.placefields_polar_theta = recording.SpikeSorting.polar_theta_placecells{spikes.cluID(i)};
            end
        end
    end
end

if isfield(recording.SpikeSorting,'center_arm_placecells')
    for i = 1:size(spikes.ts,2)  %size(recording.SpikeSorting.polar_theta_placecells,2)
        if spikes.cluID(i) <= length(recording.SpikeSorting.center_arm_placecells)
            if ~isempty(recording.SpikeSorting.center_arm_placecells{spikes.cluID(i)})
                spikes.PhasePrecession{i}.placefields_center_arm = recording.SpikeSorting.center_arm_placecells{spikes.cluID(i)};
            end
        end
    end
end

saveStruct(spikes,'cellinfo','session',session);

% save(fullfile(recording.SpikeSorting.path,[recording.name,'.spikes.cellinfo.mat']),'spikes')
disp('done')

%%
disp('Plotting Phase precession')
figure,
for i = 1:size(spikes.ts,2)
    subplot(1,3,1)
    plot(spikes.pos{i}(1,:),spikes.pos{i}(2,:),'.','markersize',8), hold on
    legend({'1','2','3','4','5','6','7','8','9'})
    plot_ThetaMaze(maze)
    axis equal
    xlim([-65,65]),ylim([-65,65]),
    subplot(1,3,2)
    histogram(spikes.polar_theta{i},'Normalization','probability'), hold on
    subplot(1,3,3)
    plot(spikes.polar_theta{i},spikes.polar_rho{i},'.','markersize',8), hold on
end

% % Plotting Phase precession
behavior = animal; % Animal Strucutre
behavior.speed_th = 10;
behavior.rim = animal.rim;
behavior.state = trials.state;
behavior.pos = animal.polar_theta;
behavior.pos_limits = animal.polar_theta_limits;
% behavior.optogenetics.pos = optogenetics.polar_theta(optogenetics.rim);
% behavior.optogenetics.trial = optogenetics.trials(optogenetics.rim);
spikes2 = [];
for i = 1:size(spikes.ts,2)
    spikes2.ts{i} = spikes.ts{i};
    spikes2.times{i} = spikes.times{i};
    spikes2.cluID(i) = spikes.cluID(i);
    spikes2.UID(i) = spikes.UID(i);
    spikes2.total(i)= spikes.total(i);
    spikes2.trials{i} = spikes.trials{i};
    spikes2.theta_freq{i} = spikes.theta_freq{i};
    spikes2.speed{i} = spikes.speed{i};
end
% Polar plot - PHASE PRECESSION
if isfield(recording.SpikeSorting,'polar_theta_placecells')
    for i = 1:size(spikes2.ts,2)  %size(recording.SpikeSorting.polar_theta_placecells,2)
        if spikes2.cluID(i) <= length(recording.SpikeSorting.polar_theta_placecells)
            if ~isempty(recording.SpikeSorting.polar_theta_placecells{spikes2.cluID(i)})
                spikes2.PhasePrecession{i} = recording.SpikeSorting.polar_theta_placecells{spikes2.cluID(i)};
            else
                spikes2.PhasePrecession{i} = [];
            end
        else
            spikes2.PhasePrecession{i} = [];
        end
    end
end
% spikes2.PhasePrecession{1} = [];

PhasePrecessionSlope1 = plot_FiringRateMap('animal',behavior,'spikes',spikes2,'trials',trials,'theta',theta,'textlabel','rim');

% Firing rate on center arm
behavior = animal;
behavior.speed_th = 10;
behavior.state = trials.state;
behavior.('pos') = animal.pos(2,:);
behavior.rim = animal.arm;
behavior.pos_limits = animal.pos_y_limits;
% behavior.optogenetics.pos = optogenetics.pos(2,optogenetics.arm);
% behavior.optogenetics.trial = optogenetics.trials(optogenetics.arm);
spikes2 = [];
for i = 1:size(spikes.ts,2)
    spikes2.ts{i} = spikes.ts{i};
    spikes2.cluID(i) = spikes.cluID(i);
    spikes2.UID(i) = spikes.UID(i);
    spikes2.total(i)= spikes.total(i);
    spikes2.times{i} = spikes.times{i};
    spikes2.trials{i} = spikes.trials{i};
    spikes2.theta_freq{i} = spikes.theta_freq{i};
    spikes2.speed{i} = spikes.speed{i};
end
% Center arm - PHASE PRECESSION
if isfield(recording.SpikeSorting,'center_arm_placecells')
    for i = 1:size(spikes2.ts,2)  %size(recording.SpikeSorting.polar_theta_placecells,2)
        if spikes2.cluID(i) <= length(recording.SpikeSorting.center_arm_placecells)
            if ~isempty(recording.SpikeSorting.center_arm_placecells{spikes2.cluID(i)})
                spikes2.PhasePrecession{i} = recording.SpikeSorting.center_arm_placecells{spikes2.cluID(i)};
            else
                spikes2.PhasePrecession{i} = [];
            end
        else
            spikes2.PhasePrecession{i} = [];
        end
    end
end
PhasePrecessionSlope2 = plot_FiringRateMap('animal',behavior,'spikes',spikes2,'trials',trials,'theta',theta,'textlabel','arm');
% save('PhasePrecessionSlope2.mat','PhasePrecessionSlope1','PhasePrecessionSlope2')

%
m = 1;
slope1 = [];
mean1 = []; median1 = []; std1 = []; skewness1 = []; % Position
mean2 = []; median2 = []; std2 = []; skewness2 = []; % Theta phase
OscillationFreq = [];
SpikeCount1 = []; Slope1 = []; R1 = [];
theta_freq = [];
speed = [];
PeakRate = [];
PeakRate = [];
FieldSize = [];
FieldWidth = [];

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
            OscillationFreq(j,m:m+elements-1) = PhasePrecession(i).OscillationFreq(j);
            theta_freq(j,m:m+elements-1) = PhasePrecession(i).theta_freq(j);
            speed(j,m:m+elements-1) = PhasePrecession(i).speed(j);
            PeakRate(j,m:m+elements-1) = PhasePrecession(i).Slope{j}.PeakRate;
            AverageRate(j,m:m+elements-1) = PhasePrecession(i).Slope{j}.AverageRate;
            FieldSize(j,m:m+elements-1) = PhasePrecession(i).Slope{j}.FieldSize;
            FieldWidth(j,m:m+elements-1) = PhasePrecession(i).Slope{j}.FieldWidth;
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
                OscillationFreq(j,m:m+elements-1) = PhasePrecession(i).OscillationFreq(j);
                theta_freq(j,m:m+elements-1) = PhasePrecession(i).theta_freq(j);
                speed(j,m:m+elements-1) = PhasePrecession(i).speed(j);
                PeakRate(j,m:m+elements-1) = PhasePrecession(i).Slope{j}.PeakRate;
                AverageRate(j,m:m+elements-1) = PhasePrecession(i).Slope{j}.AverageRate;
                FieldSize(j,m:m+elements-1) = PhasePrecession(i).Slope{j}.FieldSize;
                FieldWidth(j,m:m+elements-1) = PhasePrecession(i).Slope{j}.FieldWidth;
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
save('PhasePrecessionSlope.mat','PhasePrecessionSlope1','PhasePrecessionSlope2','slope1','mean1','std1','skewness1','mean2','std2','skewness2','OscillationFreq','theta_freq','speed','PeakRate','AverageRate','FieldSize','FieldWidth')


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


% % % % % % %
figure,
subplot(2,4,1), plot(OscillationFreq), hold on, plot(theta_freq,'k')
title('Oscillation Frequency'),
xticks([1:size(PhasePrecession(i).Slope,2)])
xticklabels(animal.state_labels)
subplot(2,4,2),
plot(OscillationFreq-theta_freq), title('Oscillation Frequency-theta')
xticks([1:size(PhasePrecession(i).Slope,2)])
xticklabels(animal.state_labels)
subplot(2,4,3),
plot(speed),title('Animal speed'),
subplot(2,4,4),
plot(OscillationFreq(1,:)./OscillationFreq(2,:),OscillationFreq(3,:)./OscillationFreq(2,:),'o'), axis equal, hold on, plot([0,2],[0,2]), plot([1,1],[0,2]), plot([0,2],[1,1])
title('Ratio'), xlabel('Pre/Cooling'), ylabel('Post/Cooling')
subplot(2,4,5),
boxplot(OscillationFreq')
subplot(2,4,6),
boxplot((OscillationFreq-theta_freq)')
subplot(2,4,7),
boxplot(speed')
subplot(2,4,8),
boxplot([OscillationFreq(1,:)./OscillationFreq(2,:);OscillationFreq(3,:)./OscillationFreq(2,:)]')
title('Ratios'),
xticks([1,2])
xticklabels({'Pre/Cooling','Post/Cooling'})



% % % % %
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
subplot(3,6,2), plot(std1),
title('Std'),
xticks([1:size(PhasePrecession(i).Slope,2)])
xticklabels(animal.state_labels)
subplot(3,6,8), boxplot((std1)')
xticks([1:size(PhasePrecession(i).Slope,2)])
xticklabels(animal.state_labels)
title('Std')
subplot(3,6,3), plot(skewness1),
title('Skewness'),
xticks([1:size(PhasePrecession(i).Slope,2)])
xticklabels(animal.state_labels)
subplot(3,6,9), boxplot((skewness1)')
xticks([1:size(PhasePrecession(i).Slope,2)])
xticklabels(animal.state_labels)
title('Skewness')

% PHASE
mean2(mean2<0) = mean2(mean2<0)+2*pi;
subplot(3,6,4), plot(mean2),
title('Mean phase'),
xticks([1:size(PhasePrecession(i).Slope,2)])
xticklabels(animal.state_labels)
subplot(3,6,10), boxplot((mean2)')
xticks([1:size(PhasePrecession(i).Slope,2)])
xticklabels(animal.state_labels)
title('Mean phase')
subplot(3,6,5), plot(std2),
title('Std phase'),
xticks([1:size(PhasePrecession(i).Slope,2)])
xticklabels(animal.state_labels)
subplot(3,6,11), boxplot((std2)')
xticks([1:size(PhasePrecession(i).Slope,2)])
xticklabels(animal.state_labels)
title('Std phase')
subplot(3,6,6), plot(skewness2-mean(skewness2)),
title('Skewness phase'),
xticks([1:size(PhasePrecession(i).Slope,2)])
xticklabels(animal.state_labels)
subplot(3,6,12), boxplot((skewness2)')
xticks([1:size(PhasePrecession(i).Slope,2)])
xticklabels(animal.state_labels)
title('Skewness phase')

% Ratios
ratio0 = mean1;
ratio1 = [ratio0(1,:)-ratio0(2,:);ratio0(3,:)-ratio0(2,:)];
subplot(3,6,13), plot(ratio1(1,:),ratio1(2,:),'o'),
title('Mean')

ratio0 = std1;
ratio1 = [ratio0(1,:)-ratio0(2,:);ratio0(3,:)-ratio0(2,:)];
subplot(3,6,14), plot(ratio1(1,:),ratio1(2,:),'o'),
title('Std')

ratio0 = skewness1;
ratio1 = [ratio0(1,:)-ratio0(2,:);ratio0(3,:)-ratio0(2,:)];
subplot(3,6,15), plot(ratio1(1,:),ratio1(2,:),'o'),
title('Skewness')

ratio0 = mean2;
ratio1 = [ratio0(1,:)-ratio0(2,:);ratio0(3,:)-ratio0(2,:)];
subplot(3,6,16), plot(ratio1(1,:),ratio1(2,:),'o'),
title('Mean phase')

ratio0 = std2;
ratio1 = [ratio0(1,:)-ratio0(2,:);ratio0(3,:)-ratio0(2,:)];
subplot(3,6,17), plot(ratio1(1,:),ratio1(2,:),'o'),
title('Std phase')

ratio0 = skewness2;
ratio1 = [ratio0(1,:)-ratio0(2,:);ratio0(3,:)-ratio0(2,:)];
subplot(3,6,18), plot(ratio1(1,:),ratio1(2,:),'o'),
title('Skewness phase')

figure,
subplot(2,4,1)
plot(AverageRate), title('Average rate (Hz)')
ratio0 = AverageRate;
ratio1 = [ratio0(1,:)-ratio0(2,:);ratio0(3,:)-ratio0(2,:)];
subplot(2,4,5), plot(ratio1(1,:),ratio1(2,:),'o'),

subplot(2,4,2)
plot(PeakRate), title('Peak rate (Hz)')
ratio0 = PeakRate;
ratio1 = [ratio0(1,:)-ratio0(2,:);ratio0(3,:)-ratio0(2,:)];
subplot(2,4,6), plot(ratio1(1,:),ratio1(2,:),'o'),

subplot(2,4,3)
plot(FieldSize), title('FieldSize (cm)')
ratio0 = FieldSize;
ratio1 = [ratio0(1,:)-ratio0(2,:);ratio0(3,:)-ratio0(2,:)];
subplot(2,4,7), plot(ratio1(1,:),ratio1(2,:),'o'),

subplot(2,4,4)
plot(FieldWidth), title('FieldWidth (cm)')
ratio0 = FieldWidth;
ratio1 = [ratio0(1,:)-ratio0(2,:);ratio0(3,:)-ratio0(2,:)];
subplot(2,4,8), plot(ratio1(1,:),ratio1(2,:),'o'),

%% % % % %
% Firing rate linearized
behavior = animal;
behavior.state = trials.state;
behavior.pos = animal.pos_linearized;
behavior.rim = zeros(size(animal.arm));
behavior.rim(find(animal.arm | animal.rim)) = 1;
behavior.pos_limits = animal.pos_linearized_limits;
behavior.maze.boundaries = [diff(animal.pos_y_limits),diff(animal.pos_y_limits)+abs(animal.maze.polar_theta_limits(1))-5];
behavior.speed_th = 10;

spikes2 = [];

for i = 1:size(spikes.ts,2)
    spikes2.ts{i} = spikes.ts{i};
    spikes2.times{i} = spikes.times{i};
    spikes2.cluID(i) = spikes.cluID(i);
    spikes2.total(i) = spikes.total(i);
    spikes2.pos{i} = interp1(behavior.time,behavior.pos,spikes2.times{i});
    %     spikes2.pos_linearized{i} = interp1(behavior.time,behavior.pos_linearized,spikes2.times{i});
    spikes2.pos{i} = interp1(behavior.time,behavior.pos,spikes2.times{i});
    spikes2.speed{i} = interp1(behavior.time,behavior.speed,spikes2.times{i});
    spikes2.rim{i} = interp1(behavior.time,behavior.rim,spikes2.times{i},'nearest');
    spikes2.trials{i} = spikes.trials{i};
    spikes2.theta_freq{i} = spikes.theta_freq{i};
    spikes2.speed{i} = spikes.speed{i};
end


firingRateMap = plot_FiringRateMapAverage('animal',behavior,'spikes',spikes2);
for i= 1:size(firingRateMap.map,2)
    firingRateMap.map{i}(isnan(firingRateMap.map{i}))= 0;
end    
firingRateMap.total = spikes2.total./behaviortime;
firingRateMap.boundaries = behavior.maze.boundaries;

sucess = saveStruct(firingRateMap,'firingRateMap','session',session);
% save([recording.name, '.firingRateMap.firingRateMap.mat'],'firingRateMap')

CoolingStates = plot_FiringRateMap2('animal',behavior,'spikes',spikes2,'trials',trials);
CoolingStates.boundaries = behavior.maze.boundaries;
CoolingStates.labels = behavior.state_labels;
for i=1:length(behavior.state_labels)
    CoolingStates.trial_count(i) = length(unique(trials.trials2(find(trials.state==i))));
end
sucess = saveStruct(CoolingStates,'firingRateMap','session',session);
% save([recording.name, '.CoolingStates.firingRateMap.mat'],'CoolingStates')

% Linearized
% PhasePrecessionSlope_linearized = plot_FiringRateMap(behavior,spikes2,trials,theta,sr,'linearized_cooling_states');

% Linearized and colored and grouped by left/right trials
trials2 = trials;
trials2.state(find(~isnan(trials2.state))) = trials.stat(trials.trials2(find(~isnan(trials2.state))));
behavior2 = behavior;
behavior2.state(find(~isnan(trials2.state))) = trials.stat(trials.trials2(find(~isnan(trials2.state))));
behavior2.state_labels = trials.labels;
% PhasePrecessionSlope_linearized_left_right = plot_FiringRateMap(behavior2,spikes2,trials2,theta,sr,'linearized_left_right');

LeftRight = plot_FiringRateMap2('animal',behavior2,'spikes',spikes2,'trials',trials);
LeftRight.boundaries = behavior.maze.boundaries;
LeftRight.labels = {'Left','Right'};
sucess = saveStruct(LeftRight,'firingRateMap','session',session);
% save([recording.name, '.LeftRight.firingRateMap.mat'],'LeftRight')

RateMapTrials = calc_FiringRateMap_trials(animal,spikes,trials);
saveStruct(RateMapTrials,'firingRateMap','session',session);

%%
% Calculating the instantaneous theta frequency
figure
indexes = find(trials.trials2 > 0 & animal.speed>20);
plot(animal.time(indexes), animal.temperature(indexes)/4,'-k'), hold on
drawnow
average_left = zeros(64,length(indexes));
average_right = zeros(64,length(indexes));

for i = 1:128
    i
    theta = [];
    theta.sr = recording.sr_lfp;
    recording.ch_theta = i;
    theta.ch_theta = recording.ch_theta;
    theta.sr_freq = 10;
    InstantaneousTheta = calcInstantaneousTheta(recording);
    
    animal.theta = interp1(InstantaneousTheta.timestamps,InstantaneousTheta.ThetaInstantFreq,animal.time);
    
    if i<65
        average_left(i,:) = nanconv(animal.theta(indexes),gausswin(200)','edge');
        plot(animal.time(indexes), average_left(i,:),'.b'), hold on
    else
        average_right(i-64,:) = nanconv(animal.theta(indexes),gausswin(200)','edge');
        plot(animal.time(indexes), average_right(i-64,:),'.r'), hold on
    end
    drawnow
end
save('test.mat','animal','average_left','average_right','indexes')
figure
plot(animal.time(indexes), animal.temperature(indexes)/4,'-k'), hold on
x1 = animal.time(indexes); y1 = median(average_left); y_std = std(average_left);
patch([x1,flip(x1)]', [y1+y_std,flip(y1-y_std)],'b','EdgeColor','none','FaceAlpha',.2)
plot(x1, y1, 'b','linewidth',2), grid on

y2 = median(average_right); y_std = std(average_right);
patch([x1,flip(x1)]', [y2+y_std,flip(y2-y_std)],'r','EdgeColor','none','FaceAlpha',.2)
plot(x1, y2, 'r','linewidth',2), grid on, xlabel('Time (s)'),ylabel('Theta (Hz)'),title('Theta freq with temperature imposed')

figure, histogram(y1-y2), grid on, xlabel('Theta frequency difference (Hz)'),title('Hemisphere comparison')

%% Speed histogram for the maze
speed_bins = [20:5:160];
acceleration_bins = [-2:0.1:2];
theta_bins = [6:0.2:10];
theta_power_bins = [5:0.5:15];
trials_slow = find(((trials.end-trials.start)/animal.sr)>3);

bins_arm = [animal.pos_y_limits(1):5:animal.pos_y_limits(2)];
bins_rim = [animal.polar_theta_limits(1):-5,5:animal.polar_theta_limits(2)];

figure(100),
colormap9 = cool(trials.total);
animal.speed2 = nanconv(animal.speed,gausswin(animal.sr/2)'/sum(gausswin(animal.sr/2)),'edge');
animal.acceleration2 = nanconv(animal.acceleration,gausswin(animal.sr/2)'/sum(gausswin(animal.sr/2)),'edge');
trials_speed = [];
trials_acceleration = [];
trials_theta = [];
trials_time = [];
trials_temp = [];

for i = 1:trials.total
    indexes = find(trials.trials2 == i);
    if ~isempty(indexes) & ~any(trials.error==i) & ~any(trials_slow==i)
        figure(100)
        subplot(2,3,1)
        plot3(animal.pos(1,indexes),animal.pos(2,indexes),animal.speed2(indexes),'color',colormap9(i,:)), hold on, axis tight
        subplot(2,3,2)
        plot3(animal.pos(1,indexes),animal.pos(2,indexes),animal.acceleration2(indexes),'color',colormap9(i,:)), hold on, axis tight
        subplot(2,3,3)
        plot3(animal.pos(1,indexes),animal.pos(2,indexes),animal.theta(indexes),'color',colormap9(i,:)), hold on, axis tight
        
        trials_speed = [trials_speed,mean(animal.speed2(indexes))];
        trials_acceleration = [trials_acceleration,mean(abs(animal.acceleration2(indexes)))];
        trials_theta = [trials_theta,mean(animal.theta(indexes))];
        trials_time = [trials_time,mean(animal.time(indexes))];
        trials_temp = [trials_temp,mean(animal.temperature(indexes))];
        
        figure(102)
        plot3(animal.theta(indexes),animal.acceleration2(indexes),animal.speed2(indexes),'.'), hold on
    end
end
figure(102),
xlabel('theta(Hz)'), ylabel('Acceleration (cm/s^2)'), zlabel('Speed (cm/s)')

figure(100)
subplot(2,3,1), xlabel('x pos(cm)'), ylabel('y pos (cm)'), zlabel('Speed (cm/s)')
subplot(2,3,2), xlabel('x pos (cm)'), ylabel('y pos (cm)'), zlabel('Acceleration (cm/s^2)')
subplot(2,3,3), xlabel('x pos (cm)'), ylabel('y pos (cm)'), zlabel('Theta (Hz)')

subplot(2,3,4), plot(trials_speed,'.-k'), hold on, plot(nanconv(trials_speed,gausswin(11)'/11,'edge'),'r','linewidth',2), xlabel('Trials'), ylabel('Speed (cm/s)')
subplot(2,3,5), plot(trials_acceleration,'.-k'), hold on, plot(nanconv(trials_acceleration,gausswin(11)'/11,'edge'),'r','linewidth',2), xlabel('Trials'), ylabel('Acceleration (cm/s^2)')
subplot(2,3,6), plot(trials_theta,'.-k'), hold on, plot(nanconv(trials_theta,gausswin(11)'/11,'edge'),'r','linewidth',2), xlabel('Trials'), ylabel('Theta (Hz)'), axis tight, if inputs.ch_temp >0; yyaxis right, plot(trials_temp,'b'),ylabel('Temperature (C)'), end

[Lia2,~] = ismember(trials.trials2,[trials.error,trials_slow],'legacy');

figure,
subplot(3,2,1)
speed_histogram = [];

for j = 1:length(bins_arm)-1
    indexes = find(animal.pos(2,:) > bins_arm(j) & animal.pos(2,:) < bins_arm(j+1) & animal.arm & animal.speed2 > speed_bins(1) & ~Lia2 & ~isnan(trials.trials2));
    temp = histogram(animal.theta(indexes),theta_bins,'Normalization','probability');
    speed_histogram(:,j) = temp.Values;
end
imagesc(bins_arm,theta_bins,speed_histogram), set(gca,'YDir','normal'), title('Theta on the arm'), xlabel('Position (cm)'), ylabel('Theta (Hz)')

subplot(3,2,2)
speed_histogram = [];

for j = 1:length(bins_rim)-1
    indexes = find(animal.polar_theta > bins_rim(j) & animal.polar_theta < bins_rim(j+1) & animal.rim & animal.speed2 > speed_bins(1) & ~Lia2 & ~isnan(trials.trials2));
    temp = histogram(animal.theta(indexes),theta_bins,'Normalization','probability');
    speed_histogram(:,j) = temp.Values;
end
imagesc(bins_rim,theta_bins,speed_histogram), set(gca,'YDir','normal'), title('Theta on the rim'), xlabel('Position (cm)'), ylabel('Theta (Hz)')

subplot(3,2,3)
speed_histogram = [];

for j = 1:length(bins_arm)-1
    indexes = find(animal.pos(2,:) > bins_arm(j) & animal.pos(2,:) < bins_arm(j+1) & animal.arm & animal.speed2 > speed_bins(1) & ~Lia2 & ~isnan(trials.trials2));
    temp = histogram(animal.speed2(indexes),speed_bins,'Normalization','probability');
    speed_histogram(:,j) = temp.Values;
end
imagesc(bins_arm,speed_bins,speed_histogram), set(gca,'YDir','normal'), title('Speed on the arm'), xlabel('Position (cm)'), ylabel('Running speed (cm/s)')


subplot(3,2,4)
speed_histogram = [];
bins_rim = [animal.polar_theta_limits(1):5:animal.polar_theta_limits(2)];
for j = 1:length(bins_rim)-1
    indexes = find(animal.polar_theta > bins_rim(j) & animal.polar_theta < bins_rim(j+1) & animal.rim & animal.speed2 > speed_bins(1) & ~Lia2 & ~isnan(trials.trials2));
    temp = histogram(animal.speed2(indexes),speed_bins,'Normalization','probability');
    speed_histogram(:,j) = temp.Values;
end
imagesc(bins_rim,speed_bins,speed_histogram), set(gca,'YDir','normal'), title('Speed on the rim'), xlabel('Position (cm)'), ylabel('Running speed (cm/s)')

% Acceleration
subplot(3,2,5)
speed_histogram = [];
for j = 1:length(bins_arm)-1
    indexes = find(animal.pos(2,:) > bins_arm(j) & animal.pos(2,:) < bins_arm(j+1) & animal.arm & animal.speed2 > speed_bins(1) & ~Lia2 & ~isnan(trials.trials2));
    temp = histogram(animal.acceleration2(indexes),acceleration_bins,'Normalization','probability');
    speed_histogram(:,j) = temp.Values;
end
imagesc(bins_arm,acceleration_bins,speed_histogram), set(gca,'YDir','normal'), title('Acceleration on the arm'), xlabel('Position (cm)'), ylabel('Acceleration (cm/s)')

subplot(3,2,6)
speed_histogram = [];
for j = 1:length(bins_rim)-1
    indexes = find(animal.polar_theta > bins_rim(j) & animal.polar_theta < bins_rim(j+1) & animal.rim & animal.speed2 > speed_bins(1) & ~Lia2 & ~isnan(trials.trials2));
    temp = histogram(animal.acceleration2(indexes),acceleration_bins,'Normalization','probability');
    speed_histogram(:,j) = temp.Values;
end
imagesc(bins_rim,acceleration_bins,speed_histogram), set(gca,'YDir','normal'), title('Acceleration on the rim'), xlabel('Position (cm)'), ylabel('Acceleration (cm/s)')

indexes = find((animal.arm | animal.rim) & animal.speed2 > speed_bins(1) & ~Lia2 & ~isnan(trials.trials2));
X = [animal.speed2;animal.acceleration2];
y = animal.theta;
mdl = fitglm(X(:,indexes)',y(indexes)','interactions')

% Acceleration

%% % Plotting theta frequency vs running speed
animal.speed2 = animal.speed;% nanconv(animal.speed,gausswin(animal.sr)'/sum(gausswin(animal.sr)),'edge');
idx = isnan(animal.pos_linearized);
animal.speed2(idx) = 0;
animal.acceleration2 = nanconv(animal.acceleration,gausswin(animal.sr)'/sum(gausswin(animal.sr)),'edge');
speeed_threshold = 10;
% plot_ThetaVsSpeed(recording,animal,cooling);
% plot_ThetaVsAcceleration(recording,animal,cooling,10);
% plot_ThetaVsAcceleration(recording,animal,cooling,30);
% plot_ThetaVsSpeedAndAcceleration(recording,animal,cooling);

% Plotting gamma frequency vs running speed
recording.ch_theta = 66;
cooling2 = cooling;
cooling2.cooling(2) = cooling2.cooling(1)+200;
plot_GammaVsSpeed(recording,animal,cooling2);
% plot_GammaVsAcceleration(recording,animal,cooling,10);
plot_spectrogram(recording,animal,cooling)
% lfp = bz_GetLFP(recording.ch_theta-1);
% speed = interp1(animal.time,animal.speed2,lfp.timestamps);
% indexes = find(speed > 20);
% lfp.timestamps = lfp.timestamps(1:length(indexes));
% lfp.data = lfp.data(indexes);
% temp = bz_PowerSpectrumSlope(lfp,2,0.1,'showfig',true);
% figure, imagesc(temp.specgram'), set(gca,'Ydir','normal')

%% % Gamma vs temperature
% stats = [];
for i = [1:recording.nChannels]
    i
    recording.ch_theta = i;
    stats{i} = plot_GammaVsTemperature(recording,animal,cooling);
    figure(1000)
    stairs(stats{i}.freqlist,mean(stats{i}.freq_cooling),'b'), hold on
    stairs(stats{i}.freqlist,mean(stats{i}.freq_nocooling),'r')
    %     close all
end
save('GammaStats.mat','stats')

%%
sessionIDs = {recording.name}; %{'Peter_MS22_180628_120341_concat'};
[session, basename, basepath, clusteringpath] = db_set_path('session',sessionIDs{1});
load('GammaStats.mat','stats')
temp = [];
colorpalet = [0,0,0; 1,0,1; 1,0,0; 0.5,0.5,0.5; 0.2,0.2,0.2; 0,1,0; 0,1,1; 0,0,0; 1,0,1; 1,0,0; 0.5,0.5,0.5; 0.2,0.2,0.2; 0,1,0; 0,1,1];
xml = LoadXml(recording.name,'.xml');
% BadChannels = [session.ChannelTags.Bad.Channels,xml.SpkGrps(session.ChannelTags.Bad.SpikeGroups).Channels+1];
GoodChannels = 1:recording.nChannels; %setdiff( 1:recording.nChannels,BadChannels);
for i = GoodChannels
    temp(:,i) = mean(stats{i}.freq_cooling)./mean(stats{i}.freq_nocooling);
end

figure(1001)

for j = 1:length(xml.SpkGrps)
    subplot(2,4,j)
    for jj = 1:length(xml.SpkGrps(j).Channels+1)
        cmap = cool(length(xml.SpkGrps(j).Channels+1));
        h1{j} = plot(stats{1}.freqlist,temp(:,xml.SpkGrps(j).Channels(jj)+1),'color',cmap(jj,:)), hold on, axis tight, ylim([0.5,1.2]),grid on,
        title('Gamma (Power ratio)'), xlabel('Frequency (Hz)'), ylabel('Ratio')
    end
end
%,legend([h1{1}(1),h1{2}(1),h1{3}(1),h1{4}(1),h1{5}(1),h1{6}(1),h1{7}(1)], {'1','2','3','4','5','6','7','8','9','10','11','12'})

clear h1

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
out = plotPlaceFieldStability(spikes,trials,sr);
% out = plotPlaceFieldStability_v2(spikes,trials,sr);

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
signal = 0.000050354 * double(LoadBinary([recording.name '.lfp'],'nChannels',recording.nChannels,'channels',recording.ch_ripple,'precision','int16','frequency',recording.sr/16));
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

%% Spike alignment across neighboring trials
unitToAnalyse = 134; % 8, 134
ratemaptrials = zeros(4*126-1,trials.total);
shiftmatrix = zeros(trials.total,trials.total);
YourVector = 1:trials.total;
YourVector = YourVector(randperm(length(YourVector)));
for i = 1:trials.total
    I = YourVector(i);
    %     I = i;
    idx = intersect(find(spikes.trials{unitToAnalyse} == I), find(spikes.rim{unitToAnalyse}==1));
    temp = histcounts(spikes.polar_theta{unitToAnalyse}(idx)+126,[1:4*126]/2);
    ratemaptrials(:,i) = nanconv(temp,gausswin(7)'/sum(gausswin(7)),'edge');
end
ratemaptrials = (ratemaptrials')./max(ratemaptrials');
figure, subplot(1,2,1), imagesc(ratemaptrials)




for i = 1:trials.total
    for j = 1:trials.total
        
        
        
        [temp,lags] = xcorr(ratemaptrials(i,:),ratemaptrials(j,:));
        [~,idxx] = max(temp);
        shiftmatrix(i,j) = lags(idxx);
    end
end
subplot(1,2,2), imagesc(shiftmatrix)

%% % Place field emergence and plateau potentials
% Session 92:  39/1254 rim -27cm trial 56, unit 90/1689 rim +91cm trial 79% Unit to check for plateau potential before emorgence of a place field
% Session 126:
Plateau_unit.id = [75,103,122,137,142];
Plateau_unit.trial = [35,45,68,70,132];
Plateau_unit.pos_linearized = [150,141,184,300,106]; % Linearized postion

% Plateau_unit.pos = [-63,-62,83,89,-21]
% Plateau_unit.state = [2,2,2,2,2] % arm=1, rim = 2;
k = 5;
shank = spikes.shankID(Plateau_unit.id(k));
peakChannel = spikes.maxWaveformCh1(Plateau_unit.id(k));
spikes.shankID(Plateau_unit.id(k));
n_trials = 20;
durationToRead = 1; % in sec
neighbors = [141,find(spikes.maxWaveformCh1==peakChannel)];

neighbors = [141,find(spikes.shankID==spikes.shankID(Plateau_unit.id(k)))];
neighbors(find(neighbors==125)) = []
%
color33 = [0.3,0.3,0.3;.5,0,0];
absolutetime = [];
animal_pos_linearized2 = [];
figure, hold on;

fileID = fopen([recording.name,'.dat'],'r');
for ii = 1:n_trials*2+1
    ii
    subplot(6,1,1:5), hold on
    timepoint = animal.time(find(animal.pos_linearized>Plateau_unit.pos_linearized(k) & trials.trials2==Plateau_unit.trial(k)+(ii-(n_trials+1)),1));
    animal_pos_linearized = animal.pos_linearized(find(animal.pos_linearized>Plateau_unit.pos_linearized(k) & trials.trials2==Plateau_unit.trial(k)+(ii-(n_trials+1)),1));
    00
    neighbors_spikes = [];
    if abs(animal_pos_linearized-Plateau_unit.pos_linearized(k))<5
        temp = LoadBinaryChunk(fileID,'frequency',sr,'nChannels',nChannels,'channels',peakChannel,'duration',durationToRead,'skip',0,'start',timepoint-0.5);
        temp = (temp-nanconv(temp,ones(221,1)/221,'edge'))/5000+ii-n_trials;
        %         temp = temp+ii*5000;
        plot(1000*([1:sr]/sr-0.5),temp,'color',color33(trials.stat(Plateau_unit.trial(k)+(ii-(n_trials+1))),:)), hold on
        spiketimes = spikes.times{Plateau_unit.id(k)}(find(spikes.trials{Plateau_unit.id(k)}==Plateau_unit.trial(k)+(ii-(n_trials+1))))-timepoint;
        
        for j = 1:length(neighbors)
            spiketimes_temp = spikes.times{neighbors(j)}(find(spikes.trials{neighbors(j)}==Plateau_unit.trial(k)+(ii-(n_trials+1))))-timepoint;
            neighbors_spikes = [neighbors_spikes;spiketimes_temp];
        end
        animal_pos_linearized2(ii) = animal_pos_linearized;
        
        if ~isempty(neighbors_spikes)
            neighbors_spikes = neighbors_spikes(find(neighbors_spikes<0.5 & neighbors_spikes >-0.5));
            if ~isempty(neighbors_spikes)
                absolutetime(ii) = neighbors_spikes(1)+timepoint;
                neighbors_spikes2 = round(sr*neighbors_spikes+sr/2);
                plot(1000*neighbors_spikes,temp(neighbors_spikes2),'ok');hold on,%,'color',color33(trials.stat(Plateau_unit.trial(k)+(ii-(n_trials+1))),:))
                subplot(6,1,6)
                plot((1000*neighbors_spikes*[1,1])',(ones(length(neighbors_spikes2),1)*[1,0]+ii-n_trials)','-k'), hold on
            end
        end
        
        if ~isempty(spiketimes)
            spiketimes = spiketimes(find(spiketimes<0.5 & spiketimes >-0.5));
            if ~isempty(spiketimes)
                absolutetime(ii) = spiketimes(1)+timepoint;
                spiketimes2 = round(sr*spiketimes+sr/2);
                subplot(6,1,1:5)
                plot(1000*spiketimes,temp(spiketimes2),'or'), hold on;%,'color',color33(trials.stat(Plateau_unit.trial(k)+(ii-(n_trials+1))),:))
                subplot(6,1,6)
                plot((1000*spiketimes*[1,1])',(ones(length(spiketimes2),1)*[1,0]+ii-n_trials)','-r'), hold on
            end
        end
        
    end
    
    
end
xlim([-500,500]), ylim([-n_trials,n_trials]), xlabel('Time (ms)'), ylabel('Trials')
fclose(fileID)
subplot(6,1,1:5)
title(['k=', num2str(k),', cell=' num2str(Plateau_unit.id(k))])


animal_pos_linearized2

%% Geisler analysis
load([recording.name, '.CoolingStates.firingRateMap.mat'],'CoolingStates')
firingRateMap_CoolingStates = CoolingStates;
% load('firingRateMap_CoolingStates.mat')

cell_metrics = LoadCellMetricBatch('sessions',{recording.name});
PyramidalIndexes = find(contains(cell_metrics.PutativeCellType,'Pyramidal Cell'));
conditions = {'Pre','Cooling','Post'};
colors = {'g','b','r'};

placefield_difference_all ={};
ccg_delay_all = {};
placefield_speed_all = {};

for iii = 1:3
    temp = firingRateMap_CoolingStates.unit(:,PyramidalIndexes,iii);
    x_bins = firingRateMap_CoolingStates.x_bins;
    boundaries = firingRateMap_CoolingStates.boundaries;
    temp(isnan(temp)) = 0;
    SpatialCoherence = [];
    condition = [];
    placefield_count = [];
    placefield_interval = [];
    placefield_state = [];
    spikes3 = [];
    times = [];
    groups = [];
    placefield_peak = [];
    placefield_speed = [];
    placefield_difference = [];
    placefield_speed_av = [];
    
    kk = 1;
    kk2 = 0;
    for i = 1:size(temp,2)
        temp2 = place_cell_condition(temp(:,i)');
        SpatialCoherence(i) = temp2.SpatialCoherence;
        condition(i) = temp2.condition;
        placefield_count(i) = temp2.placefield_count;
        placefield_interval{i} = temp2.placefield_interval;
        placefield_state(:,i) = temp2.placefield_state;
        
        %         figure(60+iii+(kk2-1)*3)
        %         subplot(4,4,kk)
        %         plot(x_bins,temp(:,i),'.-k'), hold on, plot(x_bins(find(placefield_state(:,i))),temp(find(placefield_state(:,i)),i),'or'),
        %         title([num2str(i), ' (UID ', num2str(PyramidalIndexes(i)),') ', num2str(iii), ])
        
        kk = kk + 1;
        if kk > 16
            kk = 1;
            kk2 = kk2+1;
        end
    end
    
    k = 1;
    place_cells_arm = find(sum(placefield_state(find(x_bins<boundaries(2)+150),:)));
    
    figure(50)
    subplot(3,3,1+iii-1), hold on
    for i = 1:length(place_cells_arm)
        fields = find(sum(placefield_interval{place_cells_arm(i)} < (boundaries(2)+150)/3,2));
        colors2 = rand(1,3);
        for j = 1:length(fields)
            pf_interval = x_bins(placefield_interval{place_cells_arm(i)}(fields(j),:));
            cell_id = PyramidalIndexes(place_cells_arm(i));
            spikes_infield = find(spikes.pos_linearized{cell_id} > pf_interval(1) - 2 & spikes.pos_linearized{PyramidalIndexes(place_cells_arm(i))} < pf_interval(2) + 2 & spikes.state{PyramidalIndexes(place_cells_arm(i))} == iii);
            spikes3.times{k} = spikes.times{cell_id}(spikes_infield);
            spikes3.theta_phase2{k} = spikes.theta_phase2{cell_id}(spikes_infield);
            spikes3.UID(k) = cell_id;
            times = [times; spikes3.times{k}];
            groups = [groups; k * ones(length(spikes3.times{k}),1)];
            
            plot(spikes.pos_linearized{cell_id}(spikes_infield),i * ones(1,length(spikes_infield)),'.','color',colors2), hold on,
            
            placefield_peak(k) = mean(spikes.pos_linearized{cell_id}(spikes_infield));
            placefield_speed(k) = nanmean(spikes.speed{cell_id}(spikes_infield));
            k = k + 1;
        end
    end
    
    [times,I] = sort(times);
    groups = groups(I);
    xlabel('Position (cm)'), ylabel('Placefield'), title(['Placefields - ', conditions{iii} ]), xlim([0,345]), axis tight, gridxy(boundaries)
    
    
    [ccg,t] = CCG(times,groups,'binSize',0.001,'duration',0.4);
    t = t * 1000;
    ccg_count = [];
    ccg_delay = [];
    locs = [];
    pks = [];
    kk = 1;
    before = [];
    t0 = find(t == 0);
    
    
    for i = 1:size(ccg,3)-1
        %         figure
        %         kkk = 1;
        for j = i+1:size(ccg,3)
            placefield_difference(kk) = placefield_peak(j) - placefield_peak(i);
            placefield_speed_av(kk) = nanmean([placefield_speed(j),placefield_speed(i)]);
            ccg_trace = nanconv(ccg(:,i,j)',gausswin(80)','edge');
            [ccg_count(kk),ccg_delay(kk)] = max(ccg_trace);
            locs(kk) = 0;
            pks(kk) = 0;
            before(kk) = 0;
            
            if ccg_count(kk)
                [pks_temp,locs_temp] =  findpeaks(ccg_trace);
                if ~isempty(pks_temp)
                    if ccg_delay(kk)<t0
                        temp333 = find(locs_temp<t0);
                        if ~isempty(temp333)
                            locs(kk) = locs_temp(temp333(end));
                            pks(kk) = pks_temp(temp333(end));
                            before(kk) = 1;
                        end
                    else
                        temp333 = find(locs_temp>t0,1);
                        if ~isempty(temp333)
                            locs(kk) = locs_temp(temp333);
                            pks(kk) = pks_temp(temp333);
                            before(kk) = 2;
                        end
                    end
                end
            end
            
            %             if ccg_count(kk)>0
            %                 subplot(8,8,kkk), plot(t,ccg_trace), hold on,
            %                 if length(locs)==kk & locs(kk)>0
            %                     plot(t(locs(kk)),pks(kk),'or')
            %                 end
            %                 kkk = kkk +1;
            %             end
            
            kk = kk + 1;
            
        end
    end
    
    indx = find(ccg_count==0 | pks == 0 | abs(placefield_difference)>80 | locs<50  | locs>350 ); %
    placefield_difference(indx) = [];
    placefield_speed_av(indx) = [];
    ccg_delay(indx) = [];
    ccg_count(indx) = [];
    locs(indx) = [];
    pks(indx) = [];
    ccg_delay = t(locs)';
    
    figure(50)
    subplot(3,3,4+iii-1)
    plot(placefield_difference,ccg_delay,'o'), hold on
    x = placefield_difference; y1 = ccg_delay;
    subset = find(y1 < 95 & y1 > -95);
    x = x(subset); y1 = y1(subset);
    
    P = polyfit(x,y1,1); yfit = P(1)*x+P(2); plot(x,yfit,[colors{iii},'-']);
    text(-75,-150,['Slope: ' num2str(P(1),3)],'Color','k')
    [R,P] = corrcoef(x,y1);
    text(-75,100,[conditions{iii},': R = ' num2str(R(2,1),3),', P = ', num2str(P(2,1),3)],'Color','k')
    xlabel('Distance (cm)'), ylabel('Time lag (s)'), title(['Compression - state ' num2str(iii)]), grid on, hold on, xlim([-80,80]), ylim([-110,110])
    %     [param]=sigm_fit(x,y1);
    
    subplot(3,3,7+iii-1)
    plot(placefield_difference./placefield_speed_av,ccg_delay,'o'), xlabel('L - [Distance] (s)'), ylabel('Time lag (s)'), xlim([-0.8,0.8]), grid on
    placefield_difference_all{iii} = placefield_difference;
    ccg_delay_all{iii} = ccg_delay;
    placefield_speed_all{iii} = placefield_speed_av;
end

save('PlaceFields.mat','placefield_difference_all','ccg_delay_all','placefield_speed_all')

%% % Interneuron theta frequency rate dependency analysis
load(fullfile(recording.SpikeSorting.path,'cell_metrics.mat'))
% cell_metrics = LoadCellMetricBatch('sessions',{recording.name});
InterneuronIndexes = find(contains(cell_metrics.putativeCellType,'Interneuron'));
slope1 = [];
figure,
for i = 1:length(InterneuronIndexes)
    subplot(5,6,i)
    spkdiff = nanconv(1./diff(spikes.times{InterneuronIndexes(i)}),gausswin(50)/sum(gausswin(50)),'edge');
    %     spkdiff = 1./diff(spikes.times{InterneuronIndexes(i)});
    indx = find(spikes.speed{InterneuronIndexes(i)} >5 & spikes.pos_linearized{InterneuronIndexes(i)}>0);
    x = spikes.theta_freq{InterneuronIndexes(i)}(indx);
    y1 = spkdiff(indx);
    [R,P] = corrcoef(x,y1);
    P1 = polyfit(x,y1,1); yfit = P1(1)*x+P1(2);
    if P(2,1)<0.05
        color = '.b';
        slope1(i) = P1(1);
    else
        color = '.k';
        slope1(i) = nan;
    end
    plot(x, y1,color), grid on, axis tight, hold on
    plot(x,yfit,'-k');
    title(cell_metrics.putativeCellType{InterneuronIndexes(i)})
    xlabel(['R,P: ' num2str(R(2,1),3),',', num2str(P(2,1),3)])
    ylabel([num2str(P1(1),3),' Hz/degree']), ylim([0,min([200,max(y1)])]),
end

figure,
histogram(slope1,[-4:0.2:4]), xlabel('Slope (Hz/degree)')

figure,
for i = 1:length(InterneuronIndexes)
    subplot(5,6,i)
    indx = find(spikes.speed{InterneuronIndexes(i)} >2 & spikes.pos_linearized{InterneuronIndexes(i)}>0);
    plot(spikes.temperature{InterneuronIndexes(i)}(indx), spikes.theta_phase{InterneuronIndexes(i)}(indx),'.r')
end
units = {};
trials_temperature = nan(length(InterneuronIndexes),171);
trials_theta_freq = nan(length(InterneuronIndexes),171);
trials_rate = nan(length(InterneuronIndexes),171);

for i = 1:length(InterneuronIndexes)
    
    tri = [];
    for k = 1:trials.total
        indexes = find(spikes.speed{InterneuronIndexes(i)} > 5 & spikes.trials{InterneuronIndexes(i)} == k);
        tri.temperature(k) = mean(spikes.temperature{InterneuronIndexes(i)}(indexes));
%         tri.speed(k) = mean(spikes.speed{InterneuronIndexes(i)}(indexes));
        tri.theta_freq(k) = mean(spikes.theta_freq{InterneuronIndexes(i)}(indexes));
        tri.rate(k) = length(indexes)/(sum(trials.trials2==k & animal.speed>5)/animal.sr);
    end
    trials_temperature(i,1:length(tri.rate)) = tri.temperature;
    trials_theta_freq(i,1:length(tri.rate)) = tri.theta_freq;
    trials_rate(i,1:length(tri.rate)) = tri.rate;
    figure,
    subplot(3,1,1), plot(tri.temperature), xlabel('Trial'),ylabel('Temperature'), title(['Unit ' num2str(InterneuronIndexes(i))]), grid on, axis tight
    subplot(3,1,2), plot(tri.theta_freq), xlabel('Trial'),ylabel('Theta frequency (Hz)'), grid on, axis tight
    subplot(3,1,3), plot(tri.rate), xlabel('Trial'),ylabel('Firing rate'), grid on, axis tight
    units{i} = tri;
end
figure
subplot(3,1,1), 
plot(trials_temperature',trials_rate','.'), title('Temperature')
subplot(3,1,2), 
imagesc(trials_theta_freq), title('Theta freq')
subplot(3,1,3)
imagesc(trials_rate),xlabel('Trials'),ylabel('Interneurons'), title('Firing rate')

%% % Gamma band analysis
% for i = 1:128
channels = recording.ch_theta-1;
% channels = i
phaserange = [5:0.2:12];
amprange = [5:0.2:60];
amprange = [50:1:150];
lfp = bz_GetLFP(channels,'basepath',basepath);
speed = interp1(animal.time,animal.speed2,lfp.timestamps);
temperature = interp1(animal.time,animal.temperature,lfp.timestamps);
position = interp1(animal.time,animal.pos_linearized,lfp.timestamps);

indexes = find(speed > 10 & position>0 & lfp.timestamps>cooling.nocooling(1,1)' & lfp.timestamps<cooling.nocooling(2,1)');
lfp1 = lfp; lfp1.data = lfp1.data(indexes); lfp1.timestamps = lfp1.timestamps(1:length(indexes));
[comod1] = bz_ModIndex(lfp1,phaserange,amprange,0);

fig = figure
subplot(1,3,1)
imagesc(phaserange(1:end-1),amprange(1:end-1),comod1)
colormap jet, hold on, xlabel('Frequency phase'); ylabel('Frequency amplitude'), axis xy
title('No Cooling'),
drawnow

indexes = find(speed > 10 & position>0 & lfp.timestamps>cooling.cooling(1,1)' & lfp.timestamps<cooling.cooling(2,1)');
lfp1 = lfp; lfp1.data = lfp1.data(indexes); lfp1.timestamps = lfp1.timestamps(1:length(indexes));
[comod2] = bz_ModIndex(lfp1,phaserange,amprange,0);
subplot(1,3,2)
imagesc(phaserange(1:end-1),amprange(1:end-1),comod2)
colormap jet, hold on, xlabel('Frequency phase'); ylabel('Frequency amplitude'), axis xy
title('Cooling')
drawnow

indexes = find(speed > 10 & position>0 & lfp.timestamps>cooling.nocooling(1,2)' & lfp.timestamps<cooling.nocooling(2,2)');
lfp1 = lfp; lfp1.data = lfp1.data(indexes); lfp1.timestamps = lfp1.timestamps(1:length(indexes));
[comod3] = bz_ModIndex(lfp1,phaserange,amprange,0);
subplot(1,3,3)
imagesc(phaserange(1:end-1),amprange(1:end-1),comod3)
colormap jet, hold on, xlabel('Frequency phase'); ylabel('Frequency amplitude'), axis xy
title('No Cooling')
drawnow
saveas(fig,[recording.name,'.theta_gamma_phasecoupling_',num2str(i),'.png'])
drawnow
cmin = min(min([comod1,comod2,comod3]));
cmax = max(max([comod1,comod2,comod3]));
subplot(1,3,1), clim([cmin,cmax]), hold on,  %plot(phaserange,phaserange,'w'),  plot(phaserange,2*phaserange,'w'),  plot(phaserange,3*phaserange,'w'),  plot(phaserange,4*phaserange,'w')
subplot(1,3,2), clim([cmin,cmax]), hold on,  %plot(phaserange,phaserange,'w'),  plot(phaserange,2*phaserange,'w'),  plot(phaserange,3*phaserange,'w'),  plot(phaserange,4*phaserange,'w')
subplot(1,3,3), clim([cmin,cmax]), hold on,  %plot(phaserange,phaserange,'w'),  plot(phaserange,2*phaserange,'w'),  plot(phaserange,3*phaserange,'w'),  plot(phaserange,4*phaserange,'w')

% close(fig)
% end

%% Continues
channels = recording.ch_theta-1;

lfp = bz_GetLFP(channels,'basepath',basepath);
speed = interp1(animal.time,animal.speed2,lfp.timestamps);
temperature = interp1(animal.time,animal.temperature,lfp.timestamps);
position = interp1(animal.time,animal.pos_linearized,lfp.timestamps);
phaserange = [6.5:0.2:10];
amprange = [14:0.5:60];
indexes = find(speed > 10 & position>0);
comod12 = [];
tempstruct = [];
for i = 1:(length(indexes)/(5*1250))-1
    i
    idx1 = indexes(i*(5*1250):(i+1)*(5*1250));
    lfp1 = lfp; lfp1.data = lfp1.data(idx1); lfp1.timestamps = lfp1.timestamps(1:length(idx1));
    [comod1] = bz_ModIndex(lfp1,phaserange,amprange,0);
    comod12(:,i) = mean(comod1');
    tempstruct.temperature(i) = mean(temperature(idx1));
    tempstruct.speed(i) = mean(speed(idx1));
end
figure,
subplot(4,1,1:3)
imagesc(1:size(comod12,2),amprange(1:end-1),comod12)
colormap jet, hold on, xlabel('Time'); ylabel('Frequency'), axis xy, hold on
title('No Cooling'),
drawnow
% subplot(3,1,2)
plot(1:size(comod12,2),tempstruct.temperature,'w'), title('temperature')
subplot(4,1,4)
plot(1:size(comod12,2),tempstruct.speed), title('speed')


%% Artificial theta and saw tooth assymetry
phaserange = [5:0.2:14];
amprange = [5:0.2:200];
sr = 1250;
time = [1:10000]/sr;
freq_base = 9;
theta = sin(2*pi*time*freq_base);
harmonics = [0:10];
harmonics_amplitudes = 1./[1:length(harmonics)];
figure
for i = 1:length(harmonics)
    i
    theta = theta + harmonics_amplitudes(i)*sin(2*pi*time*freq_base*(harmonics(i)+1));
    subplot(2,length(harmonics),i)
    plot(time,theta), xlim([0,0.5]), title(['Harmonics: ' num2str(harmonics(i))])
    subplot(2,length(harmonics),length(harmonics)+i)
    lfp1.data = theta'; lfp1.timestamps = time'; lfp1.samplingRate = sr;
    [comod1] = bz_ModIndex(lfp1,phaserange,amprange,0);
    imagesc(phaserange(1:end-1),amprange(1:end-1),comod1)
    colormap jet, hold on, xlabel('Frequency phase'); ylabel('Frequency amplitude'), axis xy
end

%% % Oscillation freq of Pyramidal and interneuron population
close all, clear all
sessionNames = {'Peter_MS12_170714_122034_concat','Peter_MS12_170715_111545_concat','Peter_MS12_170716_172307_concat','Peter_MS12_170717_111614_concat','Peter_MS12_170719_095305_concat'...
    'Peter_MS12_170717_111614_concat','Peter_MS13_171129_105507_concat','Peter_MS13_171130_121758_concat','Peter_MS13_171128_113924_concat','Peter_MS13_171201_130527_concat',...
    'Peter_MS21_180629_110332_concat','Peter_MS21_180627_143449_concat','Peter_MS21_180625_153927_concat','Peter_MS21_180712_103200_concat','Peter_MS21_180628_155921_concat',...
    'Peter_MS22_180628_120341_concat','Peter_MS22_180629_110319_concat','Peter_MS22_180720_110055_concat','Peter_MS22_180711_112912_concat'};
% sessionNames = {'Peter_MS12_170714_122034_concat','Peter_MS12_170715_111545_concat'};
batchName = 'All';
colors = {'r','b','g'};
colors1 = {'r','b','g','m','c','k'};
celltypes = {'Interneuron','Pyramidal'};
states = {'Pre','Cooling','Post'};
speed_threshold = 10;
bin_size = 1/100;
phase_intervals = -pi:pi/20:3*pi;
pos_intervals = [0:5:350];
spikes_all_speed = {};
spike_all_times = {};
spike_all_phase = {};
spike_all_pos_linearized = {};
boundaries = [85, 130+85];
for k = 1:length(sessionNames)
    disp(['*** Processing sessions: ', num2str(k),'/', num2str(length(sessionNames)),' sessions: ' sessionNames{k}])
    [session, basename, basepath, clusteringpath] = db_set_path('session',sessionNames{k});
    spikes = loadSpikes('clusteringpath',clusteringpath,'clusteringformat','phy','basename',basename);
    load([basename,'.trials.behavior.mat'])
    load([basename,'.animal.behavior.mat'])
    for i = 1:3
        for j = 1:2
            [cell_metrics_idxs] =  get_CellMetrics('session',basename,'putativeCellType',{celltypes{j}});
            [spike_times,idx] = sort(vertcat(spikes.times{cell_metrics_idxs}));
            spikecount = sum(spikes.total(cell_metrics_idxs));
            spike_speed = vertcat(spikes.speed{cell_metrics_idxs}); spike_speed = spike_speed(idx);
            spike_state = vertcat(spikes.state{cell_metrics_idxs}); spike_state = spike_state(idx);
            spike_trials = vertcat(spikes.trials{cell_metrics_idxs}); spike_trials = spike_trials(idx);
            spike_phase = horzcat(spikes.theta_phase{cell_metrics_idxs}); spike_phase = spike_phase(idx);
            spike_pos_linearized = vertcat(spikes.pos_linearized{cell_metrics_idxs}); spike_pos_linearized = spike_pos_linearized(idx);
            idx2 = find(spike_speed>speed_threshold & spike_state == i);
            idx3 = find(spike_trials ~= trials.error); idx2 = intersect(idx2,idx3);
            spike_speed = spike_speed(idx2);
            spike_times = spike_times(idx2); spike_times = spike_times-min(spike_times)+0.1;
            spike_phase = spike_phase(idx2);
            spike_pos_linearized = spike_pos_linearized(idx2);
            
            spikes_all_speed{k,i,j} = spike_speed;
            spike_all_times{k,i,j} = spike_times;
            spike_all_phase{k,i,j} = spike_phase;
            spike_all_pos_linearized{k,i,j} = spike_pos_linearized;
            
%             spike_rate = [bin_size:bin_size:ceil(spike_times(end))];
%             spike_rate2 = histcounts(spike_times,spike_rate);
%             spike_rate3 = nanconv(spike_rate2,gausswin(10)'/10,'edge');
            
        end
    end
end
%%
for i = 1:3
    for j = 1:2
        spike_speed = spikes_all_speed{1,i,j};
        spike_times = spike_all_times{1,i,j}+spike_times(end)+1;
        spike_phase = spike_all_phase{1,i,j};
        spike_pos_linearized = spike_all_pos_linearized{1,i,j};
        
        for k = 2:length(sessionNames)
            spike_speed = [spike_speed;spikes_all_speed{k,i,j}];
            spike_times = [spike_times;spike_all_times{k,i,j}+spike_times(end)+1];
            spike_phase = [spike_phase,spike_all_phase{k,i,j}];
            spike_pos_linearized = [spike_pos_linearized;spike_all_pos_linearized{k,i,j}];
        end
%         % Animal position histogram to normalize the spiking data
%         L = length(spike_rate3);
%         Fs = 100;
%         NFFT = 2^nextpow2(L); % Next power of 2 from length of y
%         Y = fft(spike_rate3,NFFT)/L;
%         f = Fs/2*linspace(0,1,NFFT/2+1);
        
        [ccg1,t] = CCG(spike_times,ones(size(spike_times)),'binSize',0.004,'duration',1);
        figure(100)
%         subplot(2,1,j), 
        plot(t,ccg1/max(ccg1),colors1{i+(j-1)*3}), hold on
        title('Pyramidal cells and interneurons'), ylabel('ACG'),xlabel('Time'), legend([states,states]), grid on, xlim([-0.3,0.3])
        
        figure(i+100)
        subplot(2,2,j*2-1)
        plot(spike_pos_linearized,spike_phase,['.',colors{j}]), hold on, plot(spike_pos_linearized,2*pi+spike_phase,['.',colors{j}])
        title(celltypes{j}), hold on
        plot([boundaries;boundaries],[-pi,3*pi;-pi,3*pi]','k');
        
        [slope,offset,R_value] = CircularLinearRegression(spike_phase(~isnan(spike_pos_linearized)),spike_pos_linearized(~isnan(spike_pos_linearized)),1)
        x = [1:0.01:350];
        plot(x,wrapTo2Pi(2*pi*slope*x) + offset,'m.','linewidth',1.5)
        xlabel(['Slope: ' num2str(slope),',  R-value: ', num2str(R_value)])
        
        subplot(2,2,j*2) % 2D histogram phase vs position
        [N,Xedges,Yedges] = histcounts2([spike_pos_linearized;spike_pos_linearized],[spike_phase';2*pi+spike_phase'],pos_intervals,phase_intervals);
        imagesc(Xedges,Yedges,N'); set(gca, 'YDir','normal'), hold on
        plot([boundaries;boundaries],[-pi,3*pi;-pi,3*pi]','k');
%         figure(100)
%         subplot(2,1,j*2)
%         % Plot single-sided amplitude spectrum.
%         plot(f,nanconv(2*abs(Y(1:NFFT/2+1)),gausswin(100)'/100,'edge'),colors{i}), hold on, xlim([0,12])
%         title('Single-Sided Amplitude Spectrum of y(t)')
%         xlabel('Frequency (Hz)')
%         ylabel('|Y(f)|')
    end
end

figure(100)
set(gcf,'position',[50,500,1300,800])
saveas(gcf,['C:\Users\peter\Dropbox\Buzsakilab Postdoc\Medial Septum Cooling Project\Temporal Compression\',batchName,'_ACG_AcrossCells (pyramidal and interneurons).pdf'])
figure(101)
set(gcf,'position',[50,500,1300,800])
saveas(gcf,['C:\Users\peter\Dropbox\Buzsakilab Postdoc\Medial Septum Cooling Project\Temporal Compression\',batchName,'_PhaseVsPosition_AllSpikes_Pre.png'])
figure(102)
set(gcf,'position',[50,500,1300,800])
saveas(gcf,['C:\Users\peter\Dropbox\Buzsakilab Postdoc\Medial Septum Cooling Project\Temporal Compression\',batchName,'_PhaseVsPosition_AllSpikes_Cooling.png'])
figure(103)
set(gcf,'position',[50,500,1300,800])
saveas(gcf,['C:\Users\peter\Dropbox\Buzsakilab Postdoc\Medial Septum Cooling Project\Temporal Compression\',batchName,'_PhaseVsPosition_AllSpikes_Post.png'])

%% % Analysing relationship between single unit firing rates and brain temperature
clear all, close all
MedialSeptum_Recordings
id = 93;
recording = recordings(id);
[session, basename, basepath, clusteringpath] = db_set_path('session',recording.name);

% Spikes
spikes = loadSpikes('session',session);

load([recording.name, '.temperature.timeseries.mat'])
load([recording.name, '.cooling.manipulation.mat'])
% load([recording.name, '.animal.behavior.mat'])
% load([recording.name, '.trials.behavior.mat'])

for i = 1:spikes.numcells
    spikes.temperature{i} = interp1(temperature.time,temperature.temp,spikes.times{i},'linear');
    spikes.instantaneousRate{i} = InstantaneousRate(spikes.times{i});
end

%%
close all
k = 1;
R1 = [];
P1 = [];
t_binsize = 10;
t_bins = [0:t_binsize:max(spikes.spindices(:,1))];
temperature_binnned = zeros(1,length(t_bins)-1);
recording_length = max(spikes.spindices(:,1));
for j = 1:length(t_bins)-1
    idx3 = find(temperature.time>t_bins(j) & temperature.time<t_bins(j+1));
    temperature_binnned(j) = mean(temperature.temp(idx3));
end

figure
for i = 1:spikes.numcells
    rate_binned = zeros(1,length(t_bins)-1);
    idx = find(spikes.temperature{i}>34);

    subplot(3,5,k)
    plot(spikes.times{i},spikes.instantaneousRate{i},'.'), axis tight, hold on
    plot(temperature.time,(temperature.temp-34)*spikes.total(i)/recording_length*5,'-k'), axis tight, ylim([0,10])
    title(['Unit ', num2str(i),', ',num2str(spikes.total(i)/recording_length,2),'Hz'])
    subplot(3,5,5+k)
    for j = 1:length(t_bins)-1
        idx2 = find(spikes.times{i}>t_bins(j) & spikes.times{i}<t_bins(j+1));
        rate_binned(j) = length(idx2)/t_binsize;
    end
    plot(t_bins(1:end-1)+t_binsize/2,rate_binned,'-b'), hold on
    plot(t_bins(1:end-1)+t_binsize/2,(temperature_binnned-34)*2,'-k'), axis tight, ylim([0,10])
    
    subplot(3,5,10+k)
    x = spikes.temperature{i}(idx);
    y1 = spikes.instantaneousRate{i}(idx);
    plot(x,y1,'.'), axis tight, hold on
    [R,P] = corrcoef(x,y1);
    R1 = [R1,R(2,1)];
    P1 = [P1,P(2,1)]; 
    P_fit = polyfit(x,y1,1); yfit = P_fit(1)*x+P_fit(2); plot(x,yfit,'-k');
    title(['R=',num2str(R(2,1)),',P=',num2str(P(2,1))])
    % xlabel('Time'), ylabel('Firing rate (Hz) and temperature')
    if k == 5
        figure
        k = 1;
    else
        k = k + 1; 
    end
end

figure,
histogram(R1,[-1:0.1:1]), hold on
histogram(R1(P1>0.05),[-1:0.1:1])


%% % Splitter cells affected by cooling
close all, clear all
sessionNames = {'Peter_MS12_170714_122034_concat','Peter_MS12_170715_111545_concat','Peter_MS12_170716_172307_concat','Peter_MS12_170717_111614_concat','Peter_MS12_170719_095305_concat'...
    'Peter_MS12_170717_111614_concat','Peter_MS13_171129_105507_concat','Peter_MS13_171130_121758_concat','Peter_MS13_171128_113924_concat','Peter_MS13_171201_130527_concat',...
    'Peter_MS21_180629_110332_concat','Peter_MS21_180627_143449_concat','Peter_MS21_180625_153927_concat','Peter_MS21_180712_103200_concat','Peter_MS21_180628_155921_concat',...
    'Peter_MS22_180628_120341_concat','Peter_MS22_180629_110319_concat','Peter_MS22_180720_110055_concat','Peter_MS22_180711_112912_concat'};
% sessionNames = {'Peter_MS22_180629_110319_concat','Peter_MS12_170715_111545_concat'};
batchName = 'All';
colors = {'r','b','g'};
colors1 = {'r','b','g','m','c','k'};
celltypes = {'Interneuron','Pyramidal'};
states = {'Pre','Cooling','Post'};
speed_threshold = 10;
bin_size = 1/100;
phase_intervals = -pi:pi/20:3*pi;
pos_intervals = [0:5:350];
spikes_all_speed = {};
spike_all_times = {};
spike_all_phase = {};
spike_all_pos_linearized = {};
boundaries = [85, 130+85];
spatialSplitterDegree = [];
maze_bins = [0:5:boundaries(1)];
kk = 0;

for k = 1:length(sessionNames)
    disp(['*** Processing sessions: ', num2str(k),'/', num2str(length(sessionNames)),' sessions: ' sessionNames{k}])
    [session, basename, basepath, clusteringpath] = db_set_session('sessionName',sessionNames{k});
    spikes = loadSpikes('session',session);
    trials = loadStruct('trials','behavior','session',session);
    animal = loadStruct('animal','behavior','session',session);
    % [cell_metrics_idxs] =  get_CellMetrics('session',basename,'putativeCellType',{celltypes{j}});
    
    for i = 1:3
        disp(['Processing state: ' num2str(i)])
        if length(trials.cooling) ~= length(trials.stat)
            trials.cooling = [trials.cooling,zeros(1,length(trials.stat) - length(trials.cooling))];
        end
        state_trials_left = find(trials.stat == 1 & trials.cooling == i);
        idx = ismember(trials.trials2,state_trials_left) & animal.speed > 10;
        maze_animal_count_left = histcounts(animal.pos_linearized(idx),maze_bins);
        
        state_trials_right = find(trials.stat == 2 & trials.cooling == i);
        idx = ismember(trials.trials2,state_trials_right);
        maze_animal_count_right = histcounts(animal.pos_linearized(idx),maze_bins);
        
        for iCells = 1:spikes.numcells
            idx = ismember(spikes.trials{iCells},state_trials_left) & spikes.speed{iCells} > 10;
            if any(idx) && sum(idx) > 200
                maze_spike_count_left = histcounts(spikes.pos_linearized{iCells}(idx),maze_bins);
                ratemap_left = maze_spike_count_left./ maze_animal_count_left;
            else
                ratemap_left = zeros(1,length(maze_bins)-1);
            end

            idx = ismember(spikes.trials{iCells},state_trials_right) & spikes.speed{iCells} > 10;
            if any(idx) && sum(idx) > 200
                maze_spike_count_right = histcounts(spikes.pos_linearized{iCells}(idx),maze_bins);
                ratemap_right = maze_spike_count_right./ maze_animal_count_right;
            else
                ratemap_right = zeros(1,length(maze_bins)-1);
            end
            spatialSplitterDegree(iCells+kk,i) = calc_spatialSplitterDegree(ratemap_left,ratemap_right); 
        end
    end
    kk = spikes.numcells + kk;
end

spatialSplitterDegree2 = spatialSplitterDegree(all(~isnan(spatialSplitterDegree)') & ~any(spatialSplitterDegree'==1),:);
figure, 
subplot(1,3,1)
plot(spatialSplitterDegree2','-')
[p1,h1] = signrank(spatialSplitterDegree2(:,1),spatialSplitterDegree2(:,2));
[p2,h2] = signrank(spatialSplitterDegree2(:,1),spatialSplitterDegree2(:,3));
[p3,h3] = signrank(spatialSplitterDegree2(:,2),spatialSplitterDegree2(:,3));
text(1.1,1.1,[' (1,2) ',num2str(p1),',  ',num2str(h1)]);
text(1.1,1.15,[' (1,3) ',num2str(p2),'  ,',num2str(h2)]);
text(1.1,1.20,[' (2,3) ',num2str(p3),',  ',num2str(h3)]); 
ylim([0,1.25])

subplot(1,3,2)
plot(spatialSplitterDegree2(:,1)./spatialSplitterDegree2(:,2),spatialSplitterDegree2(:,3)./spatialSplitterDegree2(:,2),'o')
hold on
[R,P,P1,RL,RU] = plotWithFit2((spatialSplitterDegree2(:,1)./spatialSplitterDegree2(:,2))',(spatialSplitterDegree2(:,3)./spatialSplitterDegree2(:,2))','k',1);
xlim([0,3]),ylim([0,3])

subplot(1,3,3)
plot(spatialSplitterDegree2(:,1),spatialSplitterDegree2(:,2),'.b'), hold on
plot(spatialSplitterDegree2(:,3),spatialSplitterDegree2(:,2),'.r')

%% % Behavioral preference
close all, clear all
sessionNames = {'Peter_MS12_170714_122034_concat','Peter_MS12_170715_111545_concat','Peter_MS12_170716_172307_concat','Peter_MS12_170717_111614_concat','Peter_MS12_170719_095305_concat'...
    'Peter_MS12_170717_111614_concat','Peter_MS13_171129_105507_concat','Peter_MS13_171130_121758_concat','Peter_MS13_171128_113924_concat','Peter_MS13_171201_130527_concat',...
    'Peter_MS21_180629_110332_concat','Peter_MS21_180627_143449_concat','Peter_MS21_180625_153927_concat','Peter_MS21_180712_103200_concat','Peter_MS21_180628_155921_concat',...
    'Peter_MS22_180628_120341_concat','Peter_MS22_180629_110319_concat','Peter_MS22_180720_110055_concat','Peter_MS22_180711_112912_concat'};
figure
clr = {'r','b','k'};
for k = 1:length(sessionNames)
    disp(['*** Processing sessions: ', num2str(k),'/', num2str(length(sessionNames)),' sessions: ' sessionNames{k}])
    [session, basename, basepath, clusteringpath] = db_set_session('sessionName',sessionNames{k});
%     spikes = loadSpikes('session',session);
    trials = loadStruct('trials','behavior','session',session);
%     animal = loadStruct('animal','behavior','session',session);
    temp3=[];
    for i = 1:3
        temp2 = find(trials.cooling==i);
%         temp2(ismember(temp2,trials.error)) = []; % Removing error trials
        temp = histcounts(trials.stat(temp2),[1,2,3]);
        temp3(i) = temp(1)/length(temp2);
        drawnow
    end
    plot([1,2,3],temp3,'o-k'), hold on
end
