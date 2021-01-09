function calc_phaseprecessionslopes(id)
% Medial Septum Circular Track
% clear all, close all
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
% id = 126 % 173, 169
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

[session, basename, basepath, clusteringpath] = db_set_path('session',recording.name);
if isempty(session.epochs.duration) | session.epochs.duration == 0
    disp('Updating DB')
%     session = db_update_session(session);
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
    %     spikes.theta_freq{i} = interp1([1:length(theta.freq)]/theta.sr_freq,theta.freq,spikes.times{i});
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
    spikes.temperature{i} = interp1(animal.time,animal.temperature,spikes.ts{i}/sr,'nearest');
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
