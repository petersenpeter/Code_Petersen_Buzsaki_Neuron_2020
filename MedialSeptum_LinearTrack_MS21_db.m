% Medial Septum Cooling project
clear all
%MedialSeptum_Recordings
% bz_database = db_credentials;
sessionName = 'Peter_MS21_180717_234514'%,'Peter_MS21_180714_232423'; % 'Peter_MS21_180808_115125_concat', 'Peter_MS21_180718_103455_concat'
tic
[session, basename, basepath, clusteringpath] = db_set_session('sessionName',sessionName);
% session = db_update_session(session)
% Intan_rec_info = read_Intan_RHD2000_file_from_basepath('/Volumes/buzsakilab/peterp03/IntanData/MS22/Peter_MS22_180719_145553');
fname = [session.general.name '.dat'];
sr = session.extracellular.sr;
sr_lfp = session.extracellular.srLfp;
nChannels = session.extracellular.nChannels;

% sr = Intan_rec_info.frequency_parameters.amplifier_sample_rate;
% time_frame = recording.time_frame;
% lfp_periods = 30*60; % in seconds

% ch_lfp = recording.ch_lfp;
% ch_medialseptum = recording.ch_medialseptum;
% ch_hippocampus = recording.ch_hippocampus;
% inputs.ch_wheel_pos = recording.ch_wheel_pos; % Wheel channel (base 1)
% inputs.ch_temp = recording.ch_temp; % Temperature data included (base 1)
% inputs.ch_peltier_voltage = recording.ch_peltier; % Peltier channel (base 1)
% inputs.ch_fan_speed = recording.ch_fan_speed; % Fan speed channel (base 1)
% inputs.ch_camera_sync = recording.ch_camera_sync;
% inputs.ch_OptiTrack_sync = recording.ch_OptiTrack_sync;
% inputs.ch_CoolingPulses = recording.ch_CoolingPulses;
% inputs.ch_opto_on = recording.ch_opto_on; % Blue laser diode
% inputs.ch_opto_off = recording.ch_opto_off; % Red laser diode
% recording.sr_lfp = sr/16;
% track_boundaries = recording.track_boundaries;
% arena = recording.arena;
% nb_lfp_periods = ceil((time_frame(2)-time_frame(1))/lfp_periods);
% temp_ = dir(fname);
% recording_length = round(temp_.bytes/sr/nChannels/2)-1;

% Optitrack: Loading position data
disp('2. Loading Optitrack tracking data')
if isfield(session.inputs,'OptitrackSync') && session.inputs.OptitrackSync.channels > 0
    if ~exist('Optitrack.mat')
        optitrack_files = find(strcmp(session.behavioralTracking.equipment,'OptiTrack, Flex 13'));
        optitrack_subsessions = session.behavioralTracking.subsession(optitrack_files);
        Optitrack = LoadOptitrack(session.behavioralTracking.filenames(optitrack_files),1,session.subSessions.mazeType(optitrack_subsessions(1)),0,0)
        save('Optitrack.mat','Optitrack')
    else
        load('Optitrack.mat');
    end
end

if  isfield(session.analysisTags, 'OptiTracktracking_offset') && ~isempty(session.analysisTags.OptiTracktracking_offset)
    Optitrack.position3D = Optitrack.position3D + session.analysisTags.OptiTracktracking_offset';
end

Optitrack.position1D = Optitrack.position3D(2,:);
% Loading digital inputs
disp('Loading digital inputs')

if ~exist('digitalchannels.mat')
    [digital_on,digital_off] = Process_IntanDigitalChannels('digitalin.dat');
    save('digitalchannels.mat','digital_on','digital_off')
else load('digitalchannels.mat'); end

prebehaviortime = 0;
behaviortime = session.general.duration;
% if recording.concat_behavior_nb > 0
%     prebehaviortime = 0;
%     if recording.concat_behavior_nb > 1
%     for i = 1:recording.concat_behavior_nb-1
%         fullpath = fullfile([datapath, recording.name(1:6) recording.animal_id], recording.concat_recordings{i}, [recording.concat_recordings{i}, '.dat']);
%         temp2_ = dir(fullpath);
%         prebehaviortime = prebehaviortime + temp2_.bytes/nChannels/2/sr;
%     end
%     end
%     i = recording.concat_behavior_nb;
%     fullpath = fullfile([datapath, recording.name(1:6) recording.animal_id], recording.concat_recordings{i}, [recording.concat_recordings{i}, '.dat']);
%     temp2_ = dir(fullpath);
%     behaviortime = temp2_.bytes/nChannels/2/sr;
% else
%      temp_ = dir(fname);
%      behaviortime = temp_.bytes/nChannels/2/sr;
% end

disp('4. Calculating behavior')
if session.inputs.OptitrackSync.channels == 0
if session.inputs.BaslerSync.channels ~= 0
    animal.sr = session.behavioralTracking.framerate(1);
    if length(session.subSessions.name) > 0
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

if session.inputs.OptitrackSync.channels ~= 0
    Optitrack.onset = min(digital_on{session.inputs.OptitrackSync.channels}(1),digital_off{session.inputs.OptitrackSync.channels}(1))/sr;
    Optitrack.offset = (session.general.duration)-max(digital_on{session.inputs.OptitrackSync.channels}(end),digital_off{session.inputs.OptitrackSync.channels}(end))/sr;
    animal.sr = Optitrack.FrameRate; % 100

    if isfield(Optitrack,'FramesPrFile') && length(Optitrack.FramesPrFile)>1
        test = find(diff(digital_on{session.inputs.OptitrackSync.channels})>3*sr/Optitrack.FrameRate);
        input_ttl = digital_on{session.inputs.OptitrackSync.channels};
        ttlremoved = 0;
        Optitrack.FramesPrFileCumSum = cumsum(Optitrack.FramesPrFile);
        for j = 1:length(test)
            if Optitrack.FramesPrFileCumSum(j) < test(j) - ttlremoved
                input_ttl(Optitrack.FramesPrFileCumSum(j)+1:test(j)-ttlremoved) = [];
                ttlremoved = + test(j) - Optitrack.FramesPrFileCumSum(j);
            elseif Optitrack.FramesPrFileCumSum(j) > test(j) - ttlremoved
                frame_diff = Optitrack.FramesPrFileCumSum(j) - ( test(j) - ttlremoved );
                input_ttl = input_ttl([1:(test(j) - ttlremoved),(test(j) - ttlremoved):Optitrack.FramesPrFileCumSum(j),(test(j) - ttlremoved)+1:end]);
                ttlremoved = ttlremoved - frame_diff
            end
        end
        animal.time = input_ttl'/sr;
    else
        animal.time = digital_on{session.inputs.OptitrackSync.channels}'/sr;
    end
    find(diff(animal.time)>0.009)
    animal.pos  = Optitrack.position3D([2,1,3],:) + [105,1,0]';
    %animal.pos(:,find(animal.pos(2,:)>70)) = 0;
    gausswin_size = animal.sr/2;
    for i = 1:3
        animal.pos(i,:) = medfilt1(animal.pos(i,:),5);
        animal.pos(i,:) = nanconv(animal.pos(i,:),gausswin(gausswin_size)','edge');
    end
    %animal.speed  = Optitrack.animal_speed3D;
    animal.speed  = [diff(animal.pos(1,:))*Optitrack.FrameRate,0];
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

animal.pos_x_limits = session.analysisTags.pos_x_limits;% [-70,240]; % x direction [5,145]
animal.pos_y_limits = session.analysisTags.pos_y_limits;% [-30,30]; % y direction
trials = trials_lineartrack(animal.pos, animal.pos_x_limits, animal.pos_y_limits);
animal.ab = zeros(size(animal.time));
animal.ba = zeros(size(animal.time));
startIndicies = trials.ab.start;
stopIndicies = trials.ab.end;
X = cumsum(accumarray(cumsum([1;stopIndicies(:)-startIndicies(:)+1]),[startIndicies(:);0]-[0;stopIndicies(:)]-1)+1);
X = X(1:end-1);
animal.ab(X) = 1;

startIndicies = trials.ba.start;
stopIndicies = trials.ba.end;
X = cumsum(accumarray(cumsum([1;stopIndicies(:)-startIndicies(:)+1]),[startIndicies(:);0]-[0;stopIndicies(:)]-1)+1);
X = X(1:end-1);
animal.ba(X) = 1;

% Loading Temperature data
if session.inputs.Temperature.channels >0
    disp('5. Loading Temperature data')
    if ~exist('temperature.mat')
        temperature = LoadTemperature(session.inputs.Temperature.channels,session.inputs.Temperature.inputType,pwd);
        animal.temperature = interp1(temperature.time,temperature.temp,animal.time);
    else
        load('temperature.mat');
        animal.temperature = interp1(temperature.time,temperature.temp,animal.time);
    end
else
    disp('No temperature data available')
    temperature = [];
end

if isfield(session.analysisTags,'CoolingSession') && session.analysisTags.CoolingSession == 0
    cooling.onsets = animal.time(round(length(animal.time)/2));
    cooling.offsets = animal.time(round(length(animal.time)));
    cooling.cooling = [cooling.onsets(1)+10;cooling.offsets(1)];
    cooling.nocooling = [[1,cooling.onsets(1)];[cooling.offsets(1)+120,behaviortime]]';
else
    if session.inputs.Temperature.channels ~= 0
        temp_range = [34,34];% temp_1 defines the upper limit on cooling, temp_2 the lower limit on no cooling
        test = find(diff(temperature.temp < temp_range(1),2)== 1);
        test(diff(test)<10*temperature.sr)=[];
        cooling.onsets = temperature.time(test(temperature.temp(test+1)-temperature.temp(test)<0));
        cooling.offsets = temperature.time(test(temperature.temp(test+1)-temperature.temp(test)>0));
        if length(cooling.offsets)<length(cooling.onsets)
            cooling.offsets = [cooling.offsets,temperature.time(end)]
        end
        cooling.cooling = zeros(2,length(cooling.onsets));
        for i= 1:length(cooling.onsets)
            find( cooling.offsets > cooling.onsets(i),1)
            cooling.cooling(:,i) = [cooling.onsets(i),cooling.offsets(find( cooling.offsets > cooling.onsets(i),1))];
        end
        %cooling.cooling = [cooling.onsets;cooling.offsets];
        cooling.cooling2 = [cooling.cooling(1,:)-20;cooling.cooling(2,:)];
        cooling.nocooling = reshape([prebehaviortime;cooling.cooling2(:);prebehaviortime+behaviortime],[2,size(cooling.cooling2,2)+1]);
        
    elseif isfield(session.inputs,'CoolingPulses') && session.inputs.CoolingPulses.channels ~= 0
        cooling.onsets = digital_on{session.inputs.CoolingPulses.channels}/sr;
        cooling.offsets = cooling.onsets + 12*60;
        cooling.cooling = [cooling.onsets(1)+10;cooling.offsets(1)];
        cooling.nocooling = [[1,cooling.onsets'];[cooling.offsets'+120,behaviortime]]';
    else
        cooling.onsets = recording.cooling_onsets;
        cooling.offsets = recording.cooling_offsets;
        cooling.cooling = [cooling.onsets(1)+10;cooling.offsets(1)]+prebehaviortime;
        cooling.nocooling = [[1,cooling.onsets(1)]+prebehaviortime;[cooling.offsets(1)+120,behaviortime]+prebehaviortime]';
    end
end

theta = [];
theta.sr = session.extracellular.srLfp;
theta.ch_theta = session.channelTags.Theta.channels;
% recording.name = session.general.name;
% recording.sr = session.extracellular.sr;
% recording.nChannels = session.extracellular.nChannels;
% recording.ch_theta = session.channelTags.Theta.channels;
InstantaneousTheta = calcInstantaneousTheta2(session);
theta.phase = InstantaneousTheta.signal_phase;
theta.phase2 = InstantaneousTheta.signal_phase2;
theta.freq = InstantaneousTheta.signal_freq;
theta.sr_freq = 10;
theta.time = [1:length(theta.phase)]/theta.sr;

trials.labels = {'AB','BA'};
trials.total = length(trials.ab.start);
figure, subplot(1,2,1)
plot(animal.pos(1,:),trials.trials{1},'b'), hold on, plot(animal.pos(1,:),trials.trials{2}+0.25,'r'),title('Trials'), axis tight
subplot(1,2,2)
plot3(animal.pos(1,:),animal.pos(2,:),animal.pos(3,:),'-k'), hold on
plot3(animal.pos(1,find(~isnan(trials.trials{1}))),animal.pos(2,find(~isnan(trials.trials{1}))),animal.pos(3,find(~isnan(trials.trials{1}))),'.b')
plot3(animal.pos(1,find(~isnan(trials.trials{2}))),animal.pos(2,find(~isnan(trials.trials{2}))),animal.pos(3,find(~isnan(trials.trials{2}))),'.r'),title('Position'), axis tight

x_hist = [0:5:300];
figure,
subplot(3,1,1)
plot(animal.time,animal.pos(1,:),'k'),axis tight, title('Position'), hold on
plot(cooling.cooling,[0,0],'b','linewidth',2), hold on
plot(cooling.nocooling,[0,0],'r','linewidth',2), 
legend({'Position','Cooling'})
subplot(3,1,2)
plot(animal.time,animal.speed),axis tight, title('Speed')
subplot(3,1,3)
plot(temperature.time,temperature.temp,'r'),axis tight, title('Temperature')
save('animal.mat','animal')
trials.temperature.ab = animal.temperature(find(diff(animal.ab)==1));
trials.temperature.ba = animal.temperature(find(diff(animal.ba)==1));

%% % Plotting theta frequency vs running speed
animal.speed2 = nanconv(animal.speed,gausswin(animal.sr/2)'/sum(gausswin(animal.sr/2)),'edge');
animal.acceleration2 = nanconv(animal.acceleration,gausswin(animal.sr/2)'/sum(gausswin(animal.sr/2)),'edge');

sloperecordings = session.analysisTags.sloperecordings+1;
invervals = [0,cumsum(session.subSessions.duration)];
interval = [invervals(sloperecordings(1)),invervals(sloperecordings(2))] 
[y1,y2,y3,y4,time1,time2,time3,time4] = plot_ThetaVsSpeed2(session,animal,cooling,interval); title(session.subSessions.notes{sloperecordings(1)}(1:18))
figure(101)
subplot(2,1,1)
plot(time1,y1,'-or'), hold on, grid on
plot(time2,y2,'-xr'), hold on, grid on
subplot(2,1,2)
plot(time3,y3,'-or'), hold on, grid on
plot(time4,y4,'-xr'), hold on, grid on
interval = [invervals(sloperecordings(3)),invervals(sloperecordings(4))]
[y1,y2,y3,y4,time1,time2,time3,time4] = plot_ThetaVsSpeed2(session,animal,cooling,interval); title(session.subSessions.notes{sloperecordings(3)}(1:18))
figure(101)
subplot(2,1,1)
plot(time1,y1,'-og'), hold on, grid on
plot(time2,y2,'-xg'), hold on, grid on, ylabel('Frequency (Hz)'), xlabel('Speed (cm/s)')
subplot(2,1,2)
plot(time3,y3,'-og'), hold on, grid on
plot(time4,y4,'-xg'), hold on, grid on, ylabel('Power'), xlabel('Speed (cm/s)')
interval = [invervals(sloperecordings(5)),invervals(sloperecordings(6))]
[y1,y2,y3,y4,time1,time2,time3,time4] = plot_ThetaVsSpeed2(session,animal,cooling,interval); title(session.subSessions.notes{sloperecordings(5)}(1:18))
figure(101)
subplot(2,1,1)
plot(time1,y1,'-ob'), hold on, grid on
plot(time2,y2,'-xb'), hold on, grid on
subplot(2,1,2)
plot(time3,y3,'-ob'), hold on, grid on
plot(time4,y4,'-xb'), hold on, grid on
legend({session.subSessions.notes{sloperecordings(1)}(1:18),session.subSessions.notes{sloperecordings(3)}(1:18),session.subSessions.notes{sloperecordings(5)}(1:18)})

speeed_threshold = 10;
interval = [invervals(sloperecordings(1)),invervals(sloperecordings(2))] 
[y1,y2,y3,y4,time1,time2,time3,time4] = plot_ThetaVsAcceleration2(session,animal,cooling,speeed_threshold,interval);
figure(100)
subplot(2,2,1)
plot(time1,y1,'or'), hold on, grid on
subplot(2,2,2)
plot(time2,y2,'xr'), hold on, grid on
subplot(2,2,3)
plot(time3,y3,'or'), hold on, grid on
subplot(2,2,4)
plot(time4,y4,'xr'), hold on, grid on
interval = [invervals(sloperecordings(3)),invervals(sloperecordings(4))]
[y1,y2,y3,y4,time1,time2,time3,time4] = plot_ThetaVsAcceleration2(session,animal,cooling,speeed_threshold,interval);
figure(100)
subplot(2,2,1)
plot(time1,y1,'og'), hold on, grid on
subplot(2,2,2)
plot(time2,y2,'og'), hold on, grid on
subplot(2,2,3)
plot(time3,y3,'og'), hold on, grid on
subplot(2,2,4)
plot(time4,y4,'og'), hold on, grid on
interval = [invervals(sloperecordings(5)),invervals(sloperecordings(6))]
[y1,y2,y3,y4,time1,time2,time3,time4] = plot_ThetaVsAcceleration2(session,animal,cooling,speeed_threshold,interval);
figure(100)
subplot(2,2,1)
plot(time1,y1,'ob'), hold on, grid on, ylabel('Frequency (Hz)'),xlabel('Acceleration (cm/s^2)'), title('Cooling')
subplot(2,2,2)
plot(time2,y2,'ob'), hold on, grid on, ylabel('Frequency (Hz)'),xlabel('Acceleration (cm/s^2)'), title('NoCooling')
subplot(2,2,3)
plot(time3,y3,'ob'), hold on, grid on, ylabel('Power'),xlabel('Acceleration (cm/s^2)')
subplot(2,2,4)
plot(time4,y4,'ob'), hold on, grid on, ylabel('Power'),xlabel('Acceleration (cm/s^2)')
legend({session.subSessions.notes{sloperecordings(1)}(1:18),session.subSessions.notes{sloperecordings(3)}(1:18),session.subSessions.notes{sloperecordings(5)}(1:18)})

%% % Loading units
pos_x_bins = animal.pos_x_limits(1):10:animal.pos_x_limits(2);

spikes = loadClusteringData(recording.name,recording.SpikeSorting.method,recording.SpikeSorting.path);

cooling.cooling3 = cooling.cooling;
cooling.nocooling3 = cooling.nocooling;
% cooling.cooling3 = cooling.cooling(:,find(cooling.cooling(1,:)>session.subSessions.Duration(1)));
% cooling.nocooling3 = cooling.nocooling(:,find(cooling.nocooling(1,:)>session.subSessions.Duration(1))-1);
index = ~isnan(trials.trials{2});
cooling.trials_on = interp1(animal.time(index),trials.trials{2}(index), cooling.cooling3,'nearest');
cooling.trials_off = interp1([0,animal.time(index)],[0,trials.trials{2}(index)], cooling.nocooling3,'nearest');
cooling.trials_on = [32,67;98,126;157,187]';
cooling.trials_off = [1,32;67,98;126,157;187,188]';

trials.time_states = [];
for j = 1:size(cooling.trials_on,2)
    trials.time_states(1,j) =  sum(trials.ab.end(cooling.trials_off(1,j):cooling.trials_off(2,j))'-trials.ab.start(cooling.trials_off(1,j):cooling.trials_off(2,j))')/animal.sr;
    trials.time_states(1,j + size(cooling.trials_on,2)) =  sum(trials.ab.end(cooling.trials_on(1,j):cooling.trials_on(2,j))'-trials.ab.start(cooling.trials_on(1,j):cooling.trials_on(2,j))')/animal.sr;
    trials.time_states(2,j) =  sum(trials.ba.end(cooling.trials_off(1,j):cooling.trials_off(2,j))'-trials.ba.start(cooling.trials_off(1,j):cooling.trials_off(2,j))')/animal.sr;
    trials.time_states(2,j + size(cooling.trials_on,2)) =  sum(trials.ba.end(cooling.trials_on(1,j):cooling.trials_on(2,j))'-trials.ba.start(cooling.trials_on(1,j):cooling.trials_on(2,j))')/animal.sr;
end

for i = 1:size(units,2)
    units(i).total = length(units(i).ts);
    units(i).ts_eeg = ceil(units(i).ts/16);
    units(i).theta_phase = theta.phase(units(i).ts_eeg);
    % units(i).theta_phase2 = theta.phase2(units(i).ts_eeg);
    units(i).theta_freq = interp1([1:length(theta.freq)]/theta.sr_freq,theta.freq,units(i).ts/sr);
    units(i).pos = interp1(animal.time,animal.pos(1,:),units(i).ts/sr);
    units(i).pos3d = interp1(animal.time,animal.pos',units(i).ts/sr)';
    units(i).speed = interp1(animal.time,animal.speed,units(i).ts/sr);
    units(i).state = interp1(animal.time,trials.state,units(i).ts/sr,'nearest');
    units(i).temperature = interp1(temperature.time,temperature.temp,units(i).ts/sr);
    %units(i).trials = interp1(animal.time,trials.trials2,units(i).ts/sr,'nearest');
end

units2plot = 1:length(units);
colors = {'b.','r.'};
RHO = [];
PVAL = [];
rate_bars = [];

chListBrainRegions = findBrainRegion(session);
% BrainRegions = (fieldnames(session.BrainRegions));

findBrainRegion = @(Channel,BrainRegions) BrainRegions{find([(struct2array(structfun(@(x) any(Channel==x.Channels)==1, session.BrainRegions,'UniformOutput',false)))])};

for i = 1:length(units2plot)
    unit_id = units2plot(i);
    figure
    for k = 1:2
        N = zeros(2*size(cooling.trials_on,2),length(pos_x_bins)-1);
        clear unit;
        unit.state = interp1(animal.time,trials.state,units(unit_id).ts/sr,'nearest');
        unit.ts = units(unit_id).ts(find(unit.state==k));
        unit.pos = interp1(animal.time,animal.pos(1,:),units(unit_id).ts(find(unit.state==k))/sr);
        unit.pos3d = interp1(animal.time,animal.pos',units(unit_id).ts(find(unit.state==k))/sr)';
        unit.speed = interp1(animal.time,animal.speed,units(unit_id).ts(find(unit.state==k))/sr);
        unit.acceleration = interp1(animal.time,animal.acceleration,units(unit_id).ts(find(unit.state==k))/sr);
        unit.theta_phase = interp1(theta.time,theta.phase,units(unit_id).ts(find(unit.state==k))/sr);
        unit.trials = interp1(animal.time,trials.trials{k},units(unit_id).ts(find(unit.state==k))/sr,'nearest');
        unit.shank = units(unit_id).shank;
        unit.peak_channel = units(unit_id).peak_channel;
        
        temp2 = hist(units(unit_id).ts/sr,[1:units(unit_id).ts(end)/(sr/1000)]/1000);
        gausswin_size=2000;
        firing_rate = conv(temp2,gausswin(gausswin_size)','same');
        unit.firing_rate = interp1([1:length(firing_rate)]/1000,firing_rate,units(unit_id).ts(find(unit.state==k))/sr);
                
        subplot(2,5,1+(k-1)*5)
        plot(animal.pos(1,:),animal.pos(3,:),'color',[0.9,0.9,0.9]), hold on
        plot(unit.pos3d(1,:),unit.pos3d(3,:),colors{k})
        title(['Position, Unit ', num2str(unit_id) ' shank ' num2str(unit.shank)]),xlabel('X'),ylabel('Y'), axis tight
%         legend({[num2str(units(unit_id).total/(units(unit_id).ts(end)/sr),3) 'Hz'],'ab','ba'}), 
        
        subplot(2,5,2+(k-1)*5)
        plot_PhasePrecession(unit,colors{k}(1)); hold on
        title(['Phase precession, ' findBrainRegion(unit.peak_channel,BrainRegions)]),xlabel('Position'),ylabel('Phase'), axis tight
        
        subplot(2,5,3+(k-1)*5)
        plot(unit.pos,unit.trials+(k-1)/2,colors{k}), hold on
        plot([animal.pos_x_limits(1),animal.pos_x_limits(1)]-5,cooling.trials_on,'b','linewidth',2),
        plot([animal.pos_x_limits(1),animal.pos_x_limits(1)]-5,cooling.trials_off,'r','linewidth',2), axis tight
        gridxy([],cooling.trials_on(:)), title('Trials'),xlabel('Position'),ylabel('Trials')

        for j = 1:size(cooling.trials_on,2)
            indexes = find(unit.trials > cooling.trials_off(1,j) & unit.trials < cooling.trials_off(2,j));
            [N_hist,edges] = histcounts(unit.pos(indexes),pos_x_bins);
            N(j,:) = N_hist;

            indexes = find(unit.trials > cooling.trials_on(1,j) & unit.trials < cooling.trials_on(2,j));
            [N_hist,edges] = histcounts(unit.pos(indexes),pos_x_bins);
            N(j+size(cooling.trials_on,2),:) = N_hist;
        end
        
        subplot(2,5,4+(k-1)*5)
        rate_bars{i} = sum(N')./trials.time_states(k,:);
        bar(reshape(rate_bars{i},[3,2])), 
        title(['Firing rate: ', num2str(units(unit_id).total/(units(unit_id).ts(end)/sr),3) 'Hz']),xlabel('States'),ylabel('Rate'), axis tight
        
        subplot(2,5,5+(k-1)*5)
        plot(pos_x_bins(1:end-1),N), axis tight,
        title('Firing rate'),xlabel('Position'),ylabel('Rate'), axis tight
    end
end
temp = vertcat(rate_bars{:})';
figure, subplot(2,1,1), plot(temp(1:3,:)), subplot(2,1,2), plot(temp(4:6,:))

animal.theta = interp1([1:length(theta.freq)]/theta.sr_freq,theta.freq,animal.time);

theta_mean = [];
for j = 1:size(cooling.trials_on,2)
    interval = [];
    interval{1} = [trials.ab.start(cooling.trials_off(1,j):cooling.trials_off(2,j))',trials.ab.end(cooling.trials_off(1,j):cooling.trials_off(2,j))'];
    interval{2} = [trials.ab.start(cooling.trials_on(1,j):cooling.trials_on(2,j))',trials.ab.end(cooling.trials_on(1,j):cooling.trials_on(2,j))'];
    interval{3} = [trials.ba.start(cooling.trials_off(1,j):cooling.trials_off(2,j))',trials.ba.end(cooling.trials_off(1,j):cooling.trials_off(2,j))'];
    interval{4} = [trials.ba.start(cooling.trials_on(1,j):cooling.trials_on(2,j))',trials.ba.end(cooling.trials_on(1,j):cooling.trials_on(2,j))'];
    
    for k = 1:4
        startIndicies = interval{k}(:,1);
        stopIndicies = interval{k}(:,2);
        X = cumsum(accumarray(cumsum([1;stopIndicies(:)-startIndicies(:)+1]),[startIndicies(:);0]-[0;stopIndicies(:)]-1)+1);
        X = X(1:end-1);
        theta_mean = [theta_mean,(mean(animal.theta(X)))];
    end
end
figure
plot(reshape(theta_mean,2,[])), legend({'ab -10', 'ba -10', 'ab +10', 'ba +10', 'ab 0', 'ba 0'})

%%
% stats = {};
% load('GammaStats.mat','stats')
recording.sr_lfp = 1250;
for i = 57%1:56%[58:recording.nChannels]
    i
    recording.ch_theta = i;
    
    stats{i} = plot_GammaVsTemperature(recording,animal,cooling);
    figure(1000)
    stairs(stats{i}.freqlist,mean(stats{i}.freq_cooling),'b'), hold on
    stairs(stats{i}.freqlist,mean(stats{i}.freq_nocooling),'r')
end
save('GammaStats.mat','stats')

%% 

[session, basename, basepath, clusteringpath] = db_set_path('session',sessionName);
load('GammaStats.mat','stats')
temp = [];
xml = LoadXml(recording.name,'.xml');
colorpalet = jet(20); %[0,0,0; 1,0,1; 1,0,0; 0.5,0.5,0.5; 0.2,0.2,0.2; 0,1,0;  0,1,1];
[BadChannels,GoodChannels] = db_BadChannels(session,xml);

for i = GoodChannels
    temp(:,i) = mean(stats{i}.freq_cooling)./mean(stats{i}.freq_nocooling);
end
freqlist = stats{i}.freqlist;
legend1 = {};
figure
for j = 1:length(xml.SpkGrps)
    subplot(4,4,j)
    h1{j} = plot(freqlist,temp(:,xml.SpkGrps(j).Channels+1)); hold on, grid on,
    title(['Shank ',num2str(j)]),xlim([freqlist(1),freqlist(end)]),ylim([0,1.4])
    if j == 1
        xlabel('Frequency (Hz)'), ylabel('Gamma (Power ratio)')
    end
end
clear h1

%% Linear track in darkness
boundary = 100;
idx1 = find(diff(animal.pos(1,:)<100) == 1);
idx2 = find(diff(animal.pos(1,:)<100) == -1);
distance = [];
idx3 = [];
figure, 
subplot(3,1,1)
plot(animal.time,animal.pos(1,:)), hold on
plot(animal.time(idx1),animal.pos(1,idx1),'or')
plot(animal.time(idx2),animal.pos(1,idx2),'xr')
for i = 50:127%length(idx1)
    interval = idx1(i):idx2(i);
    [test,test2] = min(animal.pos(1,interval));
    idx3(i) = interval(test2);
    distance(i) = 190-test;
    if distance(i) > 180 || distance(i) < 10
        distance(i) = nan;
        idx3(i) = nan;
    else
       text(animal.time(interval(test2)),animal.pos(1,interval(test2)),num2str(i))
    end
%     plot(animal.time(interval),animal.pos(1,interval),'.k')
end
distance(idx3==0 | isnan(idx3)) = [];
idx3(idx3==0 |  isnan(idx3)) = [];
subplot(3,2,3) 
plot(animal.time,animal.temperature(1,:),'-')
subplot(3,3,7)
plot(distance,'o')
subplot(3,3,8)
temperature = animal.temperature;
temp33  = distance(idx3>0);
temp44  = temperature(idx3);

idx_pre = 1:10;
idx_cooling = [11,13:36];
idx_post = 37:73;
idx_all = [idx_pre,idx_cooling,idx_post];
plot(temp44(idx_all), temp33(idx_all),'o'), xlabel('Temperature'), ylabel('Running distance (cm)'), hold on
lsline

% Pre vs cooling
[h1,p1] = kstest2(temp33(idx_pre),temp33(idx_cooling));
% Cooling vs post
[h2,p2] = kstest2(temp33(idx_cooling),temp33(idx_post));
% Pre vs post
[h3,p3] = kstest2(temp33(idx_pre),temp33(idx_post));

subplot(3,3,9)
boxplot([temp33(idx_pre),temp33(idx_cooling),temp33(idx_post)],[ones(1,length(idx_pre)),2*ones(1,length(idx_cooling)),3*ones(1,length(idx_post))]), hold on
subplot(3,2,4), plot([ones(1,length(idx_pre)),2*ones(1,length(idx_cooling)),3*ones(1,length(idx_post))],[temp33(idx_pre),temp33(idx_cooling),temp33(idx_post)],'o')

%%
spikes = loadSpikes('session',session);
for i = 1:size(spikes.ts,2)
    spikes.pos_linearized{i} = interp1(animal.time,animal.pos(1,:),spikes.times{i});
    figure, 
    plot(spikes.times{i},spikes.pos_linearized{i},'.')
end


