clear all
MedialSeptum_Recordings
id = 134;
recording = recordings(id);
if ~isempty(recording.dataroot)
    datapath = recording.dataroot;
end
datapath = 'Z:\peterp03\IntanData';
cd(fullfile(datapath, recording.animal_id, recording.name))
Intan_rec_info = read_Intan_RHD2000_file_Peter(fullfile(datapath, recording.animal_id, recording.name));
fname = [recording.name '.dat'];
nChannels = size(Intan_rec_info.amplifier_channels,2);

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
maze = recording.maze;
recording.sr_lfp = sr/16;
% track_boundaries = recording.track_boundaries;
arena = recording.arena;

animal = [];

% Loading Temperature data
if inputs.ch_temp >0
    disp('Loading Temperature data')
    if ~exist('temperature.mat')
        if isempty(recording.ch_temp_type)
            recording.ch_temp_type = 'analog';
        end
        temperature = LoadTemperature(recording.ch_temp,recording.ch_temp_type,pwd);
        %animal.temperature = interp1(temperature.time,temperature.temp,animal.time);
    else
        load('temperature.mat');
        %animal.temperature = interp1(temperature.time,temperature.temp,animal.time);
    end
else
    disp('No temperature data available')
    temperature = [];
end
% temperature.temp = nanconv(temperature.temp,gausswin(100)','edge');

disp('Loading wheel data')
if ~exist('wheeldata.mat')
    wheeldata = Load_Intan_wheel_position(Intan_rec_info,recording.ch_wheel_pos);
else
    load('wheeldata.mat');
end

if ~exist('opto_data.mat') & isfield(inputs,'ch_opto_on') && ~isempty(inputs.ch_opto_on)
    disp('Loading opto data')
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
    opto_stim_offset = opto_stim_onset+sr*15;
    for i = 1:length(opto_stim_onset)
        [~,ia,~] = intersect(opto_stim_off, opto_stim_onset(i):opto_stim_offset(i));
        if ia
            opto_stim_offset(i) = opto_stim_off(ia(1));
        end
    end
    opto.onset = opto_stim_onset/sr;
    opto.offset = opto_stim_offset/sr;
    save('opto_data.mat','opto');
    
elseif exist('opto_data.mat')
    load('opto_data.mat');
end

animal.pos = wheeldata.wheel_position;
animal.speed = abs(wheeldata.wheel_velocity);
animal.acceleration = wheeldata.sr*[0,diff(abs(wheeldata.wheel_velocity))];
animal.acceleration(find(animal.speed > 0)) = 0;
animal.time = wheeldata.time;
%animal.temperature = interp1(temperature.time,temperature.temp,animal.time);

%animal.opto = 
animal.sr = wheeldata.sr;

temp2_ = dir(fname);
behaviortime = temp2_.bytes/nChannels/2/sr;

prebehaviortime = 0;
cooling = []; 
if inputs.ch_temp ~= 0
    temp_range = [32,33];% temp_1 defines the upper limit on cooling, temp_2 the lower limit on no cooling
    test = find(diff(temperature.temp < temp_range(1),2)== 1);
    test(diff(test)<120*temperature.sr)=[];
    cooling.onsets = temperature.time(test(temperature.temp(test+1)-temperature.temp(test)<0));
    cooling.offsets = temperature.time(test(temperature.temp(test+1)-temperature.temp(test)>0));
    if length(cooling.offsets)<length(cooling.onsets)
        cooling.offsets = [cooling.offsets,temperature.time(end)]
    end
    cooling.cooling = [cooling.onsets;cooling.offsets];
    cooling.cooling2 = [cooling.onsets-20;cooling.offsets];
    cooling.nocooling = reshape([prebehaviortime;cooling.cooling2(:);prebehaviortime+behaviortime],[2,size(cooling.cooling2,2)+1]);
elseif inputs.ch_CoolingPulses ~= 0
    cooling.onsets = digital_on{inputs.ch_CoolingPulses}/sr;
    cooling.offsets = cooling.onsets + 12*60;
    cooling.cooling = [cooling.onsets(1)+10;cooling.offsets(1)];
    cooling.nocooling = [[1,cooling.onsets'];[cooling.offsets'+120,behaviortime]]';
else
    cooling.onsets = recording.cooling_onsets;
    cooling.offsets = recording.cooling_offsets;
    cooling.cooling = [cooling.onsets(1)+10;cooling.offsets(1)]+prebehaviortime;
    cooling.nocooling = [[1,cooling.onsets(1)]+prebehaviortime;[cooling.offsets(1)+120,behaviortime]+prebehaviortime]';
end

% % Opto
% cooling.onsets = opto.onset;
% cooling.offsets = opto.offset;
% cooling.cooling = [cooling.onsets;cooling.offsets];
% cooling.nocooling = [[1;cooling.onsets(1)],[cooling.offsets(1:end-1);cooling.onsets(2:end)],[cooling.offsets(end);behaviortime]];

figure,
subplot(3,1,1)
plot(animal.time,animal.pos)
title('Wheel position','fontsize',16), xlabel('Time (s)'),ylabel('Wheel position');

subplot(3,1,2)
plot(cooling.cooling,40*ones(2,size(cooling.cooling,2)),'b','linewidth',3), hold on
plot(cooling.nocooling,40*ones(2,size(cooling.nocooling,2)),'r','linewidth',3)
title('Temperature','fontsize',16), xlabel('Time (s)'),ylabel('Temperature');

subplot(3,1,3)
plot(animal.speed,'-k');
title('Wheel speed','fontsize',16), xlabel('Time (s)'),ylabel('Wheel Speed (cm/s)');

plot_ThetaVsSpeed(recording,animal,cooling);
% plot_ThetaVsAcceleration(recording,animal,cooling);

%% % Checking for time delay between the temperature and the change in theta frequency
theta = [];
theta.sr = recording.sr_lfp;
theta.ch_theta = recording.ch_theta;
InstantaneousTheta = calcInstantaneousTheta(recording);

% [signal_phase,signal_phase2,signal_freq]
theta.phase = InstantaneousTheta.signal_phase{theta.ch_theta};
theta.phase2 = InstantaneousTheta.signal_phase2{theta.ch_theta};
theta.freq = InstantaneousTheta.signal_freq{theta.ch_theta};
theta.sr_freq = 10;
theta.window = [300,600];
clear signal_phase signal_phase2

plot_temperature = [];
plot_theta = [];
for i = 1:length(recording.cooling_onsets)
    plot_temperature(:,i) = temperature.temp( (recording.cooling_onsets(i)-theta.window(1))*temperature.sr : (recording.cooling_onsets(i)+theta.window(2))*temperature.sr );
    plot_theta(:,i) = theta.freq((recording.cooling_onsets(i)-theta.window(1))*theta.sr_freq:(recording.cooling_onsets(i)+theta.window(2))*theta.sr_freq);
end

figure
subplot(2,1,1)
plot(plot_temperature), hold on, plot(mean(plot_temperature,2),'k','linewidth',2)
subplot(2,1,2)
plot(plot_theta), hold on, plot(mean(plot_theta,2),'k','linewidth',2)
