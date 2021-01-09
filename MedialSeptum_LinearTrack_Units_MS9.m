% Medial Septum Cooling project
clear all
Recordings_MedialSeptum
id = 53;

recording= recordings(id).name;
cd([datapath, recordings(id).name(1:6) recordings(id).rat_id '\' recording, '\'])
Intan_rec_info = read_Intan_RHD2000_file_Peter([datapath, recordings(id).name(1:6) recordings(id).rat_id '\' recording, '\']);
fname = 'amplifier.dat';
nbChan = size(Intan_rec_info.amplifier_channels,2);
cooling.cooling = recordings(id).cooling;
cooling.onsets = recordings(id).cooling_onsets;
cooling.offsets = recordings(id).cooling_offsets;
sr = Intan_rec_info.frequency_parameters.amplifier_sample_rate;
time_frame = recordings(id).time_frame;
lfp_periods = 30*60; % in seconds
ch_theta = recordings(id).ch_theta;
ch_lfp = recordings(id).ch_lfp;
ch_medialseptum = recordings(id).ch_medialseptum;
ch_hippocampus = recordings(id).ch_hippocampus;
inputs.ch_wheel_pos = recordings(id).ch_wheel_pos; % Wheel channel (base 1)
inputs.ch_temp = recordings(id).ch_temp; % Temperature data included (base 1)
inputs.ch_peltier_voltage = recordings(id).ch_peltier; % Peltier channel (base 1)
inputs.ch_fan_speed = recordings(id).ch_fan_speed; % Fan speed channel (base 1)
inputs.ch_camera_sync = recordings(id).ch_camera_sync;
inputs.ch_OptiTrack_sync = recordings(id).ch_OptiTrack_sync;
inputs.ch_CoolingPulses = recordings(id).ch_CoolingPulses;
track_boundaries = recordings(id).track_boundaries;
arena = recordings(id).arena;
nb_lfp_periods = ceil((time_frame(2)-time_frame(1))/lfp_periods);
temp_ = dir(fname);
recording_length = round(temp_.bytes/sr/nbChan/2)-1;

% Loading digital inputs
disp('Loading digital inputs')
if ~exist('digitalchannels.mat')
    [digital_on,digital_off] = Process_IntanDigitalChannels('digitalin.dat');
    save('digitalchannels.mat','digital_on','digital_off')
else load('digitalchannels.mat'); end
if inputs.ch_camera_sync ~= 0
    camera.onset = min(digital_on{inputs.ch_camera_sync}(1),digital_off{inputs.ch_camera_sync}(1))/sr;
    camera.offset = (temp_.bytes/sr/nbChan/2)-max(digital_on{inputs.ch_camera_sync}(end),digital_off{inputs.ch_camera_sync}(end))/sr;
end
if inputs.ch_OptiTrack_sync ~= 0
    Optitrack.onset = min(digital_on{inputs.ch_OptiTrack_sync}(1),digital_off{inputs.ch_OptiTrack_sync}(1))/sr;
    Optitrack.offset = (temp_.bytes/sr/nbChan/2)-max(digital_on{inputs.ch_OptiTrack_sync}(end),digital_off{inputs.ch_OptiTrack_sync}(end))/sr;
end
if inputs.ch_CoolingPulses ~= 0
    cooling.onsets = digital_on{inputs.ch_CoolingPulses}/sr;
end

% Loading Temperature data
disp('Loading Temperature data')
if inputs.ch_temp > 0
    if ~exist('temperature.mat')
        num_channels = length(Intan_rec_info.board_adc_channels); % ADC input info from header file
        fileinfo = dir('analogin.dat');
        num_samples = fileinfo.bytes/(num_channels * 2); % uint16 = 2 bytes
        fid = fopen('analogin.dat', 'r'); v = fread(fid, [num_channels, num_samples], 'uint16'); fclose(fid);
        v = v * 0.000050354; % convert to volts
        v_downsample = downsample(v(inputs.ch_temp,:),sr/sr_lfp);
        %v_downsample(v_downsample<1.25) = 1.25;
        temperature = nanconv((v_downsample-1.25)/0.005,gausswin(100)'/sum(gausswin(100)),'edge');
        save('temperature.mat','temperature')
        clear v_downsample v;
    else load('temperature.mat'); 
    end
else
    disp('No temperature data available')
    temperature = [];
end
cooling.cooling = [cooling.onsets'+30;cooling.offsets'];
cooling.nocooling = [[1,cooling.offsets'+30];[cooling.onsets'-30,recording_length]];

% Loading position data
disp('Loading tracking data')
if ~exist('Optitrack.mat')
    Optitrack = LoadOptitrack(recordings(id).OptiTracktracking,0,1,arena);
    save('Optitrack.mat','Optitrack')
else load('Optitrack.mat'); end
if Optitrack.FrameRate < 100
    sr_lfp = 30;
else
    sr_lfp = Optitrack.FrameRate;
end
animal = [];
animal.sr = Optitrack.FrameRate; % 100
animal.pos = 150+nanconv(Optitrack.position1D,gausswin(round(0.41*animal.sr))'/sum(gausswin(round(0.41*animal.sr))),'edge')';
animal.speed = nanconv(Optitrack.animal_speed1D,gausswin(round(0.41*animal.sr))'/sum(gausswin(round(0.41*animal.sr))),'edge');
animal.time = digital_on{inputs.ch_OptiTrack_sync}'/sr;

% temperature_cooling_thres = -64;
% temperature_nocooling_thres = -62;
% cooling_start = find(diff(temperature < temperature_cooling_thres)==1);
% cooling_stop = find(diff(temperature < temperature_cooling_thres)==-1);
% cooling = [cooling_start;cooling_stop]/sr_temperature;
% nocooling_start = find(diff(temperature > temperature_nocooling_thres)==1);
% nocooling_stop = find(diff(temperature > temperature_nocooling_thres)==-1);
% nocooling = [[0,nocooling_start];[nocooling_stop,recording_length*animal.sr]]/sr_temperature;
x_hist = [0:5:300];
figure,
subplot(3,1,1)
plot(animal.time,animal.pos,'k'),axis tight, title('Position'), hold on
plot(cooling.cooling,[0,0],'b','linewidth',2), hold on
plot(cooling.nocooling,[0,0],'r','linewidth',2), 
legend({'Position','Cooling'})
subplot(3,1,2)
plot(animal.time,animal.speed),axis tight, title('Speed')
subplot(3,1,3)
plot([1:length(temperature)]/sr_lfp,temperature,'r'),axis tight, title('Temperature')


%% % Loading units
 clustering_method = 'Klustakwik'; % .clu and .res format
% clustering_method = 'SpikingCircus'; % python file format
% clustering_method = 'Klusta'; % Kwik format
shanks = 1:3;
units = loadClusteringData(recording,shanks,clustering_method);
theta_channel = ch_lfp; % 31;

if ~exist([recording, '.eeg'])
    disp('Creating EEG file')
    downsample_dat_to_eeg([pwd '\']);
    movefile('amplifier.eeg',[recording '.eeg'])
    copyfile('amplifier.xml',[recording '.xml'])
end

% Calculating the instantaneous theta frequency
if ~exist('InstantaneousTheta.mat')
    % disp('Creating EEG file')
    % downsample_dat_to_eeg([datapath, recording,'\']);
    signal = 0.000050354 * double(LoadBinary([recording '.eeg'],'nChannels',nbChan,'channels',theta_channel,'precision','int16','frequency',sr/16)); % ,'start',start,'duration',duration
    Fs = sr/16;
    Fpass = [4,11];
    Wn_theta = [Fpass(1)/(Fs/2) Fpass(2)/(Fs/2)]; % normalized by the nyquist frequency
    [btheta,atheta] = butter(3,Wn_theta);
    signal_filtered = filtfilt(btheta,atheta,signal);
    signal_phase = atan2(imag(hilbert(signal_filtered)), signal_filtered);
    signal_phase2 = unwrap(signal_phase);
    save('InstantaneousTheta.mat','signal_phase','signal_phase2')
    clear signal,clear signal_filtered
else
    load('InstantaneousTheta.mat')
end

% disp('Plotting instantaneous theta frequency for all units')
% figure, plot(signal_phase)
for i = 1:size(units.ts,2)
    units.ts_eeg{i} = round(units.ts{i}/16);
    units.theta_phase{i} = signal_phase(units.ts_eeg{i});
    units.theta_phase2{i} = signal_phase2(units.ts_eeg{i});
    units.speed{i} = interp1(animal.time,animal.speed,units.ts{i}/sr);
    units.loc{i} = interp1(animal.time,animal.pos,units.ts{i}/sr);
end
clear signal_phase %signal_phase2
%%
place_preference = [];
place_preference_cooling = [];
place_preference_nocooling = [];

temp9 = histogram(animal.pos, x_hist);
N = temp9.Values;
cooling.loc_cooling = [];
cooling.loc_nocooling = [];
for k = 1:size(cooling.cooling,2)
    cooling_temp = find (animal.time > cooling.cooling(1,k) & animal.time < cooling.cooling(2,k));
    cooling.loc_cooling = [cooling.loc_cooling,cooling_temp];
end
temp9 = histogram(animal.pos(cooling.loc_cooling), x_hist);
N_cooling = temp9.Values;
for k = 1:size(cooling.nocooling,2)
    nocooling_temp = find (animal.time > cooling.nocooling(1,k) & animal.time < cooling.nocooling(2,k));
    cooling.loc_nocooling = [cooling.loc_nocooling,nocooling_temp];
end
temp9 = histogram(animal.pos(cooling.loc_nocooling), x_hist);
N_nocooling = temp9.Values;
for i = 1:size(units.ts,2)
    i
    spatial_spiking = histogram(units.ts{i}(1:end)./sr,animal.time);
    spatial_spiking2 = spatial_spiking.Values;
    place_preference2 = zeros(1,length(N));
    for j = 1:length(N)
            if N(j) > 1
                place_timepoints = find(animal.pos > x_hist(j) & animal.pos < x_hist(j+1));
                place_preference2(j) = sum(spatial_spiking2(place_timepoints(1:end-1)));
            end
    end
    place_preference{i} = (place_preference2./N)*animal.sr;
    place_preference{i}(isnan(place_preference{i})) = 0;
    
    % % % % % % % 
    % Cooling
    ts_cooling = [];
    for k = 1:size(cooling.cooling,2)
        cooling_ts_temp = find (units.ts{i}/sr > cooling.cooling(1,k) & units.ts{i}/sr < cooling.cooling(2,k));
        ts_cooling = [ts_cooling,cooling_ts_temp];
    end
    spatial_spiking = histogram(units.ts{i}(ts_cooling)./sr,animal.time);
    spatial_spiking2 = spatial_spiking.Values;
    place_preference2 = zeros(1,length(N));
    for j = 1:length(N)
            if N_cooling(j) > 1
                place_timepoints = find(animal.pos > x_hist(j) & animal.pos<x_hist(j+1) );
                place_preference2(j) = sum(spatial_spiking2(place_timepoints(1:end-1)));
            end
    end
    place_preference_cooling{i} = (place_preference2./N_cooling)*animal.sr;
    place_preference_cooling{i}(isnan(place_preference_cooling{i})) = 0;
    
    % % % % % % % % % % % %
    % No cooling
    ts_nocooling = [];
    for k = 1:size(cooling.nocooling,2)
        nocooling_ts_temp = find (units.ts{i}/sr > cooling.nocooling(1,k) & units.ts{i}/sr < cooling.nocooling(2,k));
        ts_nocooling = [ts_nocooling,nocooling_ts_temp];
    end
    spatial_spiking = histogram(units.ts{i}(ts_nocooling)./sr,animal.time);
    spatial_spiking2 = spatial_spiking.Values;
    place_preference2 = zeros(1,length(N));
    for j = 1:length(N)
            if N_cooling(j) > 1
                place_timepoints = find(animal.pos > x_hist(j) & animal.pos<x_hist(j+1) );
                place_preference2(j) = sum(spatial_spiking2(place_timepoints(1:end-1)));
            end
    end
    place_preference_nocooling{i} = (place_preference2./N_nocooling)*animal.sr;
    place_preference_nocooling{i}(isnan(place_preference_nocooling{i})) = 0;
end

%% % Plotting the spatial spiking
figure;
for i = 1:size(units.ts,2)
    subplot(8,6,i)
    plot(x_hist(1:end-1),place_preference_cooling{i},'b'), hold on
    plot(x_hist(1:end-1),place_preference_nocooling{i},'r')
    % plot(x_hist(1:end-1),place_preference{i}','--k')
    title(['Unit ' num2str(i)]), axis tight,xlim([35,245])
    %set(gca, 'XTickLabelMode', 'manual', 'XTickLabel', []);
    %set(gca, 'YTickLabelMode', 'manual', 'YTickLabel', []);
end
legend({'With Cooling','Without cooling','Both'}), hold off
%% % Plotting the instantaneous theta frequency
for i = 1:size(units.ts,2)
    ts_cooling = [];
    ts_nocooling = [];
    for k = 1:size(cooling,2)
        cooling_ts_temp = find (units.ts{i}/sr > cooling(1,k) & units.ts{i}/sr < cooling(2,k));
        ts_cooling = [ts_cooling;cooling_ts_temp];
    end
    for k = 1:size(nocooling,2)
        nocooling_ts_temp = find (units.ts{i}/sr > nocooling(1,k) & units.ts{i}/sr < nocooling(2,k));
        ts_nocooling = [ts_nocooling;nocooling_ts_temp];
    end
    if rem(i,10) == 1
        figure
    end
    subplot(3,5,rem(i-1,10)+1)
    plot(units.loc{i}(ts_cooling),units.theta_phase{i}(ts_cooling)-pi,'.b','markersize',0.5), hold on
    plot(units.loc{i}(ts_cooling),units.theta_phase{i}(ts_cooling)-3*pi,'.b','markersize',0.5)
    plot(units.loc{i}(ts_nocooling),units.theta_phase{i}(ts_nocooling)+pi,'.r','markersize',1)
    plot(units.loc{i}(ts_nocooling),units.theta_phase{i}(ts_nocooling)+3*pi,'.r','markersize',1), 
    hold on, plot([35,245],[0,0],'k','linewidth',0.2),
    plot([35,245],-2*pi*[1,1],'--k','linewidth',0.2),
    plot([35,245],2*pi*[1,1],'--k','linewidth',0.2)
    ax = gca;
    ax.YTick = ([-4*pi -3*pi -2*pi -pi 0 pi 2*pi 3*pi 4*pi]);
    ax.YTickLabels = ({'0','\pi','2\pi','3\pi','0','\pi','2\pi','3\pi','-4\pi',});
    hold off, axis tight, xlim([35,245]),ylim([-4*pi,4*pi]),
    title(['Unit ' num2str(i)])
    % set(gca, 'XTickLabelMode', 'manual', 'XTickLabel', []);
    % set(gca, 'YTickLabelMode', 'manual', 'YTickLabel', []);
    subplot(6,5,rem(i-1,10)+21)
    plot(units.ts{i}(ts_cooling),units.theta_phase{i}(ts_cooling)+pi,'.b','markersize',0.5), hold on
    plot(units.ts{i}(ts_cooling),units.theta_phase{i}(ts_cooling)+3*pi,'.b','markersize',0.5)
    plot(units.ts{i}(ts_nocooling),units.theta_phase{i}(ts_nocooling)+pi,'.r','markersize',1)
    plot(units.ts{i}(ts_nocooling),units.theta_phase{i}(ts_nocooling)+3*pi,'.r','markersize',1), 
    title(['Unit ' num2str(i)])
    hold off, axis tight, ylim([0,4*pi]),
end
%% % Determining the direction of the movement to separate the behavior into inbound/outbound trials




pos_norm2 = animal.pos;
pos_norm3 = animal.pos;
pos_norm4 = zeros(length(animal.pos),1);
pos_norm2(animal.pos < track_boundaries(1)) = NaN;
pos_norm3(animal.pos > track_boundaries(2)) = NaN;
pos_norm4(animal.pos > track_boundaries(2) & animal.pos < track_boundaries(1)) = NaN;

a_inbound = find(diff(isnan(pos_norm2))>0);
a_outbound = find(diff(isnan(pos_norm2))<0);
b_inbound = find(diff(isnan(pos_norm3))>0);
b_outbound = find(diff(isnan(pos_norm3))<0);
speed_min = 5;
pos_ab = [];
pos_ba = [];
for i = 1:length(a_outbound)
    a = a_inbound-a_outbound(i);
    b = b_inbound-a_outbound(i);
    a(a<0)=NaN; b(b<0)=NaN;
    a = min(a); b = min(b);
    [next,I] = sort([a,b]);
    if ~isnan(next(1))
        if I(1) == 2
            pos_norm4(a_outbound(i):a_outbound(i)+next(1)) = 1;
            pos_ab = [pos_ab;a_outbound(i),a_outbound(i)+next(1)];
        else
            pos_norm4(a_outbound(i):a_outbound(i)+next(1)) = 2;
        end
    end
end

for i = 1:length(b_outbound)
    a = b_inbound-b_outbound(i);
    b = a_inbound-b_outbound(i);
    a(a<0)=NaN; b(b<0)=NaN;
    a = min(a); b = min(b);
    [next,I] = sort([a,b]);
    if ~isnan(next(1))
        if I(1) == 2
            pos_norm4(b_outbound(i):b_outbound(i)+next(1)) = -1;
            pos_ba = [pos_ba;b_outbound(i),b_outbound(i)+next(1)];
        else
            pos_norm4(b_outbound(i):b_outbound(i)+next(1)) = -2;
        end
    end
end
% time = 1:length(animal.pos)/Optitrack.FrameRate;
figure, plot(animal.time,animal.pos), hold on, axis tight
gridxy(0,track_boundaries')
% plot(animal.time(a_inbound),track_boundaries(1)*ones(1,length(a_inbound)),'og')
% plot(animal.time(a_outbound),track_boundaries(1)*ones(1,length(a_outbound)),'or')
% plot(animal.time(b_inbound),track_boundaries(2)*ones(1,length(b_inbound)),'og')
% plot(animal.time(b_outbound),track_boundaries(2)*ones(1,length(b_outbound)),'or')
plot(animal.time(pos_norm4==1),animal.pos(pos_norm4==1),'.r')
plot(animal.time(pos_norm4==-1),animal.pos(pos_norm4==-1),'.b')
plot(cooling.cooling,[30,30],'b','linewidth',2), hold on
plot(cooling.nocooling,[30,30],'r','linewidth',2), legend({'Cooling'})
title(['Linear track (' num2str(size(pos_ab,1)) ' trials)']), xlabel('Time (s)'),ylabel('Position (cm)'), legend({'Track boundaries','Position','a -> b','a <- b'})

ts_cooling_ab2 = [];
ts_cooling_ba2 = [];
ts_nocooling_ab2 = [];
ts_nocooling_ba2 = [];
units3 = [];
loc_cooling_ab_trial = cell(size(units.ts,2),size(pos_ab,1));
loc_cooling_ba_trial = cell(size(units.ts,2),size(pos_ba,1));
loc_nocooling_ab_trial = cell(size(units.ts,2),size(pos_ab,1));
loc_nocooling_ba_trial = cell(size(units.ts,2),size(pos_ba,1));

phase_cooling_ab_trial = cell(size(units.ts,2),size(pos_ab,1));
phase_cooling_ba_trial = cell(size(units.ts,2),size(pos_ba,1));
phase_nocooling_ab_trial = cell(size(units.ts,2),size(pos_ab,1));
phase_nocooling_ba_trial = cell(size(units.ts,2),size(pos_ba,1));

speed_cooling_ab_trial = cell(size(units.ts,2),size(pos_ab,1));
speed_cooling_ba_trial = cell(size(units.ts,2),size(pos_ba,1));
speed_nocooling_ab_trial = cell(size(units.ts,2),size(pos_ab,1));
speed_nocooling_ba_trial = cell(size(units.ts,2),size(pos_ba,1));
for i = 1:size(units.ts,2)
% 	ts_cooling = [];
%     ts_nocooling = [];
%     for k = 1:size(cooling,2)
%         cooling_ts_temp = find (units.ts{i}/sr > cooling(1,k) & units.ts{i}/sr < cooling(2,k));
%         ts_cooling = [ts_cooling;cooling_ts_temp];
%     end
%     for k = 1:size(nocooling,2)
%         nocooling_ts_temp = find (units.ts{i}/sr > nocooling(1,k) & units.ts{i}/sr < nocooling(2,k));
%         ts_nocooling = [ts_nocooling;nocooling_ts_temp];
%     end
    ts_cooling_ab = [];
    ts_cooling_ba = [];
    ts_nocooling_ab = [];
    ts_nocooling_ba = [];
    for k = 1:size(cooling.cooling,2)
        for j = 1:size(pos_ba,1)
            cooling_ts_temp = find(units.speed{i} > speed_min & units.ts{i}/sr > cooling.cooling(1,k) & units.ts{i}/sr < cooling.cooling(2,k) & units.ts{i}/sr > animal.time(pos_ba(j,1)) & units.ts{i}/sr < animal.time(pos_ba(j,2)));
            ts_cooling_ba = [ts_cooling_ba,cooling_ts_temp];
            if ~isempty(cooling_ts_temp)
                loc_cooling_ba_trial{i,j} = [loc_cooling_ba_trial{i,j},units.loc{i}(cooling_ts_temp)];
                phase_cooling_ba_trial{i,j} = [phase_cooling_ba_trial{i,j},units.theta_phase{i}(cooling_ts_temp)];
                speed_cooling_ba_trial{i,j} = [speed_cooling_ba_trial{i,j},units.speed{i}(cooling_ts_temp)];
            end
        end
        for j = 1:size(pos_ab,1)
            cooling_ts_temp = find(units.speed{i} > speed_min & units.ts{i}/sr > cooling.cooling(1,k) & units.ts{i}/sr < cooling.cooling(2,k) & units.ts{i}/sr > animal.time(pos_ab(j,1)) & units.ts{i}/sr < animal.time(pos_ab(j,2)));
            ts_cooling_ab = [ts_cooling_ab,cooling_ts_temp];
            if ~isempty(cooling_ts_temp)
                loc_cooling_ab_trial{i,j} = [loc_cooling_ab_trial{i,j},units.loc{i}(cooling_ts_temp)];
                phase_cooling_ab_trial{i,j} = [phase_cooling_ab_trial{i,j},units.theta_phase{i}(cooling_ts_temp)];
                speed_cooling_ab_trial{i,j} = [speed_cooling_ab_trial{i,j},units.speed{i}(cooling_ts_temp)];
            end
        end
    end
    for k = 1:size(cooling.nocooling,2)
        for j = 1:size(pos_ba,1)
            nocooling_ts_temp = find(units.speed{i} > speed_min & units.ts{i}/sr > cooling.nocooling(1,k) & units.ts{i}/sr < cooling.nocooling(2,k) & units.ts{i}/sr > animal.time(pos_ba(j,1)) & units.ts{i}/sr < animal.time(pos_ba(j,2)));
            ts_nocooling_ba = [ts_nocooling_ba,nocooling_ts_temp];
            if ~isempty(nocooling_ts_temp)
                loc_nocooling_ba_trial{i,j} = [loc_nocooling_ba_trial{i,j},units.loc{i}(nocooling_ts_temp)];
                phase_nocooling_ba_trial{i,j} = [phase_nocooling_ba_trial{i,j},units.theta_phase{i}(nocooling_ts_temp)];
                speed_nocooling_ba_trial{i,j} = [speed_nocooling_ba_trial{i,j},units.speed{i}(nocooling_ts_temp)];
            end
        end
        for j = 1:size(pos_ab,1)
            nocooling_ts_temp = find(units.speed{i} > speed_min & units.ts{i}/sr > cooling.nocooling(1,k) & units.ts{i}/sr < cooling.nocooling(2,k) & units.ts{i}/sr > animal.time(pos_ab(j,1)) & units.ts{i}/sr < animal.time(pos_ab(j,2)));
            ts_nocooling_ab = [ts_nocooling_ab,nocooling_ts_temp];
            if ~isempty(nocooling_ts_temp)
                loc_nocooling_ab_trial{i,j} = [loc_nocooling_ab_trial{i,j},units.loc{i}(nocooling_ts_temp)];
                phase_nocooling_ab_trial{i,j} = [phase_nocooling_ab_trial{i,j},units.theta_phase{i}(nocooling_ts_temp)];
                speed_nocooling_ab_trial{i,j} = [speed_nocooling_ab_trial{i,j},units.speed{i}(nocooling_ts_temp)];
            end
        end
    end
    ts_cooling_ab2{i} = sort(ts_cooling_ab);
    ts_cooling_ba2{i} = sort(ts_cooling_ba);
    ts_nocooling_ab2{i} = sort(ts_nocooling_ab);
    ts_nocooling_ba2{i} = sort(ts_nocooling_ba);
    
    units3.ts_cooling_ab{i} = units.ts{i}(ts_cooling_ab2{i});
    units3.loc_cooling_ab{i} = units.loc{i}(ts_cooling_ab2{i});
    units3.theta_phase_cooling_ab{i} = units.theta_phase{i}(ts_cooling_ab2{i});
    units3.theta_phase_cooling_ab2{i} = units.theta_phase2{i}(ts_cooling_ab2{i});
    units3.speed_cooling_ab{i} = units.speed{i}(ts_cooling_ab2{i});
    
    units3.ts_cooling_ba{i} = units.ts{i}(ts_cooling_ba2{i});
    units3.loc_cooling_ba{i} = units.loc{i}(ts_cooling_ba2{i});
    units3.theta_phase_cooling_ba{i} = units.theta_phase{i}(ts_cooling_ba2{i});
    units3.theta_phase_cooling_ba2{i} = units.theta_phase2{i}(ts_cooling_ba2{i});
    units3.speed_cooling_ba{i} = units.speed{i}(ts_cooling_ba2{i});
    
    units3.ts_nocooling_ab{i} = units.ts{i}(ts_nocooling_ab2{i});
    units3.loc_nocooling_ab{i} = units.loc{i}(ts_nocooling_ab2{i});
    units3.theta_phase_nocooling_ab{i} = units.theta_phase{i}(ts_nocooling_ab2{i});
    units3.theta_phase_nocooling_ab2{i} = units.theta_phase2{i}(ts_nocooling_ab2{i});
    units3.speed_nocooling_ab{i} = units.speed{i}(ts_nocooling_ab2{i});
    
    units3.ts_nocooling_ba{i} = units.ts{i}(ts_nocooling_ba2{i});
    units3.loc_nocooling_ba{i} = units.loc{i}(ts_nocooling_ba2{i});
    units3.theta_phase_nocooling_ba{i} = units.theta_phase{i}(ts_nocooling_ba2{i});
    units3.theta_phase_nocooling_ba2{i} = units.theta_phase2{i}(ts_nocooling_ba2{i});
    units3.speed_nocooling_ba{i} = units.speed{i}(ts_nocooling_ba2{i});
end
nb_nocooling_ab = [];
nb_cooling_ab = [];
nb_nocooling_ba = [];
nb_cooling_ba = [];
for k = 1:size(cooling.nocooling,2)
    for j = 1:size(pos_ab,1)
        nocooling_nb_temp = find(animal.speed > speed_min & animal.time > cooling.nocooling(1,k) & animal.time < cooling.nocooling(2,k) & animal.time > animal.time(pos_ab(j,1)) & animal.time < animal.time(pos_ab(j,2)));
        if ~isempty(nocooling_nb_temp)
            nb_nocooling_ab = [nb_nocooling_ab,nocooling_nb_temp];
        end
    end
    for j = 1:size(pos_ba,1)
        nocooling_nb_temp = find(animal.speed > speed_min & animal.time > cooling.nocooling(1,k) & animal.time < cooling.nocooling(2,k) & animal.time > animal.time(pos_ba(j,1)) & animal.time < animal.time(pos_ba(j,2)));
        if ~isempty(nocooling_nb_temp)
            nb_nocooling_ba = [nb_nocooling_ba,nocooling_nb_temp];
        end
    end
end
for k = 1:size(cooling.cooling,2)
    for j = 1:size(pos_ab,1)
        cooling_nb_temp = find(animal.speed > speed_min & animal.time > cooling.cooling(1,k) & animal.time < cooling.cooling(2,k) & animal.time > animal.time(pos_ab(j,1)) & animal.time < animal.time(pos_ab(j,2)));
        if ~isempty(cooling_nb_temp)
            nb_cooling_ab = [nb_cooling_ab,cooling_nb_temp];
        end
    end
    for j = 1:size(pos_ba,1)
        cooling_nb_temp = find(animal.speed > speed_min & animal.time > cooling.cooling(1,k) & animal.time < cooling.cooling(2,k) & animal.time > animal.time(pos_ba(j,1)) & animal.time < animal.time(pos_ba(j,2)));
        if ~isempty(cooling_nb_temp)
            nb_cooling_ba = [nb_cooling_ba,cooling_nb_temp];
        end
    end
end
nb_nocooling_ab2 = length(nb_nocooling_ab)/animal.sr;
nb_nocooling_ba2 = length(nb_nocooling_ba)/animal.sr;
nb_cooling_ab2 = length(nb_cooling_ab)/animal.sr;
nb_cooling_ba2 = length(nb_cooling_ba)/animal.sr;

pos_cooling_ab = animal.pos(sort(nb_cooling_ab));
pos_cooling_ba = animal.pos(sort(nb_cooling_ba));
pos_nocooling_ab = animal.pos(sort(nb_nocooling_ab));
pos_nocooling_ba = animal.pos(sort(nb_nocooling_ba));

speed_nocooling_ab2 = animal.speed(sort(nb_nocooling_ab));
speed_nocooling_ba2 = animal.speed(sort(nb_nocooling_ba));
speed_cooling_ab2 = animal.speed(sort(nb_cooling_ab));
speed_cooling_ba2 = animal.speed(sort(nb_cooling_ba));
bin_speed = [speed_min:5:90];
hist_a = hist(speed_nocooling_ab2,bin_speed);
hist_b = hist(speed_nocooling_ba2,bin_speed);
hist_c = hist(speed_cooling_ab2,bin_speed);
hist_d = hist(speed_cooling_ba2,bin_speed);

% Speed of the animal with and without cooling
figure,
subplot(2,1,1)
plot(bin_speed,hist_a./length(speed_nocooling_ab2),'-r'), hold on
plot(bin_speed,hist_b./length(speed_nocooling_ba2),'--r'),
plot(bin_speed,hist_c./length(speed_cooling_ab2),'-b'),
plot(bin_speed,hist_d./length(speed_cooling_ba2),'--b'),axis tight
legend({'nocooling ->','nocooling <-','cooling ->','cooling <-',}),
xlabel('Speed (cm/s)'),ylabel('Probability')
subplot(2,2,3)
bar(1,sum([speed_nocooling_ab2,speed_nocooling_ba2]./animal.sr/(nb_nocooling_ab2+nb_nocooling_ba2)),'r'), hold on
bar(2,sum([speed_cooling_ab2,speed_cooling_ba2]./animal.sr/(nb_cooling_ab2+nb_cooling_ba2)),'b'),
% bar(3,sum(speed_nocooling_ba2./animal.sr/nb_nocooling_ba2),'r'),
% bar(4,sum(speed_cooling_ba2./animal.sr/nb_cooling_ba2),'b'),
title(['Speed increase: +' num2str(100*sum([speed_cooling_ab2,speed_cooling_ba2]./(nb_cooling_ab2+nb_cooling_ba2))/ sum([speed_nocooling_ab2,speed_nocooling_ba2]./(nb_nocooling_ab2+nb_nocooling_ba2))-100,3) '% '])
ylabel('Speed (cm/s)')
ax = gca; ax.XTick = ([1,2]); ax.XTickLabels = ({'No Cooling','Cooling'});
subplot(2,2,4)
bar(1,sum([speed_nocooling_ab2,speed_nocooling_ba2]./(animal.sr*sum(cooling.nocooling(2,:)-cooling.nocooling(1,:)))),'r'), hold on
bar(2,sum([speed_cooling_ab2,speed_cooling_ba2]./(animal.sr*sum(cooling.cooling(2,:)-cooling.cooling(1,:)))),'b'),
title(['Distance increase pr session time: +' num2str(100*(sum([speed_cooling_ab2,speed_cooling_ba2])./(animal.sr*sum(cooling.cooling(2,:)-cooling.cooling(1,:))))/ sum([speed_nocooling_ab2,speed_nocooling_ba2]./(animal.sr*sum(cooling.nocooling(2,:)-cooling.nocooling(1,:))))-100,3) '% '])
ylabel('Average Speed (cm/s)')
ax = gca; ax.XTick = ([1,2]); ax.XTickLabels = ({'No Cooling','Cooling'});

% Phase precession along track for each unit in both directions
drawArrow = @(x,y,varargin) quiver( x(1),y(1),x(2)-x(1),y(2)-y(1), varargin{:} ); 
for i = 1:size(units.ts,2)
    if rem(i,15) == 1
        figure
    end
    subplot(3,5,rem(i-1,15)+1)
    plot(units3.loc_cooling_ab{i},units3.theta_phase_cooling_ab{i}+pi,'.b','markersize',0.5), hold on
    plot(units3.loc_cooling_ab{i},units3.theta_phase_cooling_ab{i}+3*pi,'.b','markersize',0.5)
    plot(units3.loc_nocooling_ab{i},units3.theta_phase_nocooling_ab{i}+5*pi,'.r','markersize',1),
    plot(units3.loc_nocooling_ab{i},units3.theta_phase_nocooling_ab{i}+7*pi,'.r','markersize',1),
    plot(units3.loc_nocooling_ba{i},units3.theta_phase_nocooling_ba{i}-pi,'.r','markersize',1)
    plot(units3.loc_nocooling_ba{i},units3.theta_phase_nocooling_ba{i}-3*pi,'.r','markersize',1)
    plot(units3.loc_cooling_ba{i},units3.theta_phase_cooling_ba{i}-5*pi,'.b','markersize',0.5),
    plot(units3.loc_cooling_ba{i},units3.theta_phase_cooling_ba{i}-7*pi,'.b','markersize',0.5),
    
    plot([35,245],[0,0],'k','linewidth',0.2),
    plot([35,245],-4*pi*[1,1],'--k','linewidth',0.2),
    plot([35,245],4*pi*[1,1],'--k','linewidth',0.2),
    ax = gca;
    ax.YTick = ([-8*pi -7*pi -6*pi -5*pi -4*pi -3*pi -2*pi -pi 0 pi 2*pi 3*pi 4*pi 5*pi 6*pi 7*pi 8*pi]);
    ax.YTickLabels = ({'0','\pi','2\pi','3\pi','0','\pi','2\pi','3\pi','0','\pi','2\pi','3\pi','0','\pi','2\pi','3\pi','4\pi'});
    
    drawArrow([track_boundaries(1)-20,track_boundaries(1)-5],[6*pi,6*pi],'MaxHeadSize',5,'linewidth',1,'color','r')
    drawArrow([track_boundaries(1)-20,track_boundaries(1)-5],[2*pi,2*pi],'MaxHeadSize',5,'linewidth',1,'color','b')
    drawArrow([track_boundaries(2)+25,track_boundaries(2)+5],[-2*pi,-2*pi],'MaxHeadSize',5,'linewidth',1,'color','r')
    drawArrow([track_boundaries(2)+25,track_boundaries(2)+5],[-6*pi,-6*pi],'MaxHeadSize',5,'Color','b','LineWidth',1)
    
    hold off, axis tight, xlim([track_boundaries(1)-20,track_boundaries(2)+20]),ylim([-8*pi,8*pi]),
    title(['Unit ' num2str(i)])
end

% Average firing rate for each unit in both directions
x_hist2 = [track_boundaries(1):1:track_boundaries(2)];
counts_cooling_ab2 = hist(pos_cooling_ab,x_hist2);
counts_cooling_ba2 = hist(pos_cooling_ba,x_hist2);
counts_nocooling_ba2 = hist(pos_nocooling_ba,x_hist2);
counts_nocooling_ab2 = hist(pos_nocooling_ab,x_hist2);
for i = 1:size(units.ts,2)
     if rem(i,10) == 1
        figure
     end
    subplot(4,5,rem(i-1,10)+1)
    % subplot(14,6,i)
    counts_cooling_ab = hist(units3.loc_cooling_ab{i},x_hist2);
    plot(x_hist2,nanconv(counts_cooling_ab./counts_cooling_ab2*animal.sr,gausswin(21)'/sum(gausswin(21)),'edge'),'b'), hold on
    counts_nocooling_ab = hist(units3.loc_nocooling_ab{i},x_hist2);
    
    plot(x_hist2,nanconv(counts_nocooling_ab./counts_nocooling_ab2*animal.sr,gausswin(21)'/sum(gausswin(21)),'edge'),'r'), axis tight
    xlim([track_boundaries(1),track_boundaries(2)]),
    title(['Unit ' num2str(i)])%,ylim([0,10])
    
    subplot(4,5,rem(i-1,10)+1+10)
    %subplot(14,6,i+42)
    counts_cooling_ba = hist(units3.loc_cooling_ba{i},x_hist2);
    plot(x_hist2,nanconv(counts_cooling_ba./counts_cooling_ba2*animal.sr,gausswin(21)'/sum(gausswin(21)),'edge'),'b'), hold on
    counts_nocooling_ba = hist(units3.loc_nocooling_ba{i},x_hist2);
    
    plot(x_hist2,nanconv(counts_nocooling_ba./counts_nocooling_ba2*animal.sr,gausswin(21)'/sum(gausswin(21)),'edge'),'r'), axis tight
    xlim([track_boundaries(1),track_boundaries(2)]),
    title(['Unit ' num2str(i)])%,ylim([0,10])
end
% Spatial Coherence for each unit for cooling/noncooling and direction
bin_size = 5; % in cm;
x_hist3 = [track_boundaries(1):bin_size:track_boundaries(2)];
counts_cooling_ab2 = hist(pos_cooling_ab,x_hist3);
counts_cooling_ba2 = hist(pos_cooling_ba,x_hist3);
counts_nocooling_ab2 = hist(pos_nocooling_ab,x_hist3);
counts_nocooling_ba2 = hist(pos_nocooling_ba,x_hist3);
SpatialCoh = [];
SpatialCohP = [];
SpatialPeakFiringRate = [];
for i = 1:size(units.ts,2)
    counts = hist(units3.loc_cooling_ab{i},x_hist3)./counts_cooling_ab2*animal.sr;
    [SpatialCoh(i,1),SpatialCohP(i,1)] = SpatialCoherence(counts);
    SpatialPeakFiringRate(i,1) = max(counts);
    counts = hist(units3.loc_nocooling_ab{i},x_hist3)./counts_nocooling_ab2*animal.sr;
    [SpatialCoh(i,2),SpatialCohP(i,2)] = SpatialCoherence(counts);
    SpatialPeakFiringRate(i,2) = max(counts);
    counts = hist(units3.loc_cooling_ba{i},x_hist3)./counts_cooling_ba2*animal.sr;
    [SpatialCoh(i,3),SpatialCohP(i,3)] = SpatialCoherence(counts);
      SpatialPeakFiringRate(i,3) = max(counts);     
    counts = hist(units3.loc_nocooling_ba{i},x_hist3)./counts_nocooling_ba2*animal.sr;
    [SpatialCoh(i,4),SpatialCohP(i,4)] = SpatialCoherence(counts);
    SpatialPeakFiringRate(i,4) = max(counts);
end
figure, plot(SpatialCoh,'o'), hold on
plot([0,size(SpatialCoh,1)],[0.7 0.7],'--')
legend({'AB Cooling','AB NoCooling','BA Cooling','BA NoCooling'})
units_placecells_AB = find(sum(SpatialCoh(:,[1,2]) > 0.7,2)>1)'
units_placecells_BA = find(sum(SpatialCoh(:,[3,4]) > 0.7,2)>1)'
plot(units_placecells_AB,SpatialCoh(units_placecells_AB,[1,2]),'.k')
plot(units_placecells_BA,SpatialCoh(units_placecells_BA,[3,4]),'.k')
title('Spatial Coherence'), xlabel('Units'), ylabel('Coherence')

%% % Autocorrelation of the spike times
lag = 200;
figure
for i = 1:size(units.ts,2)
    ts_binned = zeros(1,ceil(units.ts{i}(end)/20));
    ts_binned(floor(units.ts{i}/20))=1;
    r1 = xcorr(ts_binned,lag)/length(units.ts{i});
    r1(lag+1) = 0;
    subplot(6,6,i)
    stairs([-lag:lag],r1), axis tight
    title(['Unit ' num2str(i)])
end

%% % Circular-linear regression based on Schmidt Comp Neuro 2012 (and Schmidt JNeurosci 2009)
% https://www.mathworks.com/matlabcentral/newsreader/view_thread/303398
phase_units = [4,8,10,15,26,27,27,28,28,30,32,32,32,33]; % recordings(id).phase_units
phase_units_direction = [2,1,2,1,2,1,2,1,2,2,1,1,2,1]; % recordings(id).phase_units_direction
phase_units_pos_limits = [145,200; 60,140; 65,160; 45,175; 100,145; 100,170; 65,120; 55,115; 60,120; 70,170;45,120; 175,220; 45,120;140,220]-150; % recordings(id).phase_units_pos_limits

slope1 = []; offset1 = []; slope2 = []; offset2 = []; R1 = []; R2 = []; 
for j = 1:length(phase_units)
    i = phase_units(j);
    if phase_units_direction(j) == 1
        circ1 = units3.theta_phase_nocooling_ab{i}; % Circular data
        lin1 = units3.loc_nocooling_ab{i}-track_boundaries(1); % Linear data
        circ2 = units3.theta_phase_cooling_ab{i}(); % Circular data
        lin2 = units3.loc_cooling_ab{i}-track_boundaries(1); % Linear data
    else
        circ1 = units3.theta_phase_nocooling_ba{i}; % Circular data
        lin1 = units3.loc_nocooling_ba{i}-track_boundaries(1); % Linear data
        circ2 = units3.theta_phase_cooling_ba{i}(); % Circular data
        lin2 = units3.loc_cooling_ba{i}-track_boundaries(1); % Linear data
    end
    
    indexes1 = find(lin1 > phase_units_pos_limits(j,1) & lin1 < phase_units_pos_limits(j,2));
    [slope1(j),offset1(j),R1(j)] = CircularLinearRegression(circ1(indexes1),lin1(indexes1));
    indexes2 = find(lin2 > phase_units_pos_limits(j,1) & lin2 < phase_units_pos_limits(j,2));
    [slope2(j),offset2(j),R2(j)] = CircularLinearRegression(circ2(indexes2),lin2(indexes2));

    x = phase_units_pos_limits(j,1):1:phase_units_pos_limits(j,2);
    while  2*pi*slope1(j)*x(round(length(x)./2))+offset1(j) < 0; offset1(j) = offset1(j)+2*pi; end
    while 2*pi*slope1(j)*x(round(length(x)./2))+offset1(j) > 2*pi; offset1(j) = offset1(j)-2*pi;  end
    while  2*pi*slope2(j)*x(round(length(x)./2))+offset2(j) < 0; offset2(j) = offset2(j)+2*pi; end
    while 2*pi*slope2(j)*x(round(length(x)./2))+offset2(j) > 2*pi; offset2(j) = offset2(j)-2*pi; end
    if rem(j,3) == 1
        figure
    end
    subplot(2,3,rem(j-1,3)+1)
    % figure
    %subplot(2,5,1)
    plot(lin1,circ1,'.k'),hold on, 
    plot(lin1,circ1 + 2*pi,'.k'),
    plot(lin1(indexes1),circ1(indexes1),'.r'),hold on, 
    plot(lin1(indexes1),circ1(indexes1)+2*pi,'.r'),
    plot(x,2*pi*slope1(j)*x+offset1(j),'k-','linewidth',1.5)
    xlabel('Position'),ylabel('Phase precession'),title('Circular-Linear Regression'), axis tight
    xlim([0,track_boundaries(2)-track_boundaries(1)])
    title(['Unit: ' num2str(i) ', Slope: ' num2str(slope1(j))])
    ax = gca;
    ax.YTick = ([-4*pi -3*pi -2*pi -pi 0 pi 2*pi 3*pi 4*pi]);
    ax.YTickLabels = ({'-4\pi','-3\pi','-2\pi','-\pi','0','\pi','2\pi','3\pi','4\pi'});
    gridxy([phase_units_pos_limits(j,1) phase_units_pos_limits(j,2)],'color',[0.9 0.9 0.9])
    
    subplot(2,3,rem(j-1,3)+1+3)
%    subplot(2,5,2)
    plot(lin2,circ2,'.k'),hold on,
    plot(lin2,circ2 + 2*pi,'.k'),
    plot(lin2(indexes2),circ2(indexes2),'.b'),hold on, 
    plot(lin2(indexes2),circ2(indexes2)+2*pi,'.b'),
    plot(x,2*pi*slope2(j)*x+offset2(j),'k-','linewidth',1.5)
    xlabel('Position'),ylabel('Phase precession'),title('Circular-Linear Regression'), axis tight
    xlim([0,track_boundaries(2)-track_boundaries(1)])
    title(['Unit: ' num2str(i) ', Slope: ' num2str(slope2(j))])
    ax = gca;
    ax.YTick = ([-4*pi -3*pi -2*pi -pi 0 pi 2*pi 3*pi 4*pi]);
    ax.YTickLabels = ({'-4\pi','-3\pi','-2\pi','-\pi','0','\pi','2\pi','3\pi','4\pi'});
    gridxy([phase_units_pos_limits(j,1) phase_units_pos_limits(j,2)],'color',[0.9 0.9 0.9])
end
figure, subplot(2,1,1)
plot(abs(slope1),abs(slope2),'.'), hold on
plot([0,0.03],[0,0.03])
xlabel('No Cooling'),ylabel('Cooling'),title('Circular-Linear Regression')
subplot(2,1,2)
hist(slope2./slope1,20)
xlabel('Ratio'),ylabel('Regression'),title('Circular-Linear Regression')

%% % Autocorrelation of the phase precession
bins_2pi = 100;
lag = 200;
r_max = [];
r_max_i = [];
r_min = [];
r_min_i = [];

figure
for i = 1:size(units3.theta_phase_cooling_ab,2)
    cooling_ab = units3.theta_phase_cooling_ab2{i};
    cooling_ba = units3.theta_phase_cooling_ba2{i};
    if length(cooling_ab) > 1 & length(cooling_ba) > 1
        ts_binned1 = zeros(1,ceil(cooling_ab(end)/(2*pi)*bins_2pi));
        ts_binned1(round(cooling_ab/(2*pi)*bins_2pi))=1;
        ts_binned2 = zeros(1,ceil(cooling_ba(end)/(2*pi)*bins_2pi));
        ts_binned2(round(cooling_ba/(2*pi)*bins_2pi))=1;
        r1 = xcorr([ts_binned1, zeros(1,lag), ts_binned2],lag)/sum([ts_binned1,ts_binned2]);
        r1(lag+1) = 0;
        r1 = nanconv(r1,gausswin(51)'/sum(gausswin(51)),'edge');
    else
        r1 = zeros(1,lag*2+1);
    end
    %
    cooling_ab = units3.theta_phase_nocooling_ab2{i};
    cooling_ba = units3.theta_phase_nocooling_ba2{i};
    if length(cooling_ab) > 1 & length(cooling_ba) > 1
        ts_binned1 = zeros(1,ceil(cooling_ab(end)/(2*pi)*bins_2pi));
        ts_binned1(round(cooling_ab/(2*pi)*bins_2pi))=1;
        ts_binned2 = zeros(1,ceil(cooling_ba(end)/(2*pi)*bins_2pi));
        ts_binned2(round(cooling_ba/(2*pi)*bins_2pi))=1;
        r2 = xcorr([ts_binned1, zeros(1,lag), ts_binned2],lag)/sum([ts_binned1,ts_binned2]);
        r2(lag+1) = 0;
        r2 = nanconv(r2,gausswin(51)'/sum(gausswin(51)),'edge');
    else
        r2 = zeros(1,lag*2+1);
    end
    [r_max(1,i),r_max_i(1,i)] = max(r1(lag+1+0.5*bins_2pi:lag+1+1.5*bins_2pi));
    [r_min(1,i),r_min_i(1,i)] = min(r1(lag+1:lag+1+1*bins_2pi));
    
    [r_max(2,i),r_max_i(2,i)] = max(r2(lag+1+0.5*bins_2pi:lag+1+1.5*bins_2pi));
    [r_min(2,i),r_min_i(2,i)] = min(r2(lag+1:lag+1+1*bins_2pi));
    subplot(6,6,i)
    plot((-lag:lag)/100*2*pi,[r1;r2]),
    title(['Unit ' num2str(i)]), axis tight
    ax = gca;
    ax.XTick = ([-4*pi -3*pi -2*pi -pi 0 pi 2*pi 3*pi 4*pi]);
    ax.XTickLabels = ({'-4\pi','-3\pi','-2\pi','-\pi','0','\pi','2\pi','3\pi','4\pi'});
    gridxy([-2*pi ,0, 2*pi],'color',[0.9 0.9 0.9])
end
legend({'Cooling','No Cooling'})
r_max_i(r_max_i==1) = 0;
r_max_i = r_max_i+0.5*bins_2pi;
indexs = find(sum(r_max_i==0.5*bins_2pi)==0);
figure
subplot(3,2,1), hist([r_max_i(1,indexs);r_max_i(2,indexs)]',[75:2:125]), title('Theta precession time change by cooling'), xlabel('Theta modulation')
ax = gca; ax.XTick = ([75 100 125]); ax.XTickLabels = ({'3/2\pi','2\pi','5/2\pi'});xlim([75,125]),legend({'Cooling','No Cooling'})
subplot(3,2,2), hist([(r_max(1,indexs)./r_min(1,indexs));(r_max(2,indexs)./r_min(2,indexs))]',[1:0.1:5]), title('Theta modulation'), xlabel('Ratio'),xlim([0.5,2.5])
subplot(3,2,3), hist(r_max_i(1,indexs)./r_max_i(2,indexs),20), title('Theta precession time change by cooling'), xlabel('Ratio')
subplot(3,2,4), hist((r_max(1,indexs)./r_min(1,indexs))./(r_max(2,indexs)./r_min(2,indexs)),[0.5:0.1:2.5]), title('Theta modulation'), xlabel('Ratio'),xlim([0.5,2.5])
subplot(3,1,3), plot(r_max_i(1,indexs)./r_max_i(2,indexs),(r_max(1,indexs)./r_min(1,indexs))./(r_max(2,indexs)./r_min(2,indexs)),'.')
xlabel('Theta precession time change by cooling'),ylabel('Theta modulation')

% Only for the units defined to phase precess
r_max = [];
r_max_i = [];
r_min = [];
r_min_i = [];
figure
for j = 1:length(phase_units)
    i = phase_units(j);
    if phase_units_direction(j) == 1
        lin1 = units3.loc_cooling_ab{i}-track_boundaries(1); % Linear data
        lin2 = units3.loc_nocooling_ab{i}-track_boundaries(1); % Linear data
        indexes1 = find(lin1 > phase_units_pos_limits(j,1) & lin1 < phase_units_pos_limits(j,2));
        indexes2 = find(lin2 > phase_units_pos_limits(j,1) & lin2 < phase_units_pos_limits(j,2));
        circ1 = units3.theta_phase_cooling_ab2{i}(indexes1); % Circular data
        circ2 = units3.theta_phase_nocooling_ab2{i}(indexes2); % Circular data
    else
        lin1 = units3.loc_cooling_ba{i}-track_boundaries(1); % Linear data
        lin2 = units3.loc_nocooling_ba{i}-track_boundaries(1); % Linear data
        indexes1 = find(lin1 > phase_units_pos_limits(j,1) & lin1 < phase_units_pos_limits(j,2));
        indexes2 = find(lin2 > phase_units_pos_limits(j,1) & lin2 < phase_units_pos_limits(j,2));
        circ1 = units3.theta_phase_cooling_ba2{i}(indexes1); % Circular data
        circ2 = units3.theta_phase_nocooling_ba2{i}(indexes2); % Circular data
    end
    
    if length(circ1) > 1
        ts_binned1 = zeros(1,ceil(circ1(end)/(2*pi)*bins_2pi));
        ts_binned1(round(circ1/(2*pi)*bins_2pi))=1;
        r1 = xcorr(ts_binned1,lag)/sum(ts_binned1);
        r1(lag+1) = 0;
        r1 = nanconv(r1,gausswin(51)'/sum(gausswin(51)),'edge');
    else
        r1 = zeros(1,lag*2+1);
    end
    
    if length(circ2) > 1
        ts_binned1 = zeros(1,ceil(circ2(end)/(2*pi)*bins_2pi));
        ts_binned1(round(circ2/(2*pi)*bins_2pi))=1;
        r2 = xcorr(ts_binned1,lag)/sum(ts_binned1);
        r2(lag+1) = 0;
        r2 = nanconv(r2,gausswin(51)'/sum(gausswin(51)),'edge');
    else
        r2 = zeros(1,lag*2+1);
    end
    [r_max(1,j),r_max_i(1,j)] = max(r1(lag+1+0.5*bins_2pi:lag+1+1.5*bins_2pi));
    [r_min(1,j),r_min_i(1,j)] = min(r1(lag+1:lag+1+1*bins_2pi));
    
    [r_max(2,j),r_max_i(2,j)] = max(r2(lag+1+0.5*bins_2pi:lag+1+1.5*bins_2pi));
    [r_min(2,j),r_min_i(2,j)] = min(r2(lag+1:lag+1+1*bins_2pi));
    
    subplot(4,4,j)
    plot((-lag:lag)/100*2*pi,[r1;r2]),
    title(['Unit ' num2str(i)]), axis tight
    ax = gca;
    ax.XTick = ([-4*pi -3*pi -2*pi -pi 0 pi 2*pi 3*pi 4*pi]);
    ax.XTickLabels = ({'-4\pi','-3\pi','-2\pi','-\pi','0','\pi','2\pi','3\pi','4\pi'});
    gridxy([-2*pi ,0, 2*pi],'color',[0.9 0.9 0.9])
end
legend({'Cooling','No Cooling'})
r_max_i(r_max_i==1) = 0;
r_max_i = r_max_i+0.5*bins_2pi;
indexs = find(sum(r_max_i==0.5*bins_2pi)==0);
figure
subplot(3,2,1), hist([r_max_i(1,indexs);r_max_i(2,indexs)]',[75:2:125]), title('Theta precession time change by cooling'), xlabel('Theta modulation')
ax = gca; ax.XTick = ([75 100 125]); ax.XTickLabels = ({'3/2\pi','2\pi','5/2\pi'});xlim([75,125]),legend({'Cooling','No Cooling'})
subplot(3,2,2), hist([(r_max(1,indexs)./r_min(1,indexs));(r_max(2,indexs)./r_min(2,indexs))]',[1:0.1:5]), title('Theta modulation'), xlabel('Ratio'),xlim([0.5,2.5])
subplot(3,2,3), hist(r_max_i(1,indexs)./r_max_i(2,indexs),20), title('Theta precession time change by cooling'), xlabel('Ratio')
subplot(3,2,4), hist((r_max(1,indexs)./r_min(1,indexs))./(r_max(2,indexs)./r_min(2,indexs)),[0.5:0.1:2.5]), title('Theta modulation'), xlabel('Ratio'),xlim([0.5,2.5])
subplot(3,1,3), plot(r_max_i(1,indexs)./r_max_i(2,indexs),(r_max(1,indexs)./r_min(1,indexs))./(r_max(2,indexs)./r_min(2,indexs)),'*')
xlabel('Theta precession time change by cooling'),ylabel('Theta modulation')

%% % Plotting the spike location trialwise
trial_nb_spikes_cooling_ab = [];
trial_nb_spikes_cooling_ba = [];
trial_nb_spikes_nocooling_ab = [];
trial_nb_spikes_nocooling_ba = [];
for i = 1:size(units.ts,2)
    if rem(i,12) == 1
        figure
    end 
    subplot(3,4,rem(i-1,12)+1)
    for j = 1:size(pos_ab,1)
        plot(loc_cooling_ab_trial{i,j},size(pos_ba,1)+2+j*ones(1,length(loc_cooling_ab_trial{i,j})),'.b'), hold on
        plot(loc_nocooling_ab_trial{i,j},size(pos_ba,1)+2+j*ones(1,length(loc_nocooling_ab_trial{i,j})),'.r'), hold on
        trial_nb_spikes_cooling_ab(i,j) = length(loc_cooling_ab_trial{i,j});
        trial_nb_spikes_nocooling_ab(i,j) = length(loc_nocooling_ab_trial{i,j});
    end
    plot([track_boundaries(1),track_boundaries(2)],[size(pos_ab,1),size(pos_ab,1)]+2.5);
    for j = 1:size(pos_ba,1)
        plot(loc_cooling_ba_trial{i,j},j*ones(1,length(loc_cooling_ba_trial{i,j})),'.b'), hold on
        plot(loc_nocooling_ba_trial{i,j},j*ones(1,length(loc_nocooling_ba_trial{i,j})),'.r'), hold on
        trial_nb_spikes_cooling_ba(i,j) = length(loc_cooling_ba_trial{i,j});
        trial_nb_spikes_nocooling_ba(i,j) = length(loc_nocooling_ba_trial{i,j});
    end
    xlim([track_boundaries(1),track_boundaries(2)])
    title(['Unit ' num2str(i)])
    if rem(i,4)==1
        ylabel('Trial')
    end
    if i>2*3
        xlabel('Position (cm)')
    end
end

figure, 
plot(trial_nb_spikes_cooling_ab','b'), hold on, plot(trial_nb_spikes_cooling_ba','b')
plot(trial_nb_spikes_nocooling_ab','r'), hold on, plot(trial_nb_spikes_nocooling_ba','r')

pos_bins = (track_boundaries(1):track_boundaries(2));

pos_ab_cool = zeros(1,size(pos_ab,1));
pos_ba_cool = zeros(1,size(pos_ba,1));
for k = 1:size(cooling.cooling,2)
        temp = find( animal.time(pos_ab(:,1)) > cooling.cooling(1,k) & animal.time(pos_ab(:,2)) > cooling.cooling(1,k) & animal.time(pos_ab(:,1)) < cooling.cooling(2,k) & animal.time(pos_ab(:,2)) < cooling.cooling(2,k) );
        pos_ab_cool(temp) = 1;
        temp = find( animal.time(pos_ba(:,1)) > cooling.cooling(1,k) & animal.time(pos_ba(:,2)) > cooling.cooling(1,k) & animal.time(pos_ba(:,1)) < cooling.cooling(2,k) & animal.time(pos_ba(:,2)) < cooling.cooling(2,k) );
        pos_ba_cool(temp) = 1;
end
for k = 1:size(cooling.nocooling,2)
        temp = find( animal.time(pos_ab(:,1)) > cooling.nocooling(1,k) & animal.time(pos_ab(:,2)) > cooling.nocooling(1,k) & animal.time(pos_ab(:,1)) < cooling.nocooling(2,k) & animal.time(pos_ab(:,2)) < cooling.nocooling(2,k) );
        pos_ab_cool(temp) = 2;
        temp = find( animal.time(pos_ba(:,1)) > cooling.nocooling(1,k) & animal.time(pos_ba(:,2)) > cooling.nocooling(1,k) & animal.time(pos_ba(:,1)) < cooling.nocooling(2,k) & animal.time(pos_ba(:,2)) < cooling.nocooling(2,k) );
        pos_ba_cool(temp) = 2;
end

pos_hist_ab = [];
pos_hist_ba = [];
corr_matrix = [];
for i = 1:size(units.ts,2)
%     i = phase_units(k);
    if rem(i,12) == 1
        figure
    end
    subplot(3,4,rem(i-1,12)+1)
    for j = 1:size(pos_ab,1)
        if pos_ab_cool(j) == 1
            pos_hist_ab(j,:) = nanconv(hist(loc_cooling_ab_trial{i,j},pos_bins),gausswin(51)'/sum(gausswin(51)),'edge');
        elseif pos_ab_cool(j) == 2
            pos_hist_ab(j,:) = nanconv(hist(loc_nocooling_ab_trial{i,j},pos_bins),gausswin(51)'/sum(gausswin(51)),'edge');
        end
    end
    for j = 1:size(pos_ba,1)
        if pos_ba_cool(j) == 1
            pos_hist_ba(j,:) = nanconv(hist(loc_cooling_ba_trial{i,j},pos_bins),gausswin(51)'/sum(gausswin(51)),'edge');
        elseif pos_ba_cool(j) == 2
            pos_hist_ba(j,:) = nanconv(hist(loc_nocooling_ba_trial{i,j},pos_bins),gausswin(51)'/sum(gausswin(51)),'edge');
        end
        end2
    corr_matrix = corr([pos_hist_ba',pos_hist_ab']);
    %figure(104), 
    subplot(3,4,rem(i-1,12)+1)
    imagesc(corr_matrix),set(gca,'YDir','normal'), hold on
    plot([find(diff(pos_ba_cool)~=0);find(diff(pos_ba_cool)~=0)]+0.5,[0,size(corr_matrix,1)],'-k')
    plot([0,size(corr_matrix,1)],[find(diff(pos_ba_cool)~=0);find(diff(pos_ba_cool)~=0)]+0.5,'-k')
    plot([find(diff(pos_ab_cool)~=0);find(diff(pos_ab_cool)~=0)]+0.5+size(pos_ba,1),[0,size(corr_matrix,1)],'-k')
    plot([0,size(corr_matrix,1)],[find(diff(pos_ab_cool)~=0);find(diff(pos_ab_cool)~=0)]+0.5+size(pos_ba,1),'-k')
    
    plot([0,size(corr_matrix,1)],[0,0]+0.5+size(pos_ba,1),'w')
    plot([0,0]+0.5+size(pos_ba,1),[0,size(corr_matrix,1)],'w')
        title(['Unit ' num2str(i)])
    if rem(i,4)==1
        ylabel('Trial')
    end
    if i>2*4
        xlabel('Trial')
    end
end
