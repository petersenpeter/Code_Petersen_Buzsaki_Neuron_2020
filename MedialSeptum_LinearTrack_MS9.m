% Medial Septum Cooling project
clear all
MedialSeptum_Recordings
id = 124;

recording = recordings(id);
if ~isempty(recording.dataroot)
    datapath = recording.dataroot;
end
cd([datapath, recording.name(1:6) recording.animal_id '\' recording.name, '\'])

Intan_rec_info = read_Intan_RHD2000_file_Peter([datapath, recording.name(1:6) recording.animal_id '\' recording.name, '\']);
fname = [recording.name '.dat'];
nbChan = size(Intan_rec_info.amplifier_channels,2);
cooling = recordings(id).cooling;
cooling_onsets = recordings(id).cooling_onsets;
cooling_offsets = recordings(id).cooling_offsets;
sr = Intan_rec_info.frequency_parameters.amplifier_sample_rate;
ch_theta = recordings(id).ch_theta;
ch_lfp = recordings(id).ch_lfp;
ch_medialseptum = recordings(id).ch_medialseptum;
ch_hippocampus = recordings(id).ch_hippocampus;
time_frame = recordings(id).time_frame;
lfp_periods = 30*60; % in seconds
ch_wheel_pos = recordings(id).ch_wheel_pos; % Wheel channel (base 1)
ch_temp = recordings(id).ch_temp; % Temperature data included (base 1)
ch_peltier_voltage = recordings(id).ch_peltier; % Peltier channel (base 1)
ch_fan_speed = recordings(id).ch_fan_speed; % Fan speed channel (base 1)
ch_camera_sync = recordings(id).ch_camera_sync;
ch_OptiTrack_sync = recordings(id).ch_OptiTrack_sync;
ch_CoolingPulses = recordings(id).ch_CoolingPulses;
track_boundaries = recordings(id).track_boundaries;
arena = recordings(id).arena;
nb_lfp_periods = ceil((time_frame(2)-time_frame(1))/lfp_periods);
temp_ = dir(fname);
recording_length = round(temp_.bytes/sr/nbChan/2)-1;

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

% Loading digital inputs
disp('Loading digital inputs')
if ~exist('digitalchannels.mat')
    [digital_on,digital_off] = Process_IntanDigitalChannels('digitalin.dat');
    save('digitalchannels.mat','digital_on','digital_off')
else load('digitalchannels.mat'); end
if ch_camera_sync ~= 0
    camera_onset = min(digital_on{ch_camera_sync}(1),digital_off{ch_camera_sync}(1))/sr;
    camera_offset = (temp_.bytes/sr/nbChan/2)-max(digital_on{ch_camera_sync}(end),digital_off{ch_camera_sync}(end))/sr;
end
if ch_OptiTrack_sync ~= 0
    Optitrack_onset = min(digital_on{ch_OptiTrack_sync}(1),digital_off{ch_OptiTrack_sync}(1))/sr;
    Optitrack_offset = (temp_.bytes/sr/nbChan/2)-max(digital_on{ch_OptiTrack_sync}(end),digital_off{ch_OptiTrack_sync}(end))/sr;
end
if ch_CoolingPulses ~= 0
    cooling_onsets = digital_on{ch_CoolingPulses}/sr;
end

animal = [];
animal.sr = Optitrack.FrameRate; % 100
% animal.position = nanconv([zeros(1,round(camera_onset*animal.sr)), position1D, zeros(1,round(camera_offset*animal.sr))],gausswin(round(0.2*animal.sr))'/sum(gausswin(round(0.2*animal.sr))),'edge');
% animal.speed = nanconv([zeros(1,round(camera_onset*animal.sr)), animal_speed3D, zeros(1,round(camera_offset*animal.sr))],gausswin(round(0.2*animal.sr))'/sum(gausswin(round(0.2*animal.sr))),'edge');
animal.pos = 150+nanconv(Optitrack.position1D,gausswin(round(0.41*animal.sr))'/sum(gausswin(round(0.41*animal.sr))),'edge');
animal.speed = nanconv(Optitrack.animal_speed1D,gausswin(round(0.41*animal.sr))'/sum(gausswin(round(0.41*animal.sr))),'edge');
animal.time = digital_on{ch_OptiTrack_sync}'/sr;

% Loading Temperature data
disp('Loading Temperature data')
if ch_temp >0
    if ~exist('temperature.mat')
        num_channels = length(Intan_rec_info.board_adc_channels); % ADC input info from header file
        fileinfo = dir('analogin.dat');
        num_samples = fileinfo.bytes/(num_channels * 2); % uint16 = 2 bytes
        fid = fopen('analogin.dat', 'r'); v = fread(fid, [num_channels, num_samples], 'uint16'); fclose(fid);
        v = v * 0.000050354; % convert to volts
                v_downsample = resample(v(inputs.ch_temp,:),sr_lfp, sr);
        %v_downsample(v_downsample<1.25) = 1.25;
        temperature.temp = nanconv((v_downsample-1.25)/0.005,gausswin(100)','edge');
        temperature.sr = sr_lfp;
        temperature.time = [1:length(temperature.temp)]/sr_lfp;
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

cooling = [cooling_onsets'+30;cooling_offsets'];
nocooling = [[1,cooling_offsets'+30];[cooling_onsets'-30,recording_length]];
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
plot(cooling,[0,0],'b','linewidth',2), hold on
plot(nocooling,[0,0],'r','linewidth',2), 
legend({'Position','Cooling'})
subplot(3,1,2)
plot(animal.time,animal.speed),axis tight, title('Speed')
subplot(3,1,3)
plot([1:length(temperature)]/sr_lfp,temperature,'r'),axis tight, title('Temperature')

%% Correlating the speed of the wheel and temperature with the theta power and frequency
theta_channel = ch_lfp; % 31;
% disp('Creating EEG file')
% downsample_dat_to_eeg([datapath, recording,'/']);
if ~exist([recording, '.eeg'])
    disp('Creating EEG file')
    downsample_dat_to_eeg([pwd '\']);
    movefile('amplifier.eeg',[recording '.eeg'])
    copyfile('amplifier.xml',[recording '.xml'])
end
bins_speed = [5:3:80];
bins_speed_avg = mean([bins_speed(2:end);bins_speed(1:end-1)]);
freqlist = [5.:0.025:10]; %10.^(0.4771:0.01:1.1761);
%freqlist = [4:1:150];

signal = 0.000050354 * double(LoadBinary([recording '.eeg'],'nChannels',nbChan,'channels',theta_channel,'precision','int16','frequency',sr/16)); % ,'start',start,'duration',duration
Fs = sr/16;
signal2 = resample(signal,sr_lfp,Fs);
Fpass = [1,49];
Wn_theta = [Fpass(1)/(sr_lfp/2) Fpass(2)/(sr_lfp/2)]; % normalized by the nyquist frequency
[btheta,atheta] = butter(3,Wn_theta);
signal_filtered = filtfilt(btheta,atheta,signal2);
%[wt,~,~] = awt_freqlist(signal_filtered,sr_temperature,freqlist);
%wt2 = abs(wt)'; clear wt
wt = spectrogram(signal_filtered,100,99,freqlist,sr_lfp);
wt2 = [zeros(length(freqlist),49),abs(wt), zeros(length(freqlist),50)]; clear wt

speed = interp1(animal.time,animal.speed,(1:length(signal_filtered))/sr_lfp);
pos = interp1(animal.time,animal.pos,(1:length(signal_filtered))/sr_lfp);

t_cooling = zeros(1,length(signal_filtered)); t_nocooling = zeros(1,length(signal_filtered));
for i = 1:size(cooling,2), t_cooling(cooling(1,i)*sr_lfp:cooling(2,i)*sr_lfp) = 1; end
for i = 1:size(nocooling,2), t_nocooling(nocooling(1,i)*sr_lfp:nocooling(2,i)*sr_lfp) = 1; end

wt_cooling = [];
wt_speed_cooling = [];
thetamax_cooling = [];
theta_maxpower_cooling = [];
for j = 1:length(bins_speed)-1
    indexes = find(speed > bins_speed(j) & speed < bins_speed(j+1) & t_cooling > 0); % & pos > track_boundaries(1) & pos > track_boundaries(2)
    wt_cooling{j} = wt2(:,indexes);
    wt_speed_cooling(j,:) = mean(wt2(:,indexes),2);
    [theta_maxpower_cooling(j),thetamax_cooling(j)] = max(mean(wt2(:,indexes),2));
end

wt_nocooling = [];
wt_speed_nocooling = [];
thetamax_nocooling = [];
theta_maxpower_nocooling = [];
for j = 1:length(bins_speed)-1
    indexes = find(speed > bins_speed(j) & speed < bins_speed(j+1) & t_nocooling > 0 ); % & pos > track_boundaries(1) & pos > track_boundaries(2)
    wt_nocooling{j} = wt2(:,indexes);
    wt_speed_nocooling(j,:) = mean(wt2(:,indexes),2);
    [theta_maxpower_nocooling(j),thetamax_nocooling(j)] = max(mean(wt2(:,indexes),2));
end

figure
subplot(2,2,1)
% surf(bins_lfp,bins_speed_avg,lfp_wavelets_combined,'EdgeColor','None'), view(0,90)
imagesc(bins_speed_avg,freqlist,wt_speed_cooling'), set(gca,'Ydir','normal')
axis tight, title('With Cooling'), ylabel('Powerspectrum (Hz)'),xlabel('Speed (cm/s)')
set(gca,'YTick',bins_speed), set(gca,'YTick',4:11), %set(gca,'xscale','log')
colorbar
subplot(2,2,2)
% surf(bins_lfp,bins_speed_avg,lfp_wavelets_combined,'EdgeColor','None'), view(0,90)
imagesc(bins_speed_avg,freqlist,wt_speed_nocooling'), set(gca,'Ydir','normal')
axis tight, title(['Without Cooling']), ylabel('Powerspectrum (Hz)'),xlabel('Speed (cm/s)')
set(gca,'YTick',bins_speed), set(gca,'YTick',4:11), %set(gca,'xscale','log')
colorbar
subplot(2,3,4)
stairs(freqlist,mean(wt_speed_cooling),'b'), hold on
stairs(freqlist,mean(wt_speed_nocooling),'r')
xlabel('Frequency (Hz)'), grid on, title('Average frequency'),xlim([5,10]),ylabel('Power'),ylim([0,max(mean(wt_speed_nocooling))])

subplot(2,3,5)
time = bins_speed_avg(find(thetamax_cooling>2));
y1 = freqlist(thetamax_cooling(find(thetamax_cooling>2)));
plot(time,y1,'ob'), hold on
P = polyfit(time,y1,1);
yfit = P(1)*bins_speed+P(2);
plot(bins_speed,yfit,'b-');
text(bins_speed(4),y1(4)-0.1,['Cooling slope: ', num2str(P(1),3)],'Color','blue')
[r,p] = corr(time',y1')

time = bins_speed_avg(find(thetamax_nocooling>2));
y2 = freqlist(thetamax_nocooling(find(thetamax_nocooling>2)));
plot(time,y2,'or'), hold on
P = polyfit(time,y2,1);
yfit = P(1)*bins_speed+P(2);
plot(bins_speed,yfit,'r-');
text(bins_speed(4),y2(15)+0.1,['No Cooling slope: ', num2str(P(1),3)],'Color','red')
[r,p] = corr(time',y2')
xlim([bins_speed(1),bins_speed(end)]), ylim([y1(1)-0.4,y2(end)+0.4]),
xlabel('Speed (cm/s)'),ylabel('Frequency (Hz)'), grid on, title('Peak frequency of theta')

% Power
subplot(2,3,6)
time = bins_speed_avg(find(thetamax_cooling>2));
y1 = theta_maxpower_cooling(find(thetamax_cooling>2));
plot(time,y1,'ob'), hold on
P = polyfit(time,y1,1);
yfit = P(1)*bins_speed+P(2);
plot(bins_speed,yfit,'b-');
text(bins_speed(4),y1(4)-0.1,['Cooling slope: ', num2str(P(1),3)],'Color','blue')
[r,p] = corr(time',y1')

time = bins_speed_avg(find(thetamax_nocooling>2));
y2 = theta_maxpower_nocooling(find(thetamax_nocooling>2));
plot(time,y2,'or'), hold on
P = polyfit(time,y2,1);
yfit = P(1)*bins_speed+P(2);
plot(bins_speed,yfit,'r-');
text(bins_speed(4),y1(15)+0.25,['No Cooling slope: ', num2str(P(1),3)],'Color','red')
[r,p] = corr(time',y2')
xlim([bins_speed(1),bins_speed(end)]), ylim([y1(1)-0.4,y2(end)+0.4]),
xlabel('Speed (cm/s)'),ylabel('Peak Power'), grid on, title('Amplitude theta'),xlim([bins_speed(1),bins_speed(end)])
% GLM fit
running_times = find(speed>5);
X = [animal.temperature(running_times)+37+60; speed(running_times);(animal.temperature(running_times)++37+60).*speed(running_times); [diff(speed(running_times)),0]]';
[temp_theta_maxpower,temp_theta_max] = max(wt2(:,(running_times)));
y1 = freqlist(temp_theta_max)'; % Frequency
y2 = temp_theta_maxpower'; % Power
mdl1 = fitglm(X,y1)
mdl2 = fitglm(X,y2)
theta_bins = [4:0.1:10];
hist_theta = hist2d([y1';(mdl1.predict)']',theta_bins,theta_bins);
figure, imagesc(theta_bins,theta_bins,hist_theta), xlabel('Real theta'), ylabel('Predicted theta')

%% % Counting theta cycles pr running trial
theta_channel = ch_lfp;
signal = 0.000050354 * double(LoadBinary([recording '.eeg'],'nChannels',nbChan,'channels',theta_channel,'precision','int16','frequency',sr/16)); % ,'start',start,'duration',duration
Fs = sr/16;
Fpass = [4,11];
Wn_theta = [Fpass(1)/(Fs/2) Fpass(2)/(Fs/2)]; % normalized by the nyquist frequency
[btheta,atheta] = butter(3,Wn_theta);
signal_filtered = filtfilt(btheta,atheta,signal);
signal_phase = atan2(imag(hilbert(signal_filtered)), signal_filtered);
signal_phase2 = unwrap(signal_phase);
clear signal signal_filtered signal_phase
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
            pos_ab = [pos_ab;a_outbound(i),a_outbound(i)+next(1)];
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
            pos_ba = [pos_ba;b_outbound(i),b_outbound(i)+next(1)];
        end
    end
end

theta_cycles_ab = [];
theta_cycles_ba = [];
theta_cycles_cooling_ab = [];
theta_cycles_cooling_ba = [];
theta_cycles_nocooling_ab = [];
theta_cycles_nocooling_ba = [];
theta_cycles_cooling_ab2 = [];
theta_cycles_cooling_ba2 = [];
theta_cycles_nocooling_ab2 = [];
theta_cycles_nocooling_ba2 = [];
for j = 1:size(pos_ab,1)
    for k = 1:size(nocooling,2)
        if find(animal.time(pos_ab(j,1)) > nocooling(1,k) & animal.time(pos_ab(j,2)) > nocooling(1,k) & animal.time(pos_ab(j,1)) < nocooling(2,k) & animal.time(pos_ab(j,2)) < nocooling(2,k));
            theta_cycles_nocooling_ab = [theta_cycles_nocooling_ab,j];
        end
    end
    for k = 1:size(cooling,2)
        if find(animal.time(pos_ab(j,1)) > cooling(1,k) & animal.time(pos_ab(j,2)) > cooling(1,k) & animal.time(pos_ab(j,1)) < cooling(2,k) & animal.time(pos_ab(j,2)) < cooling(2,k));
            theta_cycles_cooling_ab = [theta_cycles_cooling_ab,j];
        end
    end
end
for j = 1:size(pos_ba,1)
    for k = 1:size(nocooling,2)
        if find(animal.time(pos_ba(j,1)) > nocooling(1,k) & animal.time(pos_ba(j,2)) > nocooling(1,k) & animal.time(pos_ba(j,1)) < nocooling(2,k) & animal.time(pos_ba(j,2)) < nocooling(2,k));
            theta_cycles_nocooling_ba = [theta_cycles_nocooling_ba,j];
        end
    end
    for k = 1:size(cooling,2)
        if find(animal.time(pos_ba(j,1)) > cooling(1,k) & animal.time(pos_ba(j,2)) > cooling(1,k) & animal.time(pos_ba(j,1)) < cooling(2,k) & animal.time(pos_ba(j,2)) < cooling(2,k));
            theta_cycles_cooling_ba = [theta_cycles_cooling_ba,j];
        end
    end
end

for i = 1:size(pos_ab,1)
    theta_cycles_ab(i) = ( signal_phase2(round(animal.time(pos_ab(i,2))*Fs))-signal_phase2(round(animal.time(pos_ab(i,1))*Fs)) )/(2*pi);
end
for i = 1:size(pos_ba,1)
    theta_cycles_ba(i) = ( signal_phase2(round(animal.time(pos_ba(i,2))*Fs))-signal_phase2(round(animal.time(pos_ba(i,1))*Fs)) )/(2*pi);
end
theta_cycles = [theta_cycles_ab,theta_cycles_ba];
% Plots
figure,
histogram(theta_cycles,(18:2:100))
xlabel('Theta cycles'),ylabel('Count'),title('Theta cycles pr running trial')
%%%%%%%%%%%%
% Cooling ab
for i = 1:length(theta_cycles_cooling_ab)
    theta_cycles_cooling_ab2(i) = ( signal_phase2(round(animal.time(pos_ab(theta_cycles_cooling_ab(i),2))*Fs))-signal_phase2(round(animal.time(pos_ab(theta_cycles_cooling_ab(i),1))*Fs)) )/(2*pi);
end
% No Cooling ab
for i = 1:length(theta_cycles_nocooling_ab)
    theta_cycles_nocooling_ab2(i) = ( signal_phase2(round(animal.time(pos_ab(theta_cycles_nocooling_ab(i),2))*Fs))-signal_phase2(round(animal.time(pos_ab(theta_cycles_nocooling_ab(i),1))*Fs)) )/(2*pi);
end
% Cooling ba
for i = 1:length(theta_cycles_cooling_ba)
    theta_cycles_cooling_ba2(i) = ( signal_phase2(round(animal.time(pos_ba(theta_cycles_cooling_ba(i),2))*Fs))-signal_phase2(round(animal.time(pos_ba(theta_cycles_cooling_ba(i),1))*Fs)) )/(2*pi);
end
% No Cooling ba
for i = 1:length(theta_cycles_nocooling_ba)
    theta_cycles_nocooling_ba2(i) = ( signal_phase2(round(animal.time(pos_ba(theta_cycles_nocooling_ba(i),2))*Fs))-signal_phase2(round(animal.time(pos_ba(theta_cycles_nocooling_ba(i),1))*Fs)) )/(2*pi);
end
theta_cycles_cooling = [theta_cycles_cooling_ab2(theta_cycles_cooling_ab2<75),theta_cycles_cooling_ba2(theta_cycles_cooling_ba2<75)];
theta_cycles_nocooling = [theta_cycles_nocooling_ab2(theta_cycles_nocooling_ab2<75),theta_cycles_nocooling_ba2(theta_cycles_nocooling_ba2<75)];

figure,
histogram(theta_cycles_cooling,(18:2:75)), hold on
histogram(theta_cycles_nocooling,(18:2:75)),legend({'Cooling','No Cooling'})
gridxy(mean(theta_cycles_cooling),'color',[0 0 1],'linewidth',2)
gridxy(mean(theta_cycles_nocooling),'color',[1 0 0],'linewidth',2)
xlabel('Theta cycles'),ylabel('Count'),title('Theta cycles pr running trial'),axis tight
