% Medial Septum Cooling project
clear all
MedialSeptum_Recordings
id = 128;

recording = recordings(id);
if ~isempty(recording.dataroot)
    datapath = recording.dataroot;
end
cd([datapath, recording.name(1:6) recording.animal_id '\' recording.name, '\'])

Intan_rec_info = read_Intan_RHD2000_file_Peter([datapath, recording.name(1:6) recording.animal_id '\' recording.name, '\']);
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
Optitrack.position1D = Optitrack.position3D(2,:);
% Loading digital inputs
disp('Loading digital inputs')
if ~exist('digitalchannels.mat')
    [digital_on,digital_off] = Process_IntanDigitalChannels('digitalin.dat');
    save('digitalchannels.mat','digital_on','digital_off')
else load('digitalchannels.mat'); end

prebehaviortime = 0;
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
     temp_ = dir(fname);
     behaviortime = temp_.bytes/nChannels/2/sr;
% end

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
    
    animal.pos  = Optitrack.position3D([2,1,3],:) + [105,1,0]';
    %animal.pos(:,find(animal.pos(2,:)>70)) = 0;
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

animal.pos_x_limits = [60,180]; % x direction [5,145]
animal.pos_y_limits = [-30,30]; % y direction
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

if recording.cooling_session == 0
    cooling.onsets = animal.time(round(length(animal.time)/2));
    cooling.offsets = animal.time(round(length(animal.time)));
    cooling.cooling = [cooling.onsets(1)+10;cooling.offsets(1)];
    cooling.nocooling = [[1,cooling.onsets(1)];[cooling.offsets(1)+120,behaviortime]]';
else
    if inputs.ch_temp ~= 0
        temp_range = [32,34];% temp_1 defines the upper limit on cooling, temp_2 the lower limit on no cooling
        test = find(diff(temperature.temp < temp_range(1),2)== 1);
        test(diff(test)<10*temperature.sr)=[];
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
end

trials.labels = {'AB','BA'};
trials.total = length(trials.ab.start);
figure, subplot(1,2,1)
plot(animal.pos(1,:),trials.trials{1},'b'), hold on, plot(animal.pos(1,:),trials.trials{2}+0.25,'r'),title('Trials'), axis tight
subplot(1,2,2)
plot(animal.pos(1,:),animal.pos(2,:),'-k'), hold on
plot(animal.pos(1,find(~isnan(trials.trials{1}))),animal.pos(2,find(~isnan(trials.trials{1}))),'.b')
plot(animal.pos(1,find(~isnan(trials.trials{2}))),animal.pos(2,find(~isnan(trials.trials{2}))),'.r'),title('Position'), axis tight

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
signal2 = resample(signal,recording.sr_lfp,Fs);
Fpass = [1,49];
Wn_theta = [Fpass(1)/(recording.sr_lfp/2) Fpass(2)/(recording.sr_lfp/2)]; % normalized by the nyquist frequency
[btheta,atheta] = butter(3,Wn_theta);
signal_filtered = filtfilt(btheta,atheta,signal2);
%[wt,~,~] = awt_freqlist(signal_filtered,sr_temperature,freqlist);
%wt2 = abs(wt)'; clear wt
wt = spectrogram(signal_filtered,100,99,freqlist,recording.sr_lfp);
wt2 = [zeros(length(freqlist),49),abs(wt), zeros(length(freqlist),50)]; clear wt

speed = interp1(animal.time,animal.speed,(1:length(signal_filtered))/recording.sr_lfp);
pos = interp1(animal.time,animal.pos,(1:length(signal_filtered))/recording.sr_lfp);

t_cooling = zeros(1,length(signal_filtered)); t_nocooling = zeros(1,length(signal_filtered));
for i = 1:size(cooling,2), t_cooling(cooling(1,i)*recording.sr_lfp:cooling(2,i)*recording.sr_lfp) = 1; end
for i = 1:size(nocooling,2), t_nocooling(nocooling(1,i)*recording.sr_lfp:nocooling(2,i)*recording.sr_lfp) = 1; end

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
