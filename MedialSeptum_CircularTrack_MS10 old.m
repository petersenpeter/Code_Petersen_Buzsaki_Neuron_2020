%DOCID = '1WBqEo0OM5qdqmAD_7cGsJe0iyGf6hjgPXy-V2VlVY9c'
%result = GetGoogleSpreadsheet(DOCID); 
% Medial Septum Circular Track
clear all
Recordings_MedialSeptum
id = 60; % 63

recording = recordings(id).name;
cd([datapath, recordings(id).name(1:6) recordings(id).rat_id '\' recording, '\'])
Intan_rec_info = read_Intan_RHD2000_file_Peter([datapath, recordings(id).name(1:6) recordings(id).rat_id '\' recording, '\']);
fname = [recording '.dat'];
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
maze = recordings(id).maze;
sr_lfp = sr/16;
% track_boundaries = recordings(id).track_boundaries;
arena = recordings(id).arena;
% nb_lfp_periods = ceil((time_frame(2)-time_frame(1))/lfp_periods);
temp_ = dir(fname);
recording_length = round(temp_.bytes/sr/nbChan/2)-1;
animal = [];

% Camera tracking: Loading position data
disp('1. Loading Camera tracking data')
if inputs.ch_camera_sync
    if ~exist('Camera.mat')
        colors = [1,3]; % RGB based
        Camera = ImportCameraData(recordings(id).Cameratracking,pwd,colors,arena);
    else
        load('Camera.mat');
    end
end
% Optitrack: Loading position data
disp('2. Loading Optitrack tracking data')
if inputs.ch_OptiTrack_sync
    if ~exist('Optitrack.mat')
        Optitrack = LoadOptitrack(recordings(id).OptiTracktracking,1,arena, 0,0)
        save('Optitrack.mat','Optitrack')
    else load('Optitrack.mat'); end
end
if ~isempty(recordings(id).OptiTracktracking_offset)
    Optitrack.position3D = Optitrack.position3D + recordings(id).OptiTracktracking_offset';
end

% Loading digital inputs
disp('3. Loading digital inputs')
if ~exist('digitalchannels.mat')
    [digital_on,digital_off] = Process_IntanDigitalChannels('digitalin.dat');
    save('digitalchannels.mat','digital_on','digital_off');
%     if length(recordings(id).concat_recordings) == 0
%         [digital_on,digital_off] = Process_IntanDigitalChannels('digitalin.dat');
%         save('digitalchannels.mat','digital_on','digital_off');
%     else
%         %fullfile([datapath, recordings(id).name(1:6) recordings(id).rat_id], recordings(id).concat_recordings(recordings(id).concat_behavior_nb), 'digitalin.dat');
%         [digital_on,digital_off] = Process_IntanDigitalChannels('digitalin.dat');
%         save('digitalchannels.mat','digital_on','digital_off');
%     end
else
    load('digitalchannels.mat');
end

prebehaviortime = 0;
if recordings(id).concat_behavior_nb > 0
    prebehaviortime = 0;
    if recordings(id).concat_behavior_nb > 1
    for i = 1:recordings(id).concat_behavior_nb-1
        fullpath = fullfile([datapath, recordings(id).name(1:6) recordings(id).rat_id], recordings(id).concat_recordings{i}, [recordings(id).concat_recordings{i}, '.dat']);
        temp2_ = dir(fullpath);
        prebehaviortime = prebehaviortime + temp2_.bytes/nbChan/2;
    end
    end
    i = recordings(id).concat_behavior_nb;
    fullpath = fullfile([datapath, recordings(id).name(1:6) recordings(id).rat_id], recordings(id).concat_recordings{i}, [recordings(id).concat_recordings{i}, '.dat']);
    temp2_ = dir(fullpath);
    behaviortime = temp2_.bytes/nbChan/2;
else
    temp_ = dir(fname);
    behaviortime = temp_.bytes/nbChan/2;
end

disp('4. Calculating behavior')
if inputs.ch_camera_sync ~= 0
    animal.sr = Camera.framerate;
    if length(recordings(id).concat_recordings) > 0
        camera_pulses = find(digital_on{inputs.ch_camera_sync} > prebehaviortime & digital_on{inputs.ch_camera_sync} < prebehaviortime + recording_length*sr);
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
    animal.pos  = prebehaviortime/sr;
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
animal.arm = (animal.pos(1,:) > animal.pos_x_limits(1) & animal.pos(1,:) < animal.pos_x_limits(2) & animal.pos(2,:) > animal.pos_y_limits(1) & animal.pos(2,:) < animal.pos_y_limits(2));
animal.rim = (animal.polar_rho > animal.polar_rho_limits(1) & animal.polar_rho < animal.polar_rho_limits(2) & animal.polar_theta > animal.polar_theta_limits(1) & animal.polar_theta < animal.polar_theta_limits(2));
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
        temperature.temp = nanconv((v_downsample-1.25)/0.005,gausswin(200)','edge');
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
disp('6. Creating Cooling structure')

if inputs.ch_temp ~= 0
    temp_range = [34.];% temp_1 defines the upper limit on cooling, temp_2 the lower limit on no cooling
    test = find(diff(temperature.temp < temp_range(1),2)== 1);
    test(diff(test)<10*temperature.sr)=[];
    cooling.onsets = temperature.time(test(temperature.temp(test+1)-temperature.temp(test)<0));
    cooling.offsets = temperature.time(test(temperature.temp(test+1)-temperature.temp(test)>0))+60;
    cooling.cooling = [cooling.onsets;cooling.offsets];
    cooling.nocooling = reshape([prebehaviortime/sr;cooling.cooling(:);prebehaviortime/sr+behaviortime/sr],[2,size(cooling.cooling,2)+1]);
elseif inputs.ch_CoolingPulses ~= 0
    cooling.onsets = digital_on{inputs.ch_CoolingPulses}/sr;
    cooling.offsets = cooling.onsets + 12*60;
    cooling.cooling = [cooling.onsets(1)+10;cooling.offsets(1)];
    cooling.nocooling = [[1,cooling.onsets'];[cooling.offsets'+120,behaviortime/sr]]';
else
    cooling.cooling = [cooling.onsets(1)+10;cooling.offsets(1)]+prebehaviortime/sr;
    cooling.nocooling = [[1,cooling.onsets(1)]+prebehaviortime/sr;[cooling.offsets(1)+120,behaviortime/sr]+prebehaviortime/sr]';
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
trials.Cooling_error_ratio = length(find(animal.time(trials.start(trials.error)) > cooling.cooling(1) & animal.time(trials.start(trials.error)) < cooling.cooling(2)))/length(find(animal.time(trials.start) > cooling.cooling(1) & animal.time(trials.start) < cooling.cooling(2)));
trials.NoCooling_error_ratio_before = length(find(animal.time(trials.start(trials.error)) < cooling.cooling(1)))/length(find(animal.time(trials.start) < cooling.cooling(1)));
trials.NoCooling_error_ratio_after = length(find(animal.time(trials.start(trials.error)) > cooling.cooling(2)+60))/length(find(animal.time(trials.start) > cooling.cooling(2)+60));

figure,
subplot(1,2,1)
plot3(animal.pos(1,:),animal.pos(2,:),animal.pos(3,:),'-k','markersize',2), hold on, plot_ThetaMaze(maze)
plot3(animal.pos(1,trials.left),animal.pos(2,trials.left),animal.pos(3,trials.left),'.r','markersize',2), hold on
plot3(animal.pos(1,trials.right),animal.pos(2,trials.right),animal.pos(3,trials.right),'.b','markersize',2), title('Cartesian Coordinates')
xlabel('x (cm)'),ylabel('y (cm)')
axis equal
xlim(maze.radius_out*[-1.2,1.2]),ylim(maze.radius_out*[-1.4,1.2])

subplot(2,4,3:4)
bins_speed = [0:5:120];
plot(animal.time,animal.speed,'k'), axis tight, title('Speed of the animal'), hold on, 
plot(cooling.cooling,[0,0],'color','blue','linewidth',2)
plot(cooling.nocooling,[0,0],'color','red','linewidth',2)
gridxy(animal.time(trials.start),'color','g')
gridxy(animal.time(trials.start(trials.error)),'color','m','linewidth',2)
legend({'Error trials','All trials','Speed','Cooling','NoCooling'})
xlabel('Time (s)'), ylabel('Speed (cm/s)')

subplot(2,4,7)
cooling.times_cooling = find(animal.time(trials.all)> cooling.cooling(1) & animal.time(trials.all) < cooling.cooling(2));
cooling.times_nocooling = find(animal.time(trials.all)< cooling.cooling(1) | animal.time(trials.all) > cooling.cooling(2));
histogram(animal.speed(trials.all(cooling.times_cooling)),'BinEdges',bins_speed,'Normalization','probability'), hold on
histogram(animal.speed(trials.all(cooling.times_nocooling)),'BinEdges',bins_speed,'Normalization','probability'),
legend({'Cooling','No Cooling'})
xlabel('Speed (cm/s)'), ylabel('Probability'), title(['Speed during the trials (Total: ' num2str(length(trials.start))  ' trials)'])

subplot(2,4,8)
bar(1, trials.NoCooling_error_ratio_before, 'red'), hold on
bar(2, trials.Cooling_error_ratio, 'blue')
bar(3, trials.NoCooling_error_ratio_after, 'red'), hold on
xticks([1, 2, 3]), xticklabels({'Pre Cooling','Cooling','Post cooling'}),ylabel('Percentage of errors'),title('Error trials (%)'),axis tight,
xlim([0,4]),ylim([0,0.3])
disp('8. Finished loading the recording')

% % Position on the rim defined in polar coordinates
figure,
subplot(2,2,1)
plot(animal.pos(1,:),animal.pos(2,:),'.k','markersize',2), hold on
plot(animal.pos(1,trials.left),animal.pos(2,trials.left),'.r','markersize',2)
plot(animal.pos(1,trials.right),animal.pos(2,trials.right),'.b','markersize',2), title('Cartesian Coordinates')
xlabel('x (cm)'),ylabel('y (cm)')
plot([animal.pos_x_limits(1) animal.pos_x_limits(2) animal.pos_x_limits(2) animal.pos_x_limits(1) animal.pos_x_limits(1)],[animal.pos_y_limits(1) animal.pos_y_limits(1) animal.pos_y_limits(2) animal.pos_y_limits(2) animal.pos_y_limits(1)],'c','linewidth',2)
plot_ThetaMaze(maze)
axis equal
xlim([-70,70]),ylim([-70,70]),

subplot(2,2,[3,4])
plot(50*animal.polar_theta,animal.polar_rho,'.k','markersize',2), hold on
plot(50*animal.polar_theta(trials.left),animal.polar_rho(trials.left),'.r','markersize',2)
plot(50*animal.polar_theta(trials.right),animal.polar_rho(trials.right),'.b','markersize',2), title('Polar Coordinates')
plot(50*[animal.polar_theta_limits(1),animal.polar_theta_limits(2) animal.polar_theta_limits(2) animal.polar_theta_limits(1) animal.polar_theta_limits(1)],[animal.polar_rho_limits(1) animal.polar_rho_limits(1) animal.polar_rho_limits(2) animal.polar_rho_limits(2) animal.polar_rho_limits(1)],'g','linewidth',2)
xlim(50*[-pi,pi]),ylim([0,70]),
xlabel('Circular position (cm)'),ylabel('rho (cm)')
subplot(2,2,2)
plot(animal.pos(1,:),animal.pos(2,:),'.k','markersize',2), hold on
plot(animal.pos(1,animal.circularpart),animal.pos(2,animal.circularpart),'.g','markersize',2)
plot(animal.pos(1,animal.centralarmpart),animal.pos(2,animal.centralarmpart),'.c','markersize',2)
xlabel('x (cm)'),ylabel('y (cm)'), title('Rim and Center arm defined')
plot_ThetaMaze(maze)
axis equal
xlim([-70,70]),ylim([-70,70]),
save('animal.mat','animal')

%% % 
bins_speed = [5:3:80];
bins_speed_avg = mean([bins_speed(2:end);bins_speed(1:end-1)]);
freqlist = [5:0.025:10]; %10.^(0.4771:0.01:1.1761);
%freqlist = [4:1:150];

% if recordings(id).concat_behavior_nb > 1
%     % Create eeg file
%     noconcat_eeg_file = recordings(id).concat_recordings{recordings(id).concat_behavior_nb(1)};
%     noconcat_eeg_folder = fullfile(datapath, [recordings(id).name(1:6), recordings(id).rat_id], recordings(id).concat_recordings{recordings(id).concat_behavior_nb(1)});
%     if ~exist(fullfile(noconcat_eeg_folder,[noconcat_eeg_file '.eeg']))
%         disp('Creating EEG file')
%         downsample_dat_to_eeg(noconcat_eeg_file,noconcat_eeg_folder);
%     end
%     signal = 0.000050354 * double(LoadBinary(fullfile(noconcat_eeg_folder,[noconcat_eeg_file '.eeg']),'nChannels',nbChan,'channels',ch_theta,'precision','int16','frequency',sr/16)); % ,'start',start,'duration',duration
% else
%     % Create eeg file
%     if ~exist([recording, '.eeg'])
%         disp('Creating EEG file')
%         downsample_dat_to_eeg(recording,pwd);
%     end
%     signal = 0.000050354 * double(LoadBinary([recording '.eeg'],'nChannels',nbChan,'channels',ch_theta,'precision','int16','frequency',sr/16)); % ,'start',start,'duration',duration
% end
if ~exist([recording, '.eeg'])
    disp('Creating EEG file')
    downsample_dat_to_eeg(recording,pwd);
end
signal = 0.000050354 * double(LoadBinary([recording '.eeg'],'nChannels',nbChan,'channels',ch_theta,'precision','int16','frequency',sr_lfp)); % ,'start',start,'duration',duration
sr_theta = animal.sr;
signal2 = resample(signal,sr_theta,sr_lfp);
Fpass = [1,49];
% if sr_lfp < 100
%     Fpass = [1,14.9];
% end
Wn_theta = [Fpass(1)/(sr_theta/2) Fpass(2)/(sr_theta/2)]; % normalized by the nyquist frequency
[btheta,atheta] = butter(3,Wn_theta);
signal_filtered = filtfilt(btheta,atheta,signal2);
%[wt,~,~] = awt_freqlist(signal_filtered,sr_temperature,freqlist);
%wt2 = abs(wt)'; clear wt
wt = spectrogram(signal_filtered,100,99,freqlist,sr_theta);
wt2 = [zeros(length(freqlist),49),abs(wt), zeros(length(freqlist),50)]; clear wt

speed = interp1(animal.time,animal.speed,(1:length(signal_filtered))/sr_theta);
pos = interp1(animal.time,animal.pos',(1:length(signal_filtered))/sr_theta);

t_cooling = zeros(1,length(signal_filtered)); t_nocooling = zeros(1,length(signal_filtered));
for i = 1:size(cooling.cooling,2), t_cooling(cooling.cooling(1,i)*sr_theta:cooling.cooling(2,i)*sr_theta) = 1; end
for i = 1:size(cooling.nocooling,2), t_nocooling(cooling.nocooling(1,i)*sr_theta:cooling.nocooling(2,i)*sr_theta) = 1; end

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
set(gca,'YTick',bins_speed), set(gca,'YTick',freqlist(1):freqlist(end)), %set(gca,'xscale','log')
cmax = max(max(wt_speed_cooling(:)',wt_speed_nocooling(:)'));
colorbar,%caxis([0 cmax])
subplot(2,2,2)
% surf(bins_lfp,bins_speed_avg,lfp_wavelets_combined,'EdgeColor','None'), view(0,90)
imagesc(bins_speed_avg,freqlist,wt_speed_nocooling'), set(gca,'Ydir','normal')
axis tight, title(['Without Cooling']), ylabel('Powerspectrum (Hz)'),xlabel('Speed (cm/s)')
set(gca,'YTick',bins_speed), set(gca,'YTick',freqlist(1):freqlist(end)), %set(gca,'xscale','log')
colorbar,%caxis([0 cmax])
subplot(2,3,4)
stairs(freqlist,mean(wt_speed_cooling),'b'), hold on
stairs(freqlist,mean(wt_speed_nocooling),'r'), hold on
xlabel('Frequency (Hz)'), grid on, title('Average frequency'),xlim([freqlist(1),freqlist(end)]),ylabel('Power'),ylim([0,max(max(mean(wt_speed_nocooling),mean(wt_speed_cooling)))])

subplot(2,3,5)
time = bins_speed_avg(find(thetamax_cooling>2));
y1 = freqlist(thetamax_cooling(find(thetamax_cooling>2)));
plot(time,y1,'ob'), hold on
P = polyfit(time,y1,1);
yfit = P(1)*bins_speed+P(2);
plot(bins_speed,yfit,'b-');
text(15,y1(1)+0.2,['Cooling slope: ', num2str(P(1),3)],'Color','blue')
[r,p] = corr(time',y1')

time = bins_speed_avg(find(thetamax_nocooling>2));
y2 = freqlist(thetamax_nocooling(find(thetamax_nocooling>2)));
plot(time,y2,'or'), hold on
P = polyfit(time,y2,1);
yfit = P(1)*bins_speed+P(2);
plot(bins_speed,yfit,'r-');
text(15,y2(10)+0.4,['No Cooling slope: ', num2str(P(1),3)],'Color','red')
[r,p] = corr(time',y2');
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
text(15,y1(1)-0.1,['Cooling slope: ', num2str(P(1),3)],'Color','blue')
[r,p] = corr(time',y1')

time = bins_speed_avg(find(thetamax_nocooling>2));
y2 = theta_maxpower_nocooling(find(thetamax_nocooling>2));
plot(time,y2,'or'), hold on
P = polyfit(time,y2,1);
yfit = P(1)*bins_speed+P(2);
plot(bins_speed,yfit,'r-');
text(15,y2(1)+0.2,['No Cooling slope: ', num2str(P(1),3)],'Color','red')
[r,p] = corr(time',y2')
xlim([bins_speed(1),bins_speed(end)]), ylim([y1(1)-0.4,y2(end)+0.4]),
xlabel('Speed (cm/s)'),ylabel('Peak Power'), grid on, title('Amplitude theta'),xlim([bins_speed(1),bins_speed(end)])
% GLM fit
if ~isempty(temperature)
running_times = find(speed>5);
X = [animal.temperature(running_times)+37+60; speed(running_times);(animal.temperature(running_times)++37+60).*speed(running_times); [diff(speed(running_times)),0]]';
[temp_theta_maxpower,temp_theta_max] = max(wt2(:,(running_times)));
y1 = freqlist(temp_theta_max)'; % Frequency
y2 = temp_theta_maxpower'; % Power
mdl1 = fitglm(X,y1)
mdl2 = fitglm(X,y2)
theta_bins = [freqlist(1):0.1:freqlist(end)];
hist_theta = hist2d([y1';(mdl1.predict)']',theta_bins,theta_bins);
figure, imagesc(theta_bins,theta_bins,hist_theta), xlabel('Real theta'), ylabel('Predicted theta')
end

%% % Units
Recordings_MedialSeptum
SpikeSorting_method = recordings(id).SpikeSorting.method; % Phy (.npy files) e.g.: SpikingCircus, Kilosort. Klustakwik (.clu,.res,.spk): , KlustaSuite ()
SpikeSorting_path = recordings(id).SpikeSorting.path;
shanks = recordings(id).SpikeSorting.shanks;
units = loadClusteringData(recording,SpikeSorting_method,SpikeSorting_path,shanks);
if ~exist([recording, '.eeg'])
    disp('Creating EEG file')
    downsample_dat_to_eeg(recording,pwd);
    % movefile('amplifier.eeg',[recording '.eeg'])
    % copyfile('amplifier.xml',[recording '.xml'])
end

% Calculating the instantaneous theta frequency
theta = [];
theta.sr = sr_lfp;
theta.ch_theta = ch_theta;
[signal_phase,signal_phase2] = calcInstantaneousTheta(recording,nbChan,ch_theta,sr);
theta.phase = signal_phase;
theta.phase2 = signal_phase2;
% theta.time = [1:length(theta.phase)]/theta.sr_eeg;
clear signal_phase signal_phase2

disp('Plotting instantaneous theta frequency for all units')
% figure, plot(signal_phase)
for i = 1:size(units,2)
    % units(i).ts = units(i).ts(units(i).ts/sr < length(signal_phase)/sr_eeg);
    units(i).total = length(units(i).ts);
    units(i).ts_eeg = ceil(units(i).ts/16);
    units(i).theta_phase = theta.phase(units(i).ts_eeg);
    units(i).theta_phase2 = theta.phase2(units(i).ts_eeg);
    units(i).speed = interp1(animal.time+prebehaviortime/sr,animal.speed,units(i).ts/sr);
    units(i).pos = interp1(animal.time+prebehaviortime/sr,animal.pos',units(i).ts/sr)';
    units(i).polar_theta = interp1(animal.time+prebehaviortime/sr,animal.polar_theta,units(i).ts/sr);
    units(i).polar_rho = interp1(animal.time+prebehaviortime/sr,animal.polar_rho,units(i).ts/sr);
    units(i).arm = zeros(1,length(units(i).ts));
    units(i).arm(units(i).pos(1,:) > animal.pos_x_limits(1) & units(i).pos(1,:) < animal.pos_x_limits(2) & units(i).pos(2,:) > animal.pos_y_limits(1) & units(i).pos(2,:) < animal.pos_y_limits(2)) = 1;
    units(i).rim = zeros(1,length(units(i).ts));
    units(i).rim(units(i).polar_rho > animal.polar_rho_limits(1) & units(i).polar_rho < animal.polar_rho_limits(2) & units(i).polar_theta > animal.polar_theta_limits(1) & units(i).polar_theta < animal.polar_theta_limits(2)) = 1;
    
    %units(i).trials = interp1(animal.time+prebehaviortime/sr,trials.trials,units(i).ts/sr,'nearest');
    units(i).state = interp1(animal.time+prebehaviortime/sr,trials.state,units(i).ts/sr,'nearest');
end
disp('done')

disp('Plotting Phase precession')
figure, 
for i = 1:size(units,2)
    subplot(1,3,1)
    plot(units(i).pos(1,:),units(i).pos(2,:),'.'), hold on
    legend({'1','2','3','4','5','6','7','8','9'})
	plot_ThetaMaze(maze)
    axis equal
    xlim([-65,65]),ylim([-65,65]),
    subplot(1,3,2)
    histogram(units(i).polar_theta,'Normalization','probability'), hold on
    subplot(1,3,3)
    plot(units(i).polar_theta,units(i).polar_rho,'.'), hold on
end
%figure,
for i = 1:size(units,2)
    subplot_Peter(5,1,i)
    lin1 = units(i).pos(2,:);
    circ1 = units(i).theta_phase;
    plot(lin1,circ1,'.k','markersize',2), hold on
    plot(lin1,circ1+2*pi,'.k','markersize',2)
    ax = gca;
    ax.YTick = ([-pi 0 pi 2*pi 3*pi]);
    ax.YTickLabels = ({'-\pi','0','\pi','2\pi','3\pi'});
    hold off, axis tight, xlim([-60,60]),ylim([-pi,3*pi]),
    ylabel(['Unit ' num2str(i)])
    if mod(i,5) == 1
        title(['Position on the central arm'])
    end
    if mod(i,5) == 0
        xlabel('Position (cm)')
    end
end
% figure,
for i = 1:size(units,2)
    subplot_Peter(5,1,i)
    lin1 = units(i).polar_theta;
    circ1 = units(i).theta_phase;
    plot(lin1,circ1,'.k','markersize',2), hold on
    plot(lin1,circ1+2*pi,'.k','markersize',2)
    ax = gca;
    ax.YTick = ([-pi 0 pi 2*pi 3*pi]);
    ax.YTickLabels = ({'-\pi','0','\pi','2\pi','3\pi'});
    hold off, axis tight, xlim([-50*pi,50*pi]),ylim([-pi,3*pi]),
    ylabel(['Unit ' num2str(i)])
    if mod(i,5) == 1
        title('Position on the rim')
    end
    if mod(i,5) == 0
        xlabel('Position (cm)')
    end
end
%% % Plotting Phase precession
behavior = animal; % Animal Strucutre
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
units2(1).PhasePrecession = [];

PhasePrecessionSlope = plot_FiringRateMap(behavior,units2,trials,theta,sr);
clear behavior units2;

% Firing rate on center arm
behavior = animal;
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
units2(1).PhasePrecession = [];
PhasePrecessionSlope = plot_FiringRateMap(behavior,units2,trials,theta,sr);
clear behavior units2;

%%
% Polar plot - PHASE PRECESSION
polar_theta_placecells = cell(1,size(units,2));
for i = 1:size(recordings(id).SpikeSorting.polar_theta_placecells,2)
    if ~isempty(recordings(id).SpikeSorting.polar_theta_placecells{i})
        polar_theta_placecells{i} = recordings(id).SpikeSorting.polar_theta_placecells{i};
    end
end
% Center arm - PHASE PRECESSION
center_arm_placecells = cell(1,size(units,2));
for i = 1:size(recordings(id).SpikeSorting.center_arm_placecells,2)
    if ~isempty(recordings(id).SpikeSorting.center_arm_placecells{i})
        center_arm_placecells{i} = recordings(id).SpikeSorting.center_arm_placecells{i};
    end
end

cooling.sessions = [cooling.nocooling(1,1),cooling.nocooling(2,1);cooling.cooling(1),cooling.cooling(2);cooling.nocooling(1,2),cooling.nocooling(2,2)];
cooling.colors = {'.r','.b','.g'};
cooling.labels = {'Pre Cooling','Cooling','Post Cooling'};
colors = [0.8500, 0.3250, 0.0980; 0, 0.4470, 0.7410; 0.9290, 0.6940, 0.1250; 0.4940, 0.1840, 0.5560; 0.4660, 0.6740, 0.1880; 0.3010, 0.7450, 0.9330; 0.6350, 0.0780, 0.1840];
% Polar plot - PHASE PRECESSION
phaseprecession = [];
units_theta_trials = [];
x_hist2 = [50*animal.polar_theta_limits(1):5:50*animal.polar_theta_limits(2)];
for i = 1:size(units,2)
    for k = 1:size(cooling.sessions,1)
        subplot_Peter(5,4,i,k)
        %subplot(size(units.ts,2),1+size(cooling.sessions,1),i*(size(cooling.sessions,1)+1)-(size(cooling.sessions,1)+1)+k)
        indexes = find(units(i).ts/sr > cooling.sessions(k,1) & units(i).ts/sr < cooling.sessions(k,2) & units(i).polar_rho > animal.polar_rho_limits(1) & units(i).polar_rho < animal.polar_rho_limits(2) & units(i).polar_theta > 50*animal.polar_theta_limits(1) & units(i).polar_theta < 50*animal.polar_theta_limits(2) & units(i).speed > 10);
        for ii = trials.error
            indexes(units(i).ts(indexes)/sr > animal.time(trials.start(ii)) & units(i).ts(indexes)/sr < animal.time(trials.end(ii))) = [];
        end
        lin1 = units(i).polar_theta(indexes);
        circ1 = units(i).theta_phase(indexes);
        plot(lin1,circ1,'.k','markersize',3), hold on
        plot(lin1,circ1+2*pi,'.k','markersize',3)
        if ~isempty(polar_theta_placecells{i})
            for j = 1:size(polar_theta_placecells{i},1)
                x = polar_theta_placecells{i}(j,1):1:polar_theta_placecells{i}(j,2);
                indexes1 = find(lin1 > polar_theta_placecells{i}(j,1) & lin1 < polar_theta_placecells{i}(j,2));
                if ~isempty(indexes1)
                    plot(lin1(indexes1),circ1(indexes1),'.','color',colors(k,:),'markersize',3), hold on
                    plot(lin1(indexes1),circ1(indexes1)+2*pi,'.','color',colors(k,:),'markersize',3)
                    [slope1,offset1,R1] = CircularLinearRegression(circ1(indexes1),lin1(indexes1));
                    while  2*pi*slope1*x(round(length(x)./2))+ offset1 < 0; offset1 = offset1+2*pi; end
                    while 2*pi*slope1*x(round(length(x)./2))+ offset1 > 2*pi; offset1 = offset1-2*pi;  end
                    phaseprecession.polar.slope1(k,i,j) = slope1;
                    phaseprecession.polar.offset1(k,i,j) = offset1;
                    phaseprecession.polar.R1(k,i,j) = R1;
                    plot(x,2*pi*slope1*x + offset1,'k-','linewidth',1.5)
                    text(x(1),2*pi*slope1(1)*x(1)+offset1(1)+pi/2,num2str(slope1))
                end
            end
        end
        ax = gca;
        ax.YTick = ([-pi 0 pi 2*pi 3*pi]);
        ax.YTickLabels = ({'-\pi','0','\pi','2\pi','3\pi'});
        axis tight, xlim(50*animal.polar_theta_limits),ylim([-pi,3*pi]),
        if k == 1; ylabel(['Unit ' num2str(i)]); end 
        % if mod(i,5) == 0; xlabel('Position on rim (cm)'); end 
        if mod(i,5) == 1; title(cooling.labels{k}); end 
    end
    % Polar plot - TRIALS
    m = 1;
    subplot_Peter(5,4,i,4)
    %subplot(size(units.ts,2),1+size(cooling.sessions,1),i*(size(cooling.sessions,1)+1))
    for j = 1:length(trials.start)
        indexes = find(units(i).ts/sr > animal.time(trials.start(j)) & units(i).ts/sr < animal.time(trials.end(j)) & units(i).polar_rho > animal.polar_rho_limits(1) & units(i).polar_rho < animal.polar_rho_limits(2) & units(i).polar_theta > 50*animal.polar_theta_limits(1) & units(i).polar_theta < 50*animal.polar_theta_limits(2)  & units(i).speed > 10);
        if ismember(j,trials.error)
            plot(units(i).polar_theta(indexes),j*ones(1,length(indexes)),'.k','markersize',3), hold on
        else
            plot(units(i).polar_theta(indexes),j*ones(1,length(indexes)),'.','color',colors(trials.cooling(j),:)), hold on
            units_theta_trials{i}(m,:) = hist(units(i).polar_theta(indexes),x_hist2);
            m = m + 1;
        end
    end
    plot(50*animal.polar_theta_limits,[find(diff(trials.cooling));find(diff(trials.cooling))],'k')
    xlim(50*animal.polar_theta_limits),ylim([0,length(trials.start)]),
    if mod(i,5) == 1; title('Trials'); end
    if mod(i,5) == 0 | i == size(units,2); xlabel('Rim Position (cm)'); end 
end

% Firing rate on rim
figure
for k = 1:size(cooling.sessions,1)
    indexes2 = find(animal.time > cooling.sessions(k,1) & animal.time < cooling.sessions(k,2) & animal.polar_rho > animal.polar_rho_limits(1) & animal.polar_rho < animal.polar_rho_limits(2) & animal.polar_theta > animal.polar_theta_limits(1) & animal.polar_theta < animal.polar_theta_limits(2) & animal.speed > 10);
    for ii = trials.error
        indexes2(animal.time(indexes2) > animal.time(trials.start(ii)) & animal.time(indexes2) < animal.time(trials.end(ii))) = [];
    end
    polar_theta_counts = hist(50*animal.polar_theta(indexes2),x_hist2);
    for i = 1:size(units,2)
        subplot(ceil(size(units,2)/2),2,i)
        indexes = find(units(i).ts/sr > cooling.sessions(k,1) & units(i).ts/sr < cooling.sessions(k,2) & units(i).polar_rho > animal.polar_rho_limits(1) & units.polar_rho{i} < animal.polar_rho_limits(2) & units.polar_theta{i} > 50*animal.polar_theta_limits(1) & units.polar_theta{i} < 50*animal.polar_theta_limits(2) & units.speed{i} > 10);
        for ii = trials.error
            indexes(units(i).ts(indexes)/sr > animal.time(trials.start(ii)) & units(i).ts(indexes)/sr < animal.time(trials.end(ii))) = [];
        end
        lin1 = units(i).polar_theta(indexes);
        hist_polar_count = histc(lin1,x_hist2);
        stairs(x_hist2,animal.sr*hist_polar_count./polar_theta_counts,'color',colors(k,:)), hold on
        axis tight, xlim(50*animal.polar_theta_limits),
        title([ cooling.labels{k} ', Unit ' num2str(i)]), xlabel('Position on rim (cm)'), ylabel('Firing rate (Hz)')
    end
end
subplot(ceil(size(units,2)/2),2,1)
legend({'Pre','Cooling','Post'})

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %  % % % % % % % % % % % % % % % % % % % % %  % % % % % % % % % % % % % % % % % % % % %  % % % % % % % % % 
units_pos_trials = [];
x_hist2 = [animal.pos_y_limits(1):5:animal.pos_y_limits(2)];
for i = 1:size(units,2)
    for k = 1:size(cooling.sessions,1)
        subplot_Peter(5,4,i,k)
        % subplot(size(units.ts,2),1+size(cooling.sessions,1),i*(size(cooling.sessions,1)+1)-(size(cooling.sessions,1)+1)+k)
        indexes = find(units(i).ts/sr > cooling.sessions(k,1) & units(i).ts/sr < cooling.sessions(k,2) & units(i).pos(1,:) > animal.pos_x_limits(1) & units(i).pos(1,:) < animal.pos_x_limits(2) & units(i).pos(2,:) > animal.pos_y_limits(1) & units(i).pos(2,:) < animal.pos_y_limits(2) & units(i).speed > 10);
        for ii = trials.error
            indexes(units(i).ts(indexes)/sr > animal.time(trials.start(ii)) & units(i).ts(indexes)/sr < animal.time(trials.end(ii))) = [];
        end
        lin1 = units(i).pos(2,indexes);
        circ1 = units(i).theta_phase(indexes);
        plot(lin1,circ1,'.k','markersize',3), hold on
        plot(lin1,circ1+2*pi,'.k','markersize',3)
        if ~isempty(center_arm_placecells{i})
            for j = 1:size(center_arm_placecells{i},1)
                x = center_arm_placecells{i}(j,1):1:center_arm_placecells{i}(j,2);
                indexes1 = find(lin1 > center_arm_placecells{i}(j,1) & lin1 < center_arm_placecells{i}(j,2));
                if ~isempty(indexes1)
                    if length(indexes1) > 1
                        plot(lin1(indexes1),circ1(indexes1),'.','color',colors(k,:),'markersize',3), hold on
                        plot(lin1(indexes1),circ1(indexes1)+2*pi,'.','color',colors(k,:),'markersize',3)
                        [slope1,offset1,R1] = CircularLinearRegression(circ1(indexes1),lin1(indexes1));
                        
                        while slope1*x(round(length(x)./2))+ offset1 < 0; offset1 = offset1+2*pi; end
                        while slope1*x(round(length(x)./2))+ offset1 > 2*pi; offset1 = offset1-2*pi;  end
                        phaseprecession.arm.slope1(k,i,j) = slope1;
                        phaseprecession.arm.offset1(k,i,j) = offset1;
                        phaseprecession.arm.R1(k,i,j) = R1;
                        plot(x,2*pi*slope1*x+offset1,'k-','linewidth',1.5)
                        text(x(1),2*pi*slope1(1)*x(1)+offset1(1)+pi/2,num2str(slope1))
                    end
                end
            end
        end
        ax = gca;
        ax.YTick = ([-pi 0 pi 2*pi 3*pi]);
        ax.YTickLabels = ({'-\pi','0','\pi','2\pi','3\pi'});
        axis tight, xlim([animal.pos_y_limits]),ylim([-pi,3*pi]),
        
        if k == 1; ylabel(['Unit ' num2str(i)]); end 
        % if mod(i,5) == 0; xlabel('Position (cm)'); end 
        if mod(i,5) == 1; title(cooling.labels{k}); end
    end
    % Center Arm - TRIALS
    m = 1;
    subplot_Peter(5,4,i,4)
    %subplot(size(units.ts,2),1+size(cooling.sessions,1),i*(size(cooling.sessions,1)+1))
    for j = 1:length(trials.start)
        indexes = find(units(i).ts/sr > animal.time(trials.start(j)) & units(i).ts/sr < animal.time(trials.end(j)) & units(i).pos(1,:) > animal.pos_x_limits(1) & units(i).pos(1,:) < animal.pos_x_limits(2) & units(i).pos(2,:) > animal.pos_y_limits(1) & units(i).pos(2,:) < animal.pos_y_limits(2) & units(i).speed > 10);
        units_pos_trials{i}(m,:) = hist(units(i).pos(2,indexes),x_hist2);
        m = m + 1;
        if ismember(j,trials.error)
            plot(units(i).pos(2,indexes),j*ones(1,length(indexes)),'.k','markersize',3), hold on
        else
            plot(units(i).pos(2,indexes),j*ones(1,length(indexes)),'.','color',colors(trials.cooling(j),:)), hold on
            
        end
    end
    plot([animal.pos_y_limits],[find(diff(trials.cooling));find(diff(trials.cooling))],'k')
    xlim([animal.pos_y_limits]),ylim([0,length(trials.start)]),
    if mod(i,5) == 1; title('Trials'); end
    if mod(i,5) == 0 | i == size(units,2); xlabel('Central arm Position (cm)'); end
end

% Firing rate on arm
figure
for k = 1:size(cooling.sessions,1)
    indexes2 = find(animal.time > cooling.sessions(k,1) & animal.time < cooling.sessions(k,2) & animal.pos(1,:) > animal.pos_x_limits(1) & animal.pos(1,:) < animal.pos_x_limits(2) & animal.pos(2,:) > animal.pos_y_limits(1) & animal.pos(2,:) < animal.pos_y_limits(2) & animal.speed > 10);
    for ii = trials.error
        indexes2(animal.time(indexes2) > animal.time(trials.start(ii)) & animal.time(indexes2) < animal.time(trials.end(ii))) = [];
    end
    pos_counts = hist(animal.pos(2,indexes2),x_hist2);
    for i = 1:size(units,2)
        indexes = find(units(i).ts{i}/sr > cooling.sessions(k,1) & units(i).ts/sr < cooling.sessions(k,2) & units(i).pos(1,:) > animal.pos_x_limits(1) & units(i).pos(1,:) < animal.pos_x_limits(2) & units(i).pos(2,:) > animal.pos_y_limits(1) & units(i).pos(2,:) < animal.pos_y_limits(2) & units.speed{i} > 10);
        for ii = trials.error
            indexes(units(i).ts{i}(indexes)/sr > animal.time(trials.start(ii)) & units(i).ts(indexes)/sr < animal.time(trials.end(ii))) = [];
        end
        lin1 = units(i).pos{i}(2,indexes);
        hist_polar_count = histc(lin1,x_hist2);
        
        subplot(ceil(size(units,2)/2),2,i)
        stairs(x_hist2,animal.sr*hist_polar_count./pos_counts,'color',colors(k,:)), hold on
        axis tight, xlim(animal.pos_y_limits),
        title([ cooling.labels{k} ', Unit ' num2str(i)]), xlabel('Position on center arm (cm)'), ylabel('Firing rate (Hz)')
    end
end
subplot(ceil(size(units,2)/2),2,1)
legend({'Pre','Cooling','Post'})
% Histogram of phase precession slopes
figure
if isfield(phaseprecession, 'polar')
    for i = 1:size(phaseprecession.polar.slope1,2)
        for j = 1:size(phaseprecession.polar.slope1,3)
            phaseprecession.polar.slope1(find(abs(phaseprecession.polar.slope1(:,i,j))>0.2),i,j) = 0;
            plot(abs(phaseprecession.polar.slope1(:,i,j))), hold on
        end
    end
end
if isfield(phaseprecession, 'arm')
    for i = 1:size(phaseprecession.arm.slope1,2)
        for j = 1:size(phaseprecession.arm.slope1,3)
            if size(phaseprecession.arm.slope1,3) < 2
                phaseprecession.arm.slope1(find(abs(phaseprecession.arm.slope1)>0.2)) = 0;
                plot(abs(phaseprecession.arm.slope1))
            else
                phaseprecession.arm.slope1(find(abs(phaseprecession.arm.slope1(:,i,j))>0.2),i,j) = 0;
                plot(abs(phaseprecession.arm.slope1(:,i,j)))
            end
            
        end
    end
end
title('Phase precession')
ax = gca;
ax.XTick = ([1 2 3]);
ax.XTickLabels = ({'Pre-cooling','Cooling','Post-coolig'});

% Trailwise correlations - Central Arm
units_pos_trials2 = [];
for i = 1:size(units,2)
    for ii = 1:size(units_pos_trials{1},2)
        % units_pos_trials2{i}(:,ii) = conv(units_pos_trials{i}(:,ii),ones(1,10),'same');
        temp4 = conv(units_pos_trials{i}(:,ii),ones(1,10));
        units_pos_trials2{i}(:,ii) = temp4(1:end-9);
    end
    subplot_Peter(5,3,i,1)
    %subplot(size(units.ts,2),3,1+(i-1)*3)
    imagesc(units_pos_trials{i}), hold on, set(gca,'YDir','normal')
    plot([0,size(units_pos_trials2{i},1)],[find(diff(trials.cooling));find(diff(trials.cooling))],'w')
    if mod(i,5) == 1, title('Trialwise spiking'), end
    ylabel(['Unit ' num2str(i)])
    subplot_Peter(5,3,i,2)
    % subplot(size(units.ts,2),3,2+(i-1)*3)
    imagesc(units_pos_trials2{i}), hold on, set(gca,'YDir','normal')
    plot([0,size(units_pos_trials2{i},1)],[find(diff(trials.cooling));find(diff(trials.cooling))],'w')
    if mod(i,5) == 1, title('Central Arm'), end
    if mod(i,5) == 0 | i == size(units,2); xlabel('Position (5 cm bins)'); end 
    subplot_Peter(5,3,i,3)
    % subplot(size(units.ts,2),3,3+(i-1)*3)
    trial_corr = corrcoef(units_pos_trials2{i}');
    imagesc(trial_corr), hold on, set(gca,'YDir','normal')
    plot([0,size(units_pos_trials2{i},1)],[find(diff(trials.cooling));find(diff(trials.cooling))],'w')
    plot([find(diff(trials.cooling));find(diff(trials.cooling))],[0,size(units_pos_trials2{i},1)],'w')
    if mod(i,5) == 1, title('Correlations'), end
    if mod(i,5) == 0 | i == size(units,2); xlabel('Trials'); end 
end

% Trailwise correlations - RIM
test = find(diff(trials.cooling));
shift(1) = test(1) - sum(trials.error<test(1)+1);
shift(2) = test(2) + sum(trials.error<test(2)+1);
units_theta_trials2 = [];
for i = 1:size(units,2)
    for ii = 1:size(units_theta_trials{1},2)
        % units_theta_trials2{i}(:,ii) = conv(units_theta_trials{i}(:,ii),ones(1,10),'same');
        temp3 = conv(units_theta_trials{i}(:,ii),ones(1,10));
        units_theta_trials2{i}(:,ii) = temp3(1:end-9);
    end
    subplot_Peter(5,3,i,1)
    % subplot(size(units.ts,2),3,1+(i-1)*3)
    imagesc(units_theta_trials{i}), hold on, set(gca,'YDir','normal')
    plot([0,size(units_theta_trials2{i},1)],[shift;shift],'w')
    if mod(i,5) == 1, title('Trialwise spiking'), end
    ylabel(['Unit ' num2str(i)])
    subplot_Peter(5,3,i,2)
    % subplot(size(units.ts,2),3,2+(i-1)*3)
    imagesc(units_theta_trials2{i}), hold on, set(gca,'YDir','normal')
    plot([0,size(units_theta_trials2{i},1)],[shift;shift],'w')
    if mod(i,5) == 1, title('RIM'), end
    if mod(i,5) == 0 | i == size(units,2); xlabel('Position (5 cm bins)'); end 
    subplot_Peter(5,3,i,3)
    % subplot(size(units.ts,2),3,3+(i-1)*3)
    trial_corr = corrcoef(units_theta_trials2{i}');
    imagesc(trial_corr), hold on, set(gca,'YDir','normal')
    plot([0,size(units_theta_trials2{i},1)],[shift;shift],'w')
    plot([shift;shift],[0,size(units_theta_trials2{i},1)],'w')
    if mod(i,5) == 1, title('Correlations'), end
    if mod(i,5) == 0 | i == size(units,2); xlabel('Trials'); end 
end