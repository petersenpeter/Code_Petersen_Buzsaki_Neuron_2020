function plot_ThetaVsAcceleration(recording,animal,cooling)
disp('Plotting Theta vs running acceleration')
speeed_threshold = 30;
bins_speed = [5:5:150];
bins_acceleration = [-1.8:0.1:1.8];
%bins_acceleration = [-75:5:75];
bins_acceleration_avg = mean([bins_acceleration(2:end);bins_acceleration(1:end-1)]);
freqlist = [5:0.025:12]; %10.^(0.4771:0.01:1.1761);
if ~exist([recording.name, '.lfp'])
    disp('Creating EEG file')
    downsample_dat_to_eeg(recording.name,pwd);
end
signal = 0.000050354 * double(LoadBinary([recording.name '.lfp'],'nChannels',recording.nChannels,'channels',recording.ch_theta,'precision','int16','frequency',recording.sr_lfp)); % ,'start',start,'duration',duration
sr_theta = animal.sr;
signal2 = resample(signal,sr_theta,recording.sr_lfp);
Fpass = [1,49];
% if recording.sr_lfp < 100
%     Fpass = [1,14.9];
% end
Wn_theta = [Fpass(1)/(sr_theta/2) Fpass(2)/(sr_theta/2)]; % normalized by the nyquist frequency
[btheta,atheta] = butter(3,Wn_theta);
signal_filtered = filtfilt(btheta,atheta,signal2);
%[wt,~,~] = awt_freqlist(signal_filtered,sr_temperature,freqlist);
%wt2 = abs(wt)'; clear wt
wt = spectrogram(signal_filtered,100,99,freqlist,sr_theta);
wt2 = [zeros(length(freqlist),49),abs(wt), zeros(length(freqlist),50)]; clear wt

acceleration = interp1(animal.time,animal.acceleration2,(1:length(signal_filtered))/sr_theta);
speed = interp1(animal.time,animal.speed2,(1:length(signal_filtered))/sr_theta);
pos = interp1(animal.time,animal.pos',(1:length(signal_filtered))/sr_theta);
%temperature = interp1(animal.time,animal.temperature',(1:length(signal_filtered))/sr_theta);

t_cooling = zeros(1,length(signal_filtered)); t_nocooling = zeros(1,length(signal_filtered));
for i = 1:size(cooling.cooling,2), t_cooling(cooling.cooling(1,i)*sr_theta:cooling.cooling(2,i)*sr_theta) = 1; end
for i = 1:size(cooling.nocooling,2), t_nocooling(cooling.nocooling(1,i)*sr_theta+1:cooling.nocooling(2,i)*sr_theta) = 1; end
t_cooling = t_cooling(1:length(signal_filtered));
wt_cooling = [];
wt_acceleration_cooling = [];
thetamax_cooling = [];
theta_maxpower_cooling = [];
for jj = 1:length(bins_speed)-1
    for j = 1:length(bins_acceleration)-1
        indexes = find(speed > bins_speed(jj) & speed < bins_speed(jj+1) & acceleration > bins_acceleration(j) & acceleration < bins_acceleration(j+1) & t_cooling > 0);
        wt_cooling{j} = wt2(:,indexes);
        wt_acceleration_cooling(j,:) = mean(wt2(:,indexes),2);
        [theta_maxpower_cooling(jj,j),thetamax_cooling(jj,j)] = max(mean(wt2(:,indexes),2));
    end
end

wt_nocooling = [];
wt_acceleration_nocooling = [];
thetamax_nocooling = [];
theta_maxpower_nocooling = [];
for jj = 1:length(bins_speed)-1
    for j = 1:length(bins_acceleration)-1
        indexes = find(speed > bins_speed(jj) & speed < bins_speed(jj+1) & acceleration > bins_acceleration(j) & acceleration < bins_acceleration(j+1) & t_nocooling > 0);
        %     indexes = find(acceleration > bins_acceleration(j) & acceleration < bins_acceleration(j+1) & t_nocooling > 0 & speed>speeed_threshold); % & pos > track_boundaries(1) & pos > track_boundaries(2)
        wt_nocooling{j} = wt2(:,indexes);
        wt_acceleration_nocooling(j,:) = mean(wt2(:,indexes),2);
        [theta_maxpower_nocooling(jj,j),thetamax_nocooling(jj,j)] = max(mean(wt2(:,indexes),2));
    end
end

figure
subplot(3,2,1)
% surf(bins_lfp,bins_acceleration_avg,lfp_wavelets_combined,'EdgeColor','None'), view(0,90)
imagesc(bins_acceleration_avg,freqlist,wt_acceleration_nocooling'), set(gca,'Ydir','normal')
axis tight, title(['Normal brain temperature, speed>' num2str(speeed_threshold)]), ylabel('Powerspectrum (Hz)'),xlabel('Acceleration (cm/s^2)')
set(gca,'YTick',bins_acceleration), set(gca,'YTick',freqlist(1):freqlist(end)), %set(gca,'xscale','log')
colorbar,%caxis([0 cmax])
subplot(3,2,2)
% surf(bins_lfp,bins_acceleration_avg,lfp_wavelets_combined,'EdgeColor','None'), view(0,90)
imagesc(bins_acceleration_avg,freqlist,wt_acceleration_cooling'), set(gca,'Ydir','normal')
axis tight, title(['With Cooling, speed>' num2str(speeed_threshold)]), ylabel('Powerspectrum (Hz)'),xlabel('Acceleration (cm/s^2)')
set(gca,'YTick',bins_acceleration), set(gca,'YTick',freqlist(1):freqlist(end)), %set(gca,'xscale','log')
cmax = max(max(wt_acceleration_cooling(:)',wt_acceleration_nocooling(:)'));
colorbar,%caxis([0 cmax])

subplot(3,3,4)
stairs(freqlist,mean(wt_acceleration_cooling),'b'), hold on
stairs(freqlist,mean(wt_acceleration_nocooling),'r'), hold on
xlabel('Frequency (Hz)'), grid on, title('Average frequency'),xlim([freqlist(1),freqlist(end)]),ylabel('Power'),%ylim([0,max(max(mean(wt_acceleration_nocooling),mean(wt_acceleration_cooling)))])

% Frequency
subplot(2,2,1)
y2 = freqlist(thetamax_nocooling); 
imagesc(bins_acceleration(1:end-1),bins_speed(1:end-1),y2), hold on
title('Peak frequency of theta: NoCooling')
xlabel('Acceleration (cm/s^2)'), xlabel('Acceleration (cm/s^2)'), ylabel('Speed (cm/s)'), colorbar, clim([5.,9.6])

subplot(2,2,2)
y1 = freqlist(thetamax_cooling); 
imagesc(bins_acceleration(1:end-1),bins_speed(1:end-1),y1), hold on
title('Peak frequency of theta: Cooling'), xlabel('Acceleration (cm/s^2)'), ylabel('Speed (cm/s)'), colorbar, clim([5.,9.6])

% Power
subplot(2,2,3)
y4 = theta_maxpower_nocooling;
imagesc(bins_acceleration(1:end-1),bins_speed(1:end-1),y4), hold on
title('Amplitude theta: NoCooling2'), xlabel('Acceleration (cm/s^2)'), ylabel('Speed (cm/s)'), colorbar, clim([0.6,1.5])

subplot(2,2,4)
y3 = theta_maxpower_cooling;
imagesc(bins_acceleration(1:end-1),bins_speed(1:end-1),y3), hold on
title('Amplitude theta: Cooling'), xlabel('Acceleration (cm/s^2)'), ylabel('Speed (cm/s)'), colorbar, clim([0.6,1.5])
