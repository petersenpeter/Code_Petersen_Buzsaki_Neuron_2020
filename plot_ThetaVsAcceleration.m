function plot_ThetaVsAcceleration(recording,animal,cooling,speeed_threshold)
disp('Plotting Theta vs running acceleration')
bins_acceleration = [-1.2:0.02:1.2];
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
for j = 1:length(bins_acceleration)-1
    indexes = find(acceleration > bins_acceleration(j) & acceleration < bins_acceleration(j+1) & t_cooling > 0 & speed>speeed_threshold); % & pos > track_boundaries(1) & pos > track_boundaries(2)
    wt_cooling{j} = wt2(:,indexes);
    wt_acceleration_cooling(j,:) = mean(wt2(:,indexes),2);
    [theta_maxpower_cooling(j),thetamax_cooling(j)] = max(mean(wt2(:,indexes),2));
end

wt_nocooling = [];
wt_acceleration_nocooling = [];
thetamax_nocooling = [];
theta_maxpower_nocooling = [];
for j = 1:length(bins_acceleration)-1
    indexes = find(acceleration > bins_acceleration(j) & acceleration < bins_acceleration(j+1) & t_nocooling > 0 & speed>speeed_threshold); % & pos > track_boundaries(1) & pos > track_boundaries(2)
    wt_nocooling{j} = wt2(:,indexes);
    wt_acceleration_nocooling(j,:) = mean(wt2(:,indexes),2);
    [theta_maxpower_nocooling(j),thetamax_nocooling(j)] = max(mean(wt2(:,indexes),2));
end

figure
subplot(2,2,1)
% surf(bins_lfp,bins_acceleration_avg,lfp_wavelets_combined,'EdgeColor','None'), view(0,90)
imagesc(bins_acceleration_avg,freqlist,wt_acceleration_nocooling'), set(gca,'Ydir','normal')
axis tight, title(['Normal brain temperature, speed>' num2str(speeed_threshold)]), ylabel('Powerspectrum (Hz)'),xlabel('Acceleration (cm/s^2)')
set(gca,'YTick',bins_acceleration), set(gca,'YTick',freqlist(1):freqlist(end)), %set(gca,'xscale','log')
colorbar,%caxis([0 cmax])
subplot(2,2,2)
% surf(bins_lfp,bins_acceleration_avg,lfp_wavelets_combined,'EdgeColor','None'), view(0,90)
imagesc(bins_acceleration_avg,freqlist,wt_acceleration_cooling'), set(gca,'Ydir','normal')
axis tight, title(['With Cooling, speed>' num2str(speeed_threshold)]), ylabel('Powerspectrum (Hz)'),xlabel('Acceleration (cm/s^2)')
set(gca,'YTick',bins_acceleration), set(gca,'YTick',freqlist(1):freqlist(end)), %set(gca,'xscale','log')
cmax = max(max(wt_acceleration_cooling(:)',wt_acceleration_nocooling(:)'));
colorbar,%caxis([0 cmax])

subplot(2,3,4)
stairs(freqlist,mean(wt_acceleration_cooling),'b'), hold on
stairs(freqlist,mean(wt_acceleration_nocooling),'r'), hold on
xlabel('Frequency (Hz)'), grid on, title('Average frequency'),xlim([freqlist(1),freqlist(end)]),ylabel('Power'),%ylim([0,max(max(mean(wt_acceleration_nocooling),mean(wt_acceleration_cooling)))])

subplot(2,3,5)
time = bins_acceleration_avg(find(thetamax_cooling>2));
y1 = freqlist(thetamax_cooling(find(thetamax_cooling>2)));
plot(time,y1,'ob'), hold on
% P = polyfit(time,y1,1);
% yfit = P(1)*bins_acceleration+P(2);
% plot(bins_acceleration,yfit,'b-');
% text(15,y1(1)+0.2,['Cooling slope: ', num2str(P(1),3)],'Color','blue')
% [r,p] = corr(time',y1')

time = bins_acceleration_avg(find(thetamax_nocooling>2));
y2 = freqlist(thetamax_nocooling(find(thetamax_nocooling>2)));
plot(time,y2,'or'), hold on
% P = polyfit(time,y2,1);
% yfit = P(1)*bins_acceleration+P(2);
% plot(bins_acceleration,yfit,'r-');
% text(15,y2(10)+0.4,['No Cooling slope: ', num2str(P(1),3)],'Color','red')
% [r,p] = corr(time',y2');
xlim([bins_acceleration(1),bins_acceleration(end)]), ylim([6,12]),% ylim([y1(1)-0.4,y2(end)+0.4]),
xlabel('Acceleration (cm/s^2)'),ylabel('Frequency (Hz)'), grid on, title('Peak frequency of theta')

% Power
subplot(2,3,6)
time = bins_acceleration_avg(find(thetamax_cooling>2));
y1 = theta_maxpower_cooling(find(thetamax_cooling>2));
plot(time,y1,'ob'), hold on
% P = polyfit(time,y1,1);
% yfit = P(1)*bins_acceleration+P(2);
% plot(bins_acceleration,yfit,'b-');
% text(15,y1(1)-0.1,['Cooling slope: ', num2str(P(1),3)],'Color','blue')
% [r,p] = corr(time',y1')

time = bins_acceleration_avg(find(thetamax_nocooling>2));
y2 = theta_maxpower_nocooling(find(thetamax_nocooling>2));
plot(time,y2,'or'), hold on
% P = polyfit(time,y2,1);
% yfit = P(1)*bins_acceleration+P(2);
% plot(bins_acceleration,yfit,'r-');
% text(15,y2(1)+0.2,['No Cooling slope: ', num2str(P(1),3)],'Color','red')
% [r,p] = corr(time',y2')
xlim([bins_acceleration(1),bins_acceleration(end)]), %ylim([y1(1)-0.4,y2(end)+0.4]),
xlabel('Acceleration (cm/s^2)'),ylabel('Peak Power'), grid on, title('Amplitude theta'),xlim([bins_acceleration(1),bins_acceleration(end)])

% % GLM fit
% disp('performing a GLM fit')
% if isfield(animal,'temperature')
%     running_times = find(acceleration>5);
%     X = [temperature(running_times)+37+60; acceleration(running_times); [diff(acceleration(running_times)),0]]';
%     [temp_theta_maxpower,temp_theta_max] = max(wt2(:,(running_times)));
%     y1 = freqlist(temp_theta_max)'; % Frequency
%     y2 = temp_theta_maxpower'; % Power
%     mdl1 = fitglm(X,y1)
%     mdl2 = fitglm(X,y2)
%     theta_bins = [freqlist(1):0.1:freqlist(end)];
%     hist_theta = hist2d([y1';(mdl1.predict)']',theta_bins,theta_bins);
%     figure, imagesc(theta_bins,theta_bins,hist_theta), xlabel('Real theta'), ylabel('Predicted theta'),set(gca,'Ydir','normal')
% end
