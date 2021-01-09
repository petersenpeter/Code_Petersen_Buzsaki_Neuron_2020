function [y1,y2,y3,y4,time1,time2,time3,time4] = plot_ThetaVsAcceleration2(session,animal,cooling,speeed_threshold,interval)
disp('Plotting Theta vs running acceleration')
bins_acceleration = [-200:5:200];
%bins_acceleration = [-75:5:75];
bins_acceleration_avg = mean([bins_acceleration(2:end);bins_acceleration(1:end-1)]);
freqlist = [5:0.025:12]; %10.^(0.4771:0.01:1.1761);

if ~exist([session.general.name, '.lfp'])
    disp('Creating lfp file')
    downsample_dat_to_eeg(session.general.name,pwd);
end
signal = 0.000050354 * double(LoadBinary([session.general.name '.lfp'],'nChannels',session.extracellular.nChannels,'channels',session.channelTags.Theta.channels,'precision','int16','frequency',session.extracellular.srLfp)); % ,'start',start,'duration',duration
sr_theta = animal.sr;
signal2 = resample(signal,sr_theta,session.extracellular.srLfp);
Fpass = [1,49];
% if recording.sr_lfp < 100
%     Fpass = [1,14.9];
% end
Wn_theta = [Fpass(1)/(sr_theta/2) Fpass(2)/(sr_theta/2)]; % normalized by the nyquist frequency
[btheta,atheta] = butter(3,Wn_theta);
signal_filtered = filtfilt(btheta,atheta,signal2);
signal_filtered2 = signal_filtered;
signal_filtered = signal_filtered(interval(1)*sr_theta:interval(2)*sr_theta);
time_signal = (interval(1)*sr_theta:interval(2)*sr_theta)/sr_theta;
%[wt,~,~] = awt_freqlist(signal_filtered,sr_temperature,freqlist);
%wt2 = abs(wt)'; clear wt
wt = spectrogram(signal_filtered,100,99,freqlist,sr_theta);
wt2 = [zeros(length(freqlist),49),abs(wt), zeros(length(freqlist),50)]; clear wt

acceleration = interp1(animal.time,animal.acceleration2,time_signal);
speed = interp1(animal.time,animal.speed2,time_signal);
pos = interp1(animal.time,animal.pos',time_signal);
%temperature = interp1(animal.time,animal.temperature',(1:length(signal_filtered))/sr_theta);

% t_cooling = zeros(1,length(signal_filtered)); t_nocooling = zeros(1,length(signal_filtered));
% for i = 1:size(cooling.cooling,2), t_cooling(cooling.cooling(1,i)*sr_theta:cooling.cooling(2,i)*sr_theta) = 1; end
% for i = 1:size(cooling.nocooling,2), t_nocooling(cooling.nocooling(1,i)*sr_theta+1:cooling.nocooling(2,i)*sr_theta) = 1; end
% t_cooling = t_cooling(1:length(signal_filtered));

t_cooling = zeros(1,length(signal_filtered2)); t_nocooling = zeros(1,length(signal_filtered2));
for i = 1:size(cooling.cooling,2), t_cooling(cooling.cooling(1,i)*sr_theta:cooling.cooling(2,i)*sr_theta) = 1; end
for i = 1:size(cooling.nocooling,2), t_nocooling(cooling.nocooling(1,i)*sr_theta+1:cooling.nocooling(2,i)*sr_theta) = 1; end
t_cooling = t_cooling(interval(1)*sr_theta:interval(2)*sr_theta);
t_nocooling = t_nocooling(interval(1)*sr_theta:interval(2)*sr_theta);

wt_cooling = [];
wt_acceleration_cooling = [];
thetamax_cooling = [];
theta_maxpower_cooling = [];
for j = 1:length(bins_acceleration)-1
    indexes = find(acceleration > bins_acceleration(j) & acceleration < bins_acceleration(j+1) & t_cooling > 0 & speed > speeed_threshold); 
    wt_cooling{j} = wt2(:,indexes);
    wt_acceleration_cooling(j,:) = mean(wt2(:,indexes),2);
    [theta_maxpower_cooling(j),thetamax_cooling(j)] = max(mean(wt2(:,indexes),2));
end

wt_nocooling = [];
wt_acceleration_nocooling = [];
thetamax_nocooling = [];
theta_maxpower_nocooling = [];
for j = 1:length(bins_acceleration)-1
    indexes = find(acceleration > bins_acceleration(j) & acceleration < bins_acceleration(j+1) & t_nocooling > 0 & speed > speeed_threshold); 
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
time1 = bins_acceleration_avg(find(thetamax_cooling>2));
y1 = freqlist(thetamax_cooling(find(thetamax_cooling>2)));
plot(time1,y1,'ob'), hold on

time2 = bins_acceleration_avg(find(thetamax_nocooling>2));
y2 = freqlist(thetamax_nocooling(find(thetamax_nocooling>2)));
plot(time2,y2,'or'), hold on
xlim([bins_acceleration(1),bins_acceleration(end)]), ylim([6,12]),% ylim([y1(1)-0.4,y2(end)+0.4]),
xlabel('Acceleration (cm/s^2)'),ylabel('Frequency (Hz)'), grid on, title('Peak frequency of theta')

% Power
subplot(2,3,6)
time3 = bins_acceleration_avg(find(thetamax_cooling>2));
y3 = theta_maxpower_cooling(find(thetamax_cooling>2));
plot(time3,y3,'ob'), hold on

time4 = bins_acceleration_avg(find(thetamax_nocooling>2));
y4 = theta_maxpower_nocooling(find(thetamax_nocooling>2));
plot(time4,y4,'or'), hold on
xlim([bins_acceleration(1),bins_acceleration(end)]), %ylim([y1(1)-0.4,y2(end)+0.4]),
xlabel('Acceleration (cm/s^2)'),ylabel('Peak Power'), grid on, title('Amplitude theta'),xlim([bins_acceleration(1),bins_acceleration(end)])


