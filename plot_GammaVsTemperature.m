function stats = plot_GammaVsTemperature(recording,animal,cooling)
disp('Plotting Gamma vs Temperature')
if ~exist([recording.name, '.lfp'])
    disp('Creating lfp file')
    downsample_dat_to_eeg(recording.name,pwd);
end

signal = 0.000050354 * double(LoadBinary([recording.name '.lfp'],'nChannels',recording.nChannels,'channels',recording.ch_theta,'precision','int16','frequency',recording.sr_lfp)); % ,'start',start,'duration',duration

freqband = 'gamma';
switch freqband
        case 'theta'
            freqlist = [4:0.025:10];
            Fpass = [4,10];
            sr_theta = animal.sr;
            caxis_range = [0,1.];
        case 'gamma'
            freqlist = [20:2:100];
            Fpass = [15,140];
            sr_theta = 400;
            caxis_range = [0,0.45];
end
bins_speed = [30:10:100];
bins_speed_avg = mean([bins_speed(2:end);bins_speed(1:end-1)]);

bins_temp = floor(min(animal.temperature)):0.5:ceil(max(animal.temperature));
bins_temp_avg = mean([bins_temp(2:end);bins_temp(1:end-1)]);

running_window = sr_theta*3/4;

signal2 = resample(signal,sr_theta,recording.sr_lfp);

d = designfilt('bandstopiir','FilterOrder',2, ...
               'HalfPowerFrequency1',59,'HalfPowerFrequency2',61, ...
               'DesignMethod','butter','SampleRate',sr_theta);
signal2 = filtfilt(d,signal2);

clear signal
theta_samples_pre = 2*sr_theta;
theta_samples_post = 2*sr_theta;
window_time = [-theta_samples_pre:theta_samples_post]/sr_theta;
window_stim = [theta_samples_pre-sr_theta/2:theta_samples_pre+sr_theta/2];
window_prestim = [1:theta_samples_pre-sr_theta/2-1];
Wn_theta = [Fpass(1)/(sr_theta/2) Fpass(2)/(sr_theta/2)]; % normalized by the nyquist frequency
[btheta,atheta] = butter(3,Wn_theta);

signal_filtered = filtfilt(btheta,atheta,signal2);
clear signal2
t_cooling = zeros(1,length(signal_filtered)); t_nocooling = zeros(1,length(signal_filtered));
speed = interp1(animal.time,animal.speed,(1:length(signal_filtered))/sr_theta);
temp = interp1(animal.time,animal.temperature',(1:length(signal_filtered))/sr_theta);

filter_time = [min(find(~isnan(speed)==1)):max(find(~isnan(speed)==1))];
signal_filtered = signal_filtered(filter_time);
speed = speed(filter_time);
acceleration = [diff(speed),0];
temp = temp(filter_time);

wt2 = spectrogram(signal_filtered,running_window,running_window-1,freqlist,sr_theta);
wt2 = [zeros(length(freqlist),running_window/2-1),abs(wt2), zeros(length(freqlist),running_window/2)]; clear wt

for i = 1:size(cooling.cooling,2), t_cooling(cooling.cooling(1,i)*sr_theta:cooling.cooling(2,i)*sr_theta) = 1; end
for i = 1:size(cooling.nocooling,2), t_nocooling(max(cooling.nocooling(1,i)*sr_theta,1):cooling.nocooling(2,i)*sr_theta) = 1; end
t_cooling = t_cooling(filter_time);
t_nocooling = t_nocooling(filter_time);
wt_cooling = [];
wt_speed_cooling = [];
thetamax_cooling = [];
theta_maxpower_cooling = [];
for j = 1:length(bins_speed)-1
    indexes = find(speed > bins_speed(j) & speed < bins_speed(j+1) & t_cooling > 0); % & pos > track_boundaries(1) & pos > track_boundaries(2)
    wt_cooling{j} = wt2(:,indexes);
    wt_speed_cooling(j,:) = mean(wt2(:,indexes),2);
    [theta_maxpower_cooling(j),thetamax_cooling(j)] = max(mean(wt2(:,indexes),2));
    theta_maxpower_cooling(j) = mean(mean(wt2(:,indexes),2));
    [~,thetamax_cooling(j)] = max(nanconv(mean(wt2(:,indexes),2),[1:4,3:-1:1]'/sum([1:4,3:-1:1]),'edge'));
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
    theta_maxpower_nocooling(j) = mean(mean(wt2(:,indexes),2));
    [~,thetamax_nocooling(j)] = max(nanconv(mean(wt2(:,indexes),2),[1:4,3:-1:1]'/sum([1:4,3:-1:1]),'edge'));
end

figure
subplot(2,2,1)
% surf(bins_lfp,bins_speed_avg,lfp_wavelets_combined,'EdgeColor','None'), view(0,90)
imagesc(bins_speed_avg,freqlist,wt_speed_cooling'), set(gca,'Ydir','normal')
axis tight, title('With Cooling'), ylabel('Powerspectrum (Hz)'),xlabel('Speed (cm/s)')
%set(gca,'XTick',bins_speed), set(gca,'YTick',freqlist(1):freqlist(end)), %set(gca,'xscale','log')
cmax = max(max(wt_speed_cooling(:)',wt_speed_nocooling(:)'));
colorbar,caxis([0 cmax])
subplot(2,2,2)
% surf(bins_lfp,bins_speed_avg,lfp_wavelets_combined,'EdgeColor','None'), view(0,90)
imagesc(bins_speed_avg,freqlist,wt_speed_nocooling'), set(gca,'Ydir','normal')
axis tight, title(['Without Cooling']), ylabel('Powerspectrum (Hz)'),xlabel('Speed (cm/s)')
% set(gca,'YTick',bins_speed), set(gca,'YTick',freqlist(1):freqlist(end)), %set(gca,'xscale','log')
colorbar,caxis([0 cmax])
subplot(2,3,4)
plot(freqlist,mean(wt_speed_cooling),'b'), hold on
plot(freqlist,mean(wt_speed_nocooling),'r'), hold on
xlabel('Frequency (Hz)'), grid on, title('Average frequency'),xlim([freqlist(1),freqlist(end)]),ylabel('Power'),%ylim([0,max(max(mean(wt_speed_nocooling),mean(wt_speed_cooling)))])

stats.freq_cooling = wt_speed_cooling;
stats.freq_nocooling = wt_speed_nocooling;
stats.freqlist = freqlist;
subplot(2,3,5)
time = bins_speed_avg(find(thetamax_cooling>2));
y1 = freqlist(thetamax_cooling(find(thetamax_cooling>2)));
plot(time,y1,'ob'), hold on
P = polyfit(time,y1,1);
yfit = P(1)*bins_speed+P(2);
plot(bins_speed,yfit,'b-');
% text(15,y1(1)+0.2,['Cooling slope: ', num2str(P(1),3)],'Color','blue')
% [r,p] = corr(time',y1')

time = bins_speed_avg(find(thetamax_nocooling>2));
y2 = freqlist(thetamax_nocooling(find(thetamax_nocooling>2)));
plot(time,y2,'or'), hold on
P = polyfit(time,y2,1);
yfit = P(1)*bins_speed+P(2);
plot(bins_speed,yfit,'r-');
% text(15,y2(10)+0.4,['No Cooling slope: ', num2str(P(1),3)],'Color','red')
% [r,p] = corr(time',y2');
%xlim([bins_speed(1),bins_speed(end)]), %ylim([y1(1)-0.4,y2(end)+0.4]),
xlabel('Speed (cm/s)'),ylabel('Frequency (Hz)'), grid on, title('Peak frequency')

% Power
subplot(2,3,6)
time = bins_speed_avg(find(thetamax_cooling>2));
y1 = theta_maxpower_cooling(find(thetamax_cooling>2));
plot(time,y1,'ob'), hold on
P = polyfit(time,y1,1);
yfit = P(1)*bins_speed+P(2);
plot(bins_speed,yfit,'b-');
% text(15,y1(1)-0.1,['Cooling slope: ', num2str(P(1),3)],'Color','blue')
% [r,p] = corr(time',y1')

time = bins_speed_avg(find(thetamax_nocooling>2));
y2 = theta_maxpower_nocooling(find(thetamax_nocooling>2));
plot(time,y2,'or'), hold on
P = polyfit(time,y2,1);
yfit = P(1)*bins_speed+P(2);
plot(bins_speed,yfit,'r-');
% text(15,y2(1)+0.2,['No Cooling slope: ', num2str(P(1),3)],'Color','red')
% [r,p] = corr(time',y2')
xlim([bins_speed(1),bins_speed(end)]), % ylim([y1(1)-0.4,y2(end)+0.4]),
xlabel('Speed (cm/s)'),ylabel('Power'), grid on, title('Amplitude'),xlim([bins_speed(1),bins_speed(end)])

stats.amplitude_Cooling = y1;
stats.amplitude_NoCooling = y2;
stats.band = freqband;
print(['GammaVsTemperature_Ch' num2str(recording.ch_theta)],'-dpdf')

% % 
% % % Theta/Gamma vs Temperature
% % wt_cooling = [];
% % wt_temp_cooling = [];
% % thetamax_cooling = [];
% % theta_maxpower_cooling = [];
% % for j = 1:length(bins_temp)-1
% %     indexes = find(temp > bins_temp(j) & temp < bins_temp(j+1) & speed > bins_speed(1)); % & pos > track_boundaries(1) & pos > track_boundaries(2)
% %     wt_cooling{j} = wt2(:,indexes);
% %     wt_temp_cooling(j,:) = mean(wt2(:,indexes),2);
% %     theta_maxpower_cooling(j) = mean(mean(wt2(:,indexes),2));
% %     [~,thetamax_cooling(j)] = max(nanconv(mean(wt2(:,indexes),2),[1:4,3:-1:1]'/sum([1:4,3:-1:1]),'edge'));
% % end
% % 
% % figure
% % subplot(2,2,1)
% % % surf(bins_lfp,bins_speed_avg,lfp_wavelets_combined,'EdgeColor','None'), view(0,90)
% % imagesc(bins_temp_avg,freqlist,wt_temp_cooling'), set(gca,'Ydir','normal')
% % axis tight, title('With Cooling'), ylabel('Powerspectrum (Hz)'),xlabel('Temperature (C)')
% % %set(gca,'YTick',bins_speed), set(gca,'YTick',freqlist(1):freqlist(end)), %set(gca,'xscale','log')
% % cmax = max(max(wt_temp_cooling));
% % colorbar,%caxis([0 cmax])
% % 
% % subplot(2,2,2)
% % if ~isempty(wt_temp_cooling)
% %     stairs(freqlist,mean(wt_temp_cooling),'b'), hold on
% % end
% % xlabel('Frequency (Hz)'), grid on, title('Average frequency'),xlim([freqlist(1),freqlist(end)]),ylabel('Power'),%ylim([0,max(mean(wt_temp_cooling))])
% % 
% % % Frequency
% % subplot(2,2,3)
% % time = bins_temp_avg(find(thetamax_cooling>2));
% % y1 = freqlist(thetamax_cooling(find(thetamax_cooling>2)));
% % plot(time,y1,'ob'), hold on
% % P = polyfit(time,y1,1);
% % yfit = P(1)*bins_temp+P(2);
% % plot(bins_temp,yfit,'b-');
% % text(time(5),y1(1)+0.2,['Cooling slope: ', num2str(P(1),3)],'Color','blue')
% % [r,p] = corr(time',y1')
% % xlabel('Temperature (C)'),ylabel('Frequency (Hz)'), grid on, title('Frequency'),xlim([bins_temp(1),bins_temp(end)]),%ylim([freqlist(1),freqlist(end)])
% % 
% % % Power
% % subplot(2,2,4)
% % time = bins_temp_avg;
% % y1 = theta_maxpower_cooling;
% % plot(time,y1,'ob'), hold on
% % P = polyfit(time,y1,1);
% % yfit = P(1)*bins_temp+P(2);
% % plot(bins_temp,yfit,'b-');
% % text(time(5),y1(1),['Cooling slope: ', num2str(P(1),3)],'Color','blue')
% % [r,p] = corr(time',y1')
% % xlabel('Temperature (C)'),ylabel('Power'), grid on, title('Amplitude'),xlim([bins_temp(1),bins_temp(end)])

% %% Acceleration 
% bins_speed = [-10:0.5:10];
% bins_speed_avg = mean([bins_speed(2:end);bins_speed(1:end-1)]);
% speed = acceleration;
% wt_cooling = [];
% wt_speed_cooling = [];
% thetamax_cooling = [];
% theta_maxpower_cooling = [];
% for j = 1:length(bins_speed)-1
%     indexes = find(speed > bins_speed(j) & speed < bins_speed(j+1) & t_cooling > 0); % & pos > track_boundaries(1) & pos > track_boundaries(2)
%     wt_cooling{j} = wt2(:,indexes);
%     wt_speed_cooling(j,:) = mean(wt2(:,indexes),2);
%     [theta_maxpower_cooling(j),thetamax_cooling(j)] = max(mean(wt2(:,indexes),2));
%     theta_maxpower_cooling(j) = mean(mean(wt2(:,indexes),2));
%     [~,thetamax_cooling(j)] = max(nanconv(mean(wt2(:,indexes),2),[1:4,3:-1:1]'/sum([1:4,3:-1:1]),'edge'));
% end
% 
% wt_nocooling = [];
% wt_speed_nocooling = [];
% thetamax_nocooling = [];
% theta_maxpower_nocooling = [];
% for j = 1:length(bins_speed)-1
%     indexes = find(speed > bins_speed(j) & speed < bins_speed(j+1) & t_nocooling > 0 ); % & pos > track_boundaries(1) & pos > track_boundaries(2)
%     wt_nocooling{j} = wt2(:,indexes);
%     wt_speed_nocooling(j,:) = mean(wt2(:,indexes),2);
%     [theta_maxpower_nocooling(j),thetamax_nocooling(j)] = max(mean(wt2(:,indexes),2));
%     theta_maxpower_nocooling(j) = mean(mean(wt2(:,indexes),2));
%     [~,thetamax_nocooling(j)] = max(nanconv(mean(wt2(:,indexes),2),[1:4,3:-1:1]'/sum([1:4,3:-1:1]),'edge'));
% end
% 
% figure
% subplot(2,2,1)
% % surf(bins_lfp,bins_speed_avg,lfp_wavelets_combined,'EdgeColor','None'), view(0,90)
% imagesc(bins_speed_avg,freqlist,wt_speed_cooling'), set(gca,'Ydir','normal')
% axis tight, title('With Cooling'), ylabel('Powerspectrum (Hz)'),xlabel('Acceleration (cm/s^2)')
% %set(gca,'XTick',bins_speed), set(gca,'YTick',freqlist(1):freqlist(end)), %set(gca,'xscale','log')
% cmax = max(max(wt_speed_cooling(:)',wt_speed_nocooling(:)'));
% colorbar,caxis([0 cmax])
% subplot(2,2,2)
% % surf(bins_lfp,bins_speed_avg,lfp_wavelets_combined,'EdgeColor','None'), view(0,90)
% imagesc(bins_speed_avg,freqlist,wt_speed_nocooling'), set(gca,'Ydir','normal')
% axis tight, title(['Without Cooling']), ylabel('Powerspectrum (Hz)'),xlabel('Acceleration (cm/s^2)')
% % set(gca,'YTick',bins_speed), set(gca,'YTick',freqlist(1):freqlist(end)), %set(gca,'xscale','log')
% colorbar,caxis([0 cmax])
% subplot(2,3,4)
% stairs(freqlist,mean(wt_speed_cooling),'b'), hold on
% stairs(freqlist,mean(wt_speed_nocooling),'r'), hold on
% xlabel('Frequency (Hz)'), grid on, title('Average frequency'),xlim([freqlist(1),freqlist(end)]),ylabel('Power'),%ylim([0,max(max(mean(wt_speed_nocooling),mean(wt_speed_cooling)))])
% 
% subplot(2,3,5)
% time = bins_speed_avg(find(thetamax_cooling>2));
% y1 = freqlist(thetamax_cooling(find(thetamax_cooling>2)));
% plot(time,y1,'ob'), hold on
% P = polyfit(time,y1,1);
% yfit = P(1)*bins_speed+P(2);
% plot(bins_speed,yfit,'b-');
% text(bins_speed_avg(2),y1(1)+0.2,['Cooling slope: ', num2str(P(1),3)],'Color','blue')
% [r,p] = corr(time',y1')
% 
% time = bins_speed_avg(find(thetamax_nocooling>2));
% y2 = freqlist(thetamax_nocooling(find(thetamax_nocooling>2)));
% plot(time,y2,'or'), hold on
% P = polyfit(time,y2,1);
% yfit = P(1)*bins_speed+P(2);
% plot(bins_speed,yfit,'r-');
% text(bins_speed_avg(2),y2(10)+0.4,['No Cooling slope: ', num2str(P(1),3)],'Color','red')
% [r,p] = corr(time',y2');
% %xlim([bins_speed(1),bins_speed(end)]), %ylim([y1(1)-0.4,y2(end)+0.4]),
% xlabel('Acceleration (cm/s^2)'),ylabel('Frequency (Hz)'), grid on, title('Peak frequency')
% 
% % Power
% subplot(2,3,6)
% time = bins_speed_avg(find(thetamax_cooling>2));
% y1 = theta_maxpower_cooling(find(thetamax_cooling>2));
% plot(time,y1,'ob'), hold on
% P = polyfit(time,y1,1);
% yfit = P(1)*bins_speed+P(2);
% plot(bins_speed,yfit,'b-');
% text(bins_speed_avg(2),y1(1)-0.1,['Cooling slope: ', num2str(P(1),3)],'Color','blue')
% [r,p] = corr(time',y1')
% 
% time = bins_speed_avg(find(thetamax_nocooling>2));
% y2 = theta_maxpower_nocooling(find(thetamax_nocooling>2));
% plot(time,y2,'or'), hold on
% P = polyfit(time,y2,1);
% yfit = P(1)*bins_speed+P(2);
% plot(bins_speed,yfit,'r-');
% text(bins_speed_avg(2),y2(1)+0.2,['No Cooling slope: ', num2str(P(1),3)],'Color','red')
% [r,p] = corr(time',y2')
% xlim([bins_speed(1),bins_speed(end)]), ylim([y1(1)-0.4,y2(end)+0.4]),
% xlabel('Acceleration (cm/s^2)'),ylabel('Power'), grid on, title('Amplitude'),xlim([bins_speed(1),bins_speed(end)])
