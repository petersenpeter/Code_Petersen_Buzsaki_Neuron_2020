function thetaVsSpeed = plot_ThetaVsSpeed3(session,animal,cooling,interval,animal_speed_max)
disp('Plotting Theta vs running speed')
bins_speed = [10:3:animal_speed_max]; % [-40:5:40]; % [5:3:60]
bins_speed_avg = mean([bins_speed(2:end);bins_speed(1:end-1)]);
freqlist = [5.5:0.1:11]; %10.^(0.4771:0.01:1.1761);
if ~exist([session.general.name, '.lfp'])
    disp('Creating lfp file')
    downsample_dat_to_eeg(session.general.name,pwd);
end
signal = 0.000195 * double(LoadBinary([session.general.name '.lfp'],'nChannels',session.extracellular.nChannels,'channels',session.channelTags.Theta.channels,'precision','int16','frequency',session.extracellular.srLfp)); % ,'start',start,'duration',duration
sr_theta = animal.sr;
signal2 = resample(signal,sr_theta,session.extracellular.srLfp);
Fpass = [1,49];
if sr_theta < 101
    % Filtering the band 
    Fpass = [1,14.9];
end
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

speed = interp1(animal.time,animal.speed,time_signal);
% speed = interp1(animal.time,animal.speed,(1:length(signal_filtered))/sr_theta);
% pos = interp1(animal.time,animal.pos',(1:length(signal_filtered))/sr_theta);
%temperature = interp1(animal.time,animal.temperature',(1:length(signal_filtered))/sr_theta);

t_cooling = zeros(1,length(signal_filtered2)); t_nocooling = zeros(1,length(signal_filtered2));
for i = 1:size(cooling.cooling,2), t_cooling(cooling.cooling(1,i)*sr_theta:cooling.cooling(2,i)*sr_theta) = 1; end
for i = 1:size(cooling.nocooling,2), t_nocooling(cooling.nocooling(1,i)*sr_theta+1:cooling.nocooling(2,i)*sr_theta) = 1; end
t_cooling = t_cooling(interval(1)*sr_theta:interval(2)*sr_theta);
t_nocooling = t_nocooling(interval(1)*sr_theta:interval(2)*sr_theta);

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
imagesc(bins_speed_avg,freqlist,wt_speed_nocooling'), set(gca,'Ydir','normal')
axis tight, title(['No cooling']), ylabel('Powerspectrum (Hz)'),xlabel('Speed (cm/s)')
set(gca,'YTick',bins_speed), set(gca,'YTick',freqlist(1):freqlist(end)), %set(gca,'xscale','log')
colorbar,%caxis([0 cmax])
subplot(2,2,2)
% surf(bins_lfp,bins_speed_avg,lfp_wavelets_combined,'EdgeColor','None'), view(0,90)
imagesc(bins_speed_avg,freqlist,wt_speed_cooling'), set(gca,'Ydir','normal')
axis tight, title('With Cooling'), ylabel('Powerspectrum (Hz)'),xlabel('Speed (cm/s)')
set(gca,'YTick',bins_speed), set(gca,'YTick',freqlist(1):freqlist(end)), %set(gca,'xscale','log')
cmax = max(max(wt_speed_cooling(:)',wt_speed_nocooling(:)'));
colorbar,%caxis([0 cmax])

subplot(2,3,4)
plot(freqlist,nanmean(wt_speed_cooling),'b'), hold on
plot(freqlist,nanmean(wt_speed_nocooling),'r'), hold on
xlabel('Frequency (Hz)'), grid on, title('Average frequency'),xlim([freqlist(1),freqlist(end)]),ylabel('Power'),
if ~isnan(max(max(mean(wt_speed_nocooling),mean(wt_speed_cooling))))
    ylim([0,max(max(mean(wt_speed_nocooling),mean(wt_speed_cooling)))])
end

subplot(2,3,5)
time1 = bins_speed_avg(find(thetamax_cooling>2));
y1 = freqlist(thetamax_cooling(find(thetamax_cooling>2)));
if ~isempty(y1)
    plot(time1,y1,'ob'), hold on
%     P = polyfit(time1,y1,1);
%     yfit = P(1)*bins_speed+P(2);
%     plot(bins_speed,yfit,'b-');
%     text(15,y1(1)+0.2,['Cooling slope: ', num2str(P(1),3)],'Color','blue')
%     [r,p] = corr(time',y1')
end
time2 = bins_speed_avg(find(thetamax_nocooling>2));
y2 = freqlist(thetamax_nocooling(find(thetamax_nocooling>2)));
plot(time2,y2,'or'), hold on
% P = polyfit(time,y2,1);
% yfit = P(1)*bins_speed+P(2);
% plot(bins_speed,yfit,'r-');
% text(15,y2(10)+0.4,['No Cooling slope: ', num2str(P(1),3)],'Color','red')
% [r,p] = corr(time',y2');
xlim([bins_speed(1),bins_speed(end)]), ylim([5,10]),ylim([y1(1)-0.4,y2(end)+0.4]),
xlabel('Speed (cm/s)'),ylabel('Frequency (Hz)'), grid on, title('Peak frequency of theta')

% Power
subplot(2,3,6)
time3 = bins_speed_avg(find(thetamax_cooling>2));
y3 = theta_maxpower_cooling(find(thetamax_cooling>2));
if ~isempty(y3)
    plot(time3,y3,'ob'), hold on
%     P = polyfit(time,y1,1);
%     yfit = P(1)*bins_speed+P(2);
%     plot(bins_speed,yfit,'b-');
%     text(15,y1(1)-0.1,['Cooling slope: ', num2str(P(1),3)],'Color','blue')
%     [r,p] = corr(time',y1')
end
time4 = bins_speed_avg(find(thetamax_nocooling>2));
y4 = theta_maxpower_nocooling(find(thetamax_nocooling>2));
plot(time4,y4,'or'), hold on
% P = polyfit(time,y2,1);
% yfit = P(1)*bins_speed+P(2);
% plot(bins_speed,yfit,'r-');
% text(15,y2(1)+0.2,['No Cooling slope: ', num2str(P(1),3)],'Color','red')
% [r,p] = corr(time',y2')
xlim([bins_speed(1),bins_speed(end)]), ylim([0,y4(end)+0.8]),
xlabel('Speed (cm/s)'),ylabel('Peak Power'), grid on, title('Amplitude theta'),xlim([bins_speed(1),bins_speed(end)])

thetaVsSpeed.peakFreq.cooling = y1;
thetaVsSpeed.peakFreq.noCooling = y2;
thetaVsSpeed.peakPower.cooling = y3;
thetaVsSpeed.peakPower.noCooling = y4;
thetaVsSpeed.peakFreq.speed_cooling = time1;
thetaVsSpeed.peakFreq.speed_noCooling = time2;
thetaVsSpeed.peakPower.speed_cooling = time3;
thetaVsSpeed.peakPower.speed_noCooling = time4;
thetaVsSpeed.powerSpec.cooling = nanmean(wt_speed_cooling);
thetaVsSpeed.powerSpec.noCooling = nanmean(wt_speed_nocooling);
thetaVsSpeed.freqlist = freqlist;
% % GLM fit
% disp('performing a GLM fit')
% if isfield(animal,'temperature')
%     running_times = find(speed>5);
%     X = [temperature(running_times)+37+60; speed(running_times); [diff(speed(running_times)),0]]';
%     [temp_theta_maxpower,temp_theta_max] = max(wt2(:,(running_times)));
%     y1 = freqlist(temp_theta_max)'; % Frequency
%     y2 = temp_theta_maxpower'; % Power
%     mdl1 = fitglm(X,y1)
%     mdl2 = fitglm(X,y2)
%     theta_bins = [freqlist(1):0.1:freqlist(end)];
%     hist_theta = hist2d([y1';(mdl1.predict)']',theta_bins,theta_bins);
%     figure, imagesc(theta_bins,theta_bins,hist_theta), xlabel('Real theta'), ylabel('Predicted theta'),set(gca,'Ydir','normal')
% end
