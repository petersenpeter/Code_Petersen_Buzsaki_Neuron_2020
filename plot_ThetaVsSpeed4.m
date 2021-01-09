function thetaVsSpeed = plot_ThetaVsSpeed4(session,animal,cooling,interval,animal_speed_max)
disp('Plotting Theta vs running speed')
bins_speed = [10:3:animal_speed_max]; % [-40:5:40]; % [5:3:60]
bins_speed_avg = mean([bins_speed(2:end);bins_speed(1:end-1)]);
freqlist = [5.5:0.1:11]; %10.^(0.4771:0.01:1.1761);
if ~exist([session.general.name, '.lfp'])
    disp('Creating lfp file')
    bz_LFPfromDat(pwd,'noPrompts',true)
%     downsample_dat_to_eeg(session.general.name,pwd);
end
signal = 0.000195 * double(LoadBinary([session.general.name '.lfp'],'nChannels',session.extracellular.nChannels,'channels',session.channelTags.Theta.channels,'precision','int16','frequency',session.extracellular.srLfp)); % ,'start',start,'duration',duration
sr_theta = animal.sr;
signal2 = resample(signal,sr_theta,session.extracellular.srLfp);
Fpass = [1,49];
if sr_theta < 101
    % Filtering the band 
    Fpass = [1,14.9];
end
colors = {'r','b','g'};
colors1 = {'or','ob','og'};
conditions = {'Pre','Cooling','Post'};

Wn_theta = [Fpass(1)/(sr_theta/2) Fpass(2)/(sr_theta/2)]; % normalized by the nyquist frequency
[btheta,atheta] = butter(3,Wn_theta);
signal_filtered = filtfilt(btheta,atheta,signal2);
signal_filtered2 = signal_filtered;
signal_filtered = signal_filtered(interval(1)*sr_theta:interval(2)*sr_theta);
time_signal = (interval(1)*sr_theta:interval(2)*sr_theta)/sr_theta;

wt = spectrogram(signal_filtered,100,99,freqlist,sr_theta);
wt2 = [zeros(length(freqlist),49),abs(wt), zeros(length(freqlist),50)]; clear wt
speed = interp1(animal.time,animal.speed,time_signal);

figure
for i = 1:3
    t_cooling = zeros(1,length(signal_filtered2));
    
    if i ==1 % PRE
        t_cooling(cooling.nocooling(1,1)*sr_theta+1:cooling.nocooling(2,1)*sr_theta) = 1; 
        t_cooling = t_cooling(interval(1)*sr_theta:interval(2)*sr_theta);
        
    elseif i == 2 % COOLING
        t_cooling(cooling.cooling(1,1)*sr_theta:cooling.cooling(2,1)*sr_theta) = 1;
        t_cooling = t_cooling(interval(1)*sr_theta:interval(2)*sr_theta);
        
    elseif i == 3 % POST
        t_cooling(cooling.nocooling(1,2)*sr_theta+1:cooling.nocooling(2,2)*sr_theta) = 1; 
        t_cooling = t_cooling(interval(1)*sr_theta:interval(2)*sr_theta);
    end
    
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
    
    subplot(2,3,i) % Spectrograms
    % surf(bins_lfp,bins_speed_avg,lfp_wavelets_combined,'EdgeColor','None'), view(0,90)
    imagesc(bins_speed_avg,freqlist,wt_speed_cooling'), set(gca,'Ydir','normal')
    axis tight, title([conditions{i}]), ylabel('Powerspectrum (Hz)'),xlabel('Speed (cm/s)')
    set(gca,'YTick',bins_speed), set(gca,'YTick',freqlist(1):freqlist(end)), %set(gca,'xscale','log')
    colorbar,%caxis([0 cmax])

    
    subplot(2,3,4) % Stats: average frequency
    plot(freqlist,nanmean(wt_speed_cooling),colors{i}), hold on
    xlabel('Frequency (Hz)'), grid on, title('Average frequency'),xlim([freqlist(1),freqlist(end)]), ylabel('Power'),
%     if ~isnan(max(max(mean(wt_speed_cooling),mean(wt_speed_cooling))))
%         ylim([0,max(max(mean(wt_speed_nocooling),mean(wt_speed_cooling)))])
%     end
%     ylim([0,5])
    subplot(2,3,5) % Stats: theta peak frequency vs speed
    time1 = bins_speed_avg(find(thetamax_cooling>2));
    y1 = freqlist(thetamax_cooling(find(thetamax_cooling>2)));
    if ~isempty(y1)
        plot(time1,y1,colors1{i}), hold on
    end
    xlim([bins_speed(1),bins_speed(end)]), 
%     ylim([5,10]),ylim([y1(1)-0.4,y2(end)+0.4]),
    xlabel('Speed (cm/s)'),ylabel('Frequency (Hz)'), grid on, title('Peak frequency of theta')
    
    subplot(2,3,6) % Stats: theta peak power vs speed
    time3 = bins_speed_avg(find(thetamax_cooling>2));
    y3 = theta_maxpower_cooling(find(thetamax_cooling>2));
    if ~isempty(y3)
        plot(time3,y3,colors1{i}), hold on
        %     P = polyfit(time,y1,1);
        %     yfit = P(1)*bins_speed+P(2);
        %     plot(bins_speed,yfit,'b-');
        %     text(15,y1(1)-0.1,['Cooling slope: ', num2str(P(1),3)],'Color','blue')
        %     [r,p] = corr(time',y1')
    end
    xlim([bins_speed(1),bins_speed(end)]),
%     ylim([0,y4(end)+0.8]),
    xlabel('Speed (cm/s)'),ylabel('Peak Power'), grid on, title('Amplitude theta'),xlim([bins_speed(1),bins_speed(end)])
    thetaVsSpeed.speed{i} = time1;
    thetaVsSpeed.peakFreq{i} = y1;
    thetaVsSpeed.peakPower{i} = y3;
    thetaVsSpeed.peakFreq1{i} = time1;
    thetaVsSpeed.peakPower1{i} = time3;
    thetaVsSpeed.powerSpec{i} = nanmean(wt_speed_cooling);
    
end
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
