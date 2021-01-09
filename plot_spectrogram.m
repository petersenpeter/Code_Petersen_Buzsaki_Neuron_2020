function plot_spectrogram(recording,animal,cooling)
disp('Plotting Theta vs running speed')
freqlist = 10.^(0.6:0.01:2); %[4:0.3:50]; %10.^(0.4771:0.01:1.1761);
signal = 0.000195 * double(LoadBinary([recording.name '.lfp'],'nChannels',recording.nChannels,'channels',recording.ch_theta,'precision','int16','frequency',recording.sr_lfp)); % ,'start',start,'duration',duration
sr_theta = 250;
signal2 = resample(signal,sr_theta,recording.sr_lfp);
Fpass = [3,120];
if recording.sr_lfp < 100
    Fpass = [3,99];
end
Wn_theta = [Fpass(1)/(sr_theta/2) Fpass(2)/(sr_theta/2)]; % normalized by the nyquist frequency
[btheta,atheta] = butter(3,Wn_theta);
signal_filtered = filtfilt(btheta,atheta,signal2);

%[wt,~,~] = awt_freqlist(signal_filtered,sr_temperature,freqlist);
%wt2 = abs(wt)'; clear wt
wt = spectrogram(signal_filtered,250,249,freqlist,sr_theta);
wt2 = [zeros(length(freqlist),124),abs(wt), zeros(length(freqlist),125)]; clear wt signal2 signal
speed = interp1(animal.time,animal.speed2,(1:length(signal_filtered))/sr_theta);
temperature = interp1(animal.time,animal.temperature,(1:length(signal_filtered))/sr_theta);
position = interp1(animal.time,animal.pos_linearized,(1:length(signal_filtered))/sr_theta);
indexes = find(speed > 10 & position>0);

figure
subplot(5,1,1:3)
yyaxis left, set(gca, 'YScale', 'log'), ylim([freqlist(1),freqlist(end)])
yyaxis right, yticks([])
imagesc([1:length(indexes)]/sr_theta,freqlist,wt2(:,indexes)), set(gca,'Ydir','normal'), clim([0,15])

subplot(5,1,4)
plot([1:length(indexes)]/sr_theta,speed(indexes),'k'), ylabel('Speed (cm/s)'), axis tight, ylim([0,150])
subplot(5,1,5)
plot([1:length(indexes)]/sr_theta,temperature(indexes),'k'), xlabel('Time (s)'), ylabel('Temperature (degrees)'), axis tight

% FOOOF harmonics analysis
% Calculate a power spectrum with Welch's 
%%
windowTime = 3;
harmonics_intervals = [6,10;13,20;21,28;28,38;62,100];
peaks_freq = [];
peaks_power = [];
temperature_bins = [];
harmonics_power = [];
for i = 1:length(indexes)/(sr_theta*windowTime)-1
%     [psd, freqs] = pwelch(signal_filtered(), 300, [], [], sr_theta);
    % Transpose, to make FOOOF inputs row vectors
%     freqs = freqs';
%     psd = psd';
    idx1 = indexes(i*sr_theta*windowTime:(i+1)*sr_theta*windowTime);
    temperature_bins(i) = mean(temperature(idx1));
    psd = mean(wt2(:,idx1)');
    freqs = freqlist;
    % FOOOF settings
    settings = struct();  % Use defaults
    settings.max_n_peaks = 5;
    f_range = [5, 45];
    
    % Run FOOOF
    fooof_results = fooof(freqs, psd, f_range, settings,1);
    
    % Print out the FOOOF Results
    figure(1), 
    subplot(2,2,1)
    cla, fooof_plot(fooof_results), ylim([-0.4,1.5])
    subplot(2,2,2), hold on
    plot(i*ones(1,size(fooof_results.gaussian_params,1)),fooof_results.gaussian_params(:,1),'o')
    
    for ii=1:size(harmonics_intervals,1)
        idx = find(freqlist>harmonics_intervals(ii,1) & freqlist<harmonics_intervals(ii,2));
        [peaks_power(ii,i),peak_index] = max(psd(idx));
        peaks_freq(ii,i) = freqlist(idx(peak_index));
        harmonics_power(ii,i) = mean(psd(idx));
    end
    subplot(2,2,4), hold on
    plot(i*ones(1,length(peaks_freq(:,i))),peaks_freq(:,i),'o')
    subplot(2,2,3), cla
    plot(freqlist,psd), hold on
    plot(peaks_freq(:,i),peaks_power(:,i),'xk'), xlim([5,45]), ylim([0,20])
    pause(0.3)
end
subplot(2,2,2)
ylabel('Frequency (Hz)'),xlabel('Time (sec)')
subplot(2,2,4)
ylabel('Frequency (Hz)'),xlabel('Time (sec)')
figure
subplot(1,3,1)
plot(temperature_bins,peaks_freq,'o')
ylabel('Frequency (Hz)'),xlabel('Temperature (C)')
subplot(1,3,2)
plot(temperature_bins,harmonics_power,'o')
ylabel('Band power'),xlabel('Temperature (C)')
subplot(1,3,3)
plot(temperature_bins,peaks_power,'o')
ylabel('Peak Power'),xlabel('Temperature (C)')

