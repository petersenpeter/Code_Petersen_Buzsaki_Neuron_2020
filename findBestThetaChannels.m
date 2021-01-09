% function findBestThetaChannels(recording)
clear all
MedialSeptum_Recordings
% Processed datasets in MS12: 78,79,80,81,
% Processed datasets in MS13: 92
% Processed datasets in MS14: 
% Processed datasets in MS18:
id = 126; % 63

recording = recordings(id);
f_all = [2 20];
f_theta = [4 10];
numfreqs = 100;
Fs = recording.sr/16;
sr_eeg = Fs;

smoothfact = 10;
thsmoothfact = 10;
thFFTfreqs = logspace(log10(f_all(1)),log10(f_all(2)),numfreqs);
ThetaChannels = 1:recording.nChannels;
nThetaChannels = recording.nChannels;
usechannels = ThetaChannels;
THmeanspec = [];
numhistbins = 21;
histbins = linspace(0,1,numhistbins);
downsample_value = 10;

peakTH = [];
allLFP = 0.000050354 * double(LoadBinary([recording.name '.eeg'],'nChannels',nThetaChannels,'channels',ThetaChannels,'precision','int16','frequency',recording.sr/16,'start',6000,'duration',4000));
% Wn_theta = [f_all(1)/(sr_eeg/2) f_all(2)/(sr_eeg/2)]; % normalized by the nyquist frequency
% [btheta,atheta] = butter(3,Wn_theta);
% allLFP = filtfilt(btheta,atheta,allLFP)';
allLFP = downsample(allLFP,downsample_value);
Fs = Fs/downsample_value;

window = 10;
noverlap = 9;
window = Fs;
noverlap = noverlap/window*Fs;
% allLFP = bz_LoadBinary(rawlfppath,'frequency',Fs,...
%      'nchannels',nChannels,'channels',usechannels+1,'downsample',downsamplefactor,...
%     'start',scoretime(1),'duration',diff(scoretime));

parfor idx = 1:nThetaChannels
    % Get spectrogram and calculate theta ratio
    LFPchanidx = find(usechannels==ThetaChannels(idx));
    thFFTspec = spectrogram(single(allLFP(:,LFPchanidx)),window,noverlap,thFFTfreqs,Fs);
    thFFTspec = (abs(thFFTspec));

    thfreqs = find(thFFTfreqs>=f_theta(1) & thFFTfreqs<=f_theta(2));
    thpower = sum((thFFTspec(thfreqs,:)),1);
    allpower = sum((thFFTspec),1);

    thratio = thpower./allpower;    %Narrowband Theta
    thratio = smooth(thratio,thsmoothfact);
    thratio = (thratio-min(thratio))./max(thratio-min(thratio));
    
    % Histogram and diptest of Theta
    THhist(:,idx) = hist(thratio,histbins);
    % Dip test of theta doesn't get used... could be incorporated for
    % selection?
    % dipTH(idx) = hartigansdiptest_ss(sort(thratio));
    
    % Ratio of Theta Peak to sorrounding in mean spectrum (for selection)
    meanspec = (mean(thFFTspec,2));
    meanthratio = sum((meanspec(thfreqs)))./sum((meanspec(:)));
    
    %Record the spec and peak ratio for later comparison between chans
    THmeanspec(:,idx) = meanspec;
    peakTH(idx) = meanthratio;
end
disp('Saving results to mat file')
save('ThetaPowerAcrossChannels.mat','THmeanspec','peakTH')
disp('Plotting results')
figure, plot(thFFTfreqs,THmeanspec), title('Theta power across channels'), xlabel('Frequency (Hz)')
