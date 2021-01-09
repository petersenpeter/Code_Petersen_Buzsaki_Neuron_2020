function [InstantaneousTheta] = calcInstantaneousTheta(recording,varargin)
p = inputParser;
addParameter(p,'forceReload',false,@islogical);
addParameter(p,'saveMat',true,@islogical);
addParameter(p,'saveAs','InstantaneousTheta',@isstr);

parse(p,varargin{:})

forceReload = p.Results.forceReload;
saveMat = p.Results.saveMat;
saveAs = p.Results.saveAs;

saveAsFullfile = fullfile([recording.name,'.',saveAs,'.lfp.mat']);

if ~exist([recording.name, '.lfp'],'file')
    disp('Creating lfp file')
    bz_LFPfromDat(pwd,'noPrompts',true)
end

if ~forceReload && exist(saveAsFullfile,'file')
    disp('Loading existing InstantaneousTheta.lfp.mat file')
    InstantaneousTheta = [];
    load(saveAsFullfile)
    if isempty(InstantaneousTheta) || isnumeric(InstantaneousTheta.ThetaInstantFreq)
        disp(['InstantaneousTheta not calculated correctly. Hold on'])
        InstantaneousTheta = [];
        forceReload = true;
    elseif isempty(InstantaneousTheta) || size(InstantaneousTheta.ThetaInstantFreq,2)<recording.ch_theta || isempty(InstantaneousTheta.ThetaInstantFreq{recording.ch_theta})
        forceReload = true;
        disp(['Selected channel not calculated yet. Hold on'])
    end
end

% Calculating the instantaneous theta frequency
if ~exist(saveAsFullfile,'file') || forceReload
    sr_eeg = recording.sr/16;
    disp('Calculating the instantaneous theta frequency')
    signal = 0.000195 * double(LoadBinary([recording.name '.lfp'],'nChannels',recording.nChannels,'channels',recording.ch_theta,'precision','int16','frequency',recording.sr/16)); % ,'start',start,'duration',duration
    Fpass = [4,10];
    Wn_theta = [Fpass(1)/(sr_eeg/2) Fpass(2)/(sr_eeg/2)]; % normalized by the nyquist frequency
    [btheta,atheta] = butter(3,Wn_theta);
    signal_filtered = filtfilt(btheta,atheta,signal)';
    hilbert1 = hilbert(signal_filtered);
    signal_phase = atan2(imag(hilbert1), real(hilbert1));
    signal_phase2 = unwrap(signal_phase);
%     ThetaInstantFreq = 1250/(2*pi)*diff(signal_phase2);
    ThetaInstantFreq = (sr_eeg)./diff(find(diff(signal_phase>0)==1));
    ThetaInstantTime = cumsum(diff(find(diff(signal_phase>0)==1)))/sr_eeg;
    ThetaInstantFreq(find(ThetaInstantFreq>11)) = nan;
    ThetaInstantFreq = nanconv(ThetaInstantFreq,gauss(7,1)/sum(gauss(7,1)),'edge');
    
    % Theta frequency
    freqlist = [4:0.1:10];
    [wt,w,wt_t] = spectrogram(signal_filtered,sr_eeg,9*sr_eeg/10,freqlist,sr_eeg);
    wt = medfilt2(abs(wt),[2,10]);
    h = 1/10*ones(10,1);
    H= h*h';
    wt = filter2(H,wt);
    [~,index] = max(wt);
    signal_freq = freqlist(index);
    signal_power = max(wt);
    
    %max(mean(wt2(:,indexes),2))
    %signal_freq = sr_eeg/(2*pi)*diff(signal_phase2);
    InstantaneousTheta.ThetaInstantFreq{recording.ch_theta} = ThetaInstantFreq;
    InstantaneousTheta.timestamps = ThetaInstantTime;
    InstantaneousTheta.signal_phase{recording.ch_theta} = signal_phase;
    InstantaneousTheta.signal_phase2{recording.ch_theta} = signal_phase2;
    InstantaneousTheta.signal_freq{recording.ch_theta} = signal_freq;
    InstantaneousTheta.signal_power{recording.ch_theta} = signal_power;
    InstantaneousTheta.signal_time = wt_t;
    if saveMat
        save(saveAsFullfile,'InstantaneousTheta')
        disp('InstantaneousTheta saved to disk')
    end
    clear signal signal_filtered
    
end
