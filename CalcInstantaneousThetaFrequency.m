function [signal_phase,signal_phase2] = CalcInstantaneousThetaFrequency(datapath,theta_channel)
% Calculating the instantaneous theta frequency
disp('Calculating the instantaneous theta frequency')
Intan_rec_info = read_Intan_RHD2000_file_Peter(datapath);
sr = Intan_rec_info.frequency_parameters.amplifier_sample_rate;
nbChan = size(Intan_rec_info.amplifier_channels,2);
signal = 0.000050354 * double(LoadBinary([datapath,'amplifier.eeg'],'nChannels',nbChan,'channels',theta_channel,'precision','int16','frequency',sr/16)); % ,'start',start,'duration',duration
Fs = sr/16;
Fpass = [4,11];
Wn_theta = [Fpass(1)/(Fs/2) Fpass(2)/(Fs/2)]; % normalized by the nyquist frequency
[btheta,atheta] = butter(3,Wn_theta);
signal_filtered = filtfilt(btheta,atheta,signal);
signal_phase = atan2(imag(hilbert(signal_filtered)), signal_filtered);
signal_phase2 = unwrap(signal_phase);
save('InstantaneousTheta.mat','signal_phase','signal_phase2')
