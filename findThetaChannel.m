% Find best theta channel
f_all = [2 20];
f_theta = [5 10];
numfreqs = 100;
thfreqlist = logspace(log10(f_all(1)),log10(f_all(2)),numfreqs);

[thFFTspec,thFFTfreqs] = spectrogram(allLFP(:,chanidx),window,noverlap,thfreqlist,Fs);
thFFTspec = (abs(thFFTspec));

thfreqs = find(thFFTfreqs>=f_theta(1) & thFFTfreqs<=f_theta(2));
thpower = sum((thFFTspec(thfreqs,:)),1);
allpower = sum((thFFTspec),1);

thratio = thpower./allpower;    %Narrowband Theta
thratio = smooth(thratio,thsmoothfact);
thratio = (thratio-min(thratio))./max(thratio-min(thratio));
