%% % Read temperature from the Intan analog in

% Medial Septum Cooling project
% clear all
% datapath = '/Volumes/P/IntanData/'; % Location of the recording computer
% datapath = '/Volumes/TurtlePower/DataBank/Buzsakilab/';
datapath = 'Z:\peterp03\IntanData\MS23\';
% Recordings_MedialSeptum
% id = 19;

% recording = 'Peter_MS23_190820_173321';
recording = 'Peter_MS23_190821_105812';
% recording = 'Peter_MS23_190822_093044';
% recording = 'Peter_MS23_190826_100705';

Intan_rec_info = read_Intan_RHD2000_file_Peter([datapath, recording,'/']);
sr = Intan_rec_info.frequency_parameters.amplifier_sample_rate;

num_channels = length(Intan_rec_info.board_adc_channels); % ADC input info from header file
fileinfo = dir([datapath, recording,'/', 'analogin.dat']);
num_samples = fileinfo.bytes/(num_channels * 2); % uint16 = 2 bytes
fid = fopen([datapath, recording,'/', 'analogin.dat'], 'r');
v = fread(fid, [num_channels, num_samples], 'uint16');
fclose(fid);
v  = v * 0.000050354; % convert to volts
channel_analog = 1:4;
colors = {'k','b','r','m'};

% Peter_MS23_190820_173321
% probe_labels = {'CA1 Left back','CA1 Left front','CA1 Right','MS Implant'};
% temp_correction = [35.97,34.6,34.89,33.14]-35.97;
% cooling_intervals = [1234,2326;2518,3076; 3084,3733; 3865,4457; 4520,5170; 5284,6055; 6482,7102; 7382,7862; 8266,9087; 9088,9826];

%Peter_MS23_190821_105812
probe_labels = {'CA1 Left front','CA1 Right','Room temperature','MS Implant'};
temp_correction = [0, 1.7, 0 3.35]+1;
cooling_intervals = [1506,2500; 2840,3660; 3660,4557; 4558,5289; 6515,7471; 8590,9516; 10396,11396; 13186,14186; 15108,16000; 16015,16900];


%%
temperature = {[],[],[],[]};
samples = floor(size(v,2)/(sr/2));
for j =1:length(channel_analog)
    v2 = mean(reshape(v(j,1:samples*sr/2),[sr/2,samples]));
    temperature{j} = (v2-1.25)/0.00495;  % Translating the voltage into temperature
    temperature{j} = nanconv(temperature{j},gausswin(11)'/sum(gausswin(11)),'edge');  % Translating the voltage into temperature
end
sr_temp = 1/2;
%%
temp23 = []
colormap2 = parula(size(cooling_intervals,1));
for jj = 1:size(cooling_intervals,1)
   temp23(jj) = min(temperature{4}(2*cooling_intervals(jj,1):2*cooling_intervals(jj,2)));
end
[~,idx] = sort(temp23);
colormap2(idx,:) = colormap2;
figure
for j =1:length(channel_analog)
%     subplot(2,2,j)
    plot([1:length(temperature{j})]/2,temperature{j}+temp_correction(j),'-','color',colors{j}), hold on
%     title([probe_labels{j}, ': ' num2str(temperature{j}(end),4) ' C'],'fontsize',13), 
    xlabel('Time (s)'),ylabel('Temperature (C)'), grid on, 
end
legend(probe_labels), axis tight

figure, hold on
for j = 1:length(channel_analog)-1
    subplot(1,3,j), hold on
    x_data = temperature{4}+temp_correction(4);
    y_data = temperature{j}+temp_correction(j);
    for jj = 1:size(cooling_intervals,1)
        
        plot(x_data(2*cooling_intervals(jj,1):2*cooling_intervals(jj,2))-x_data(2*cooling_intervals(jj,1)),y_data(2*cooling_intervals(jj,1):2*cooling_intervals(jj,2))-y_data(2*cooling_intervals(jj,1)),'color',colormap2(jj,:))
    end
    title('Temperature - MS probe vs CA1'), xlabel('MS Probe'), ylabel(probe_labels{j}), axis tight
    xlim([-20,0]),ylim([-2.5,0.1])
end

figure
fitlimits = [25,180;25,180;25,180;1,70];
fit_startPoints = [0.05,0.05,0.05,0.1];

for j = 1:length(channel_analog)
    temp_average = nan(size(cooling_intervals,1),2*max(diff(cooling_intervals')));
    subplot(2,3,j), hold on
    y_data = temperature{j}+temp_correction(j);
    for jj = 1:size(cooling_intervals,1)
        t_data = y_data(2*cooling_intervals(jj,1):2*cooling_intervals(jj,2))-y_data(2*cooling_intervals(jj,1));
        temp_average(jj,1:length(t_data)) = t_data;
        plot([1:length(2*cooling_intervals(jj,1):2*cooling_intervals(jj,2))]/2,y_data(2*cooling_intervals(jj,1):2*cooling_intervals(jj,2))-y_data(2*cooling_intervals(jj,1)),'color',colormap2(jj,:))
    end
    title(probe_labels{j}), xlabel('MS Probe'), ylabel('Temperature'), axis tight
    xlim([0,600])
    subplot(2,3,5), hold on
    plot([1:length(temp_average)]/2,-nanmean(temp_average)/min(nanmean(temp_average)),'linewidth',2)
    
    % Fitting an exponential to the temperature data
    if any(j == [1,2,4])
        x = [1:length(temp_average)]/2;
        t_span = fitlimits(j,1):fitlimits(j,2);
        x = x(t_span);
        y = -nanmean(temp_average)/min(nanmean(temp_average))+1;
        y = y(t_span);
        g = fittype('a-b*exp(-c*x)');
        StartPoint = [0,-1,fit_startPoints(j)];
        f0 = fit(x',y',g,'StartPoint',StartPoint);
        subplot(2,3,6)
        plot(x,y,'.',x,f0(x),'r-'); hold on
        names = coeffvalues(f0);
        text(1,0.5-j/10,num2str(1/names(3)))
    end
end
subplot(2,3,5)
legend(probe_labels), axis tight, ylim([-1,0]), xlim([0,600])

%% Baseline temperature readings without manipulation
datapath = 'Z:\peterp03\IntanData\MS23\';
recordings = {'Peter_MS23_190821_153956','Peter_MS23_190822_093044','Peter_MS23_190823_153247','Peter_MS23_190826_100705','Peter_MS23_190827_144829','Peter_MS23_190828_151011'};
recordings = {'Peter_MS23_190826_100705'};
temperature = {[],[],[],[]};
channel_analog = 4;
colors = {'k','b','r','m'};

for i = 1:length(recordings)
    i
    recording = recordings{i};
    Intan_rec_info = read_Intan_RHD2000_file_Peter([datapath, recording,'/']);
    sr = Intan_rec_info.frequency_parameters.amplifier_sample_rate;
    num_channels = length(Intan_rec_info.board_adc_channels); % ADC input info from header file
    fileinfo = dir([datapath, recording,'/', 'analogin.dat']);
    num_samples = fileinfo.bytes/(num_channels * 2); % uint16 = 2 bytes
    fid = fopen([datapath, recording,'/', 'analogin.dat'], 'r');
    v = fread(fid, [num_channels, num_samples], 'uint16');
    fclose(fid);
    v = v * 0.000050354; % convert to volts
    samples = floor(size(v,2)/(sr/2));
    for j = 1:channel_analog
        temperature{j} = [temperature{j},mean(reshape(v(j,1:samples*sr/2),[sr/2,samples]))];
    end
    clear v
end

for j =1:channel_analog
    temperature{j} = (temperature{j}-1.25)/0.00495;  % Translating the voltage into temperature
    temperature{j} = nanconv(temperature{j},gausswin(11)'/sum(gausswin(11)),'edge');  % Translating the voltage into temperature
end
% save('Z:\peterp03\IntanData\MS23\Peter_MS23_190821_153956_concat\temperature.mat','temperature')

%%
% load('Z:\peterp03\IntanData\MS23\Peter_MS23_190821_153956_concat\temperature.mat','temperature')
probe_labels = {'CA1 Left front','CA1 Right','Room temperature','MS Implant'};
temp_correction = [0, 1.50, 0 3.10];
bins_temperature = [15:0.1:38];
figure
k = 1
for j =1:channel_analog
%     if j == 4
%         temp = temperature{j}(1:230400)+temp_correction(j);
%     else
        temp = temperature{j}+temp_correction(j);
%     end
    if j == 3
        subplot(3,3,[7,8])
    else
        subplot(3,3,[1,2,4,5])
    end
    plot([1:length(temp)]/2/3600,temp,'-','color',colors{j}), hold on, axis tight
    xlabel('Time (hours)'),ylabel('Temperature (C)'), grid on, 
    
    
    N = histcounts(temp,bins_temperature,'Normalization','probability');
    if j == 3
        subplot(3,3,9)
        plot(bins_temperature(1:end-1)+0.005,N,'color',colors{j}), hold on
        xlim([19,20.6])
    else
        subplot(3,3,3)
        plot(bins_temperature(1:end-1)+0.005,N,'color',colors{j}), hold on
        xlim([34.5,38])
        subplot(3,3,6)
        plot(k,mean(temp),'pk'), hold on
        plot(k,mean(temp)+std(temp),'ok'), hold on
        plot(k,mean(temp)-std(temp),'ok'), hold on
        k = k +1;
    end
end
subplot(3,3,6)
xticks([1,2,3]), xticklabels({probe_labels{[1,2,4]}}), xtickangle(45)
ylabel('Temperature (C)')
subplot(3,3,3)
xlabel('Temperature (C)'), title('Distributions')
subplot(3,3,9)
xlabel('Temperature (C)')
subplot(3,3,[1,2,4,5])
legend(probe_labels{[1,2,4]}), axis tight
