% Intan_Instantaneous_Temperatu_4inputs
% Reads the analog/aux channels from a live Intan recording
%
%
% by Peter Petersen
% petersen.peter@gmail.com

cd 'K:\IntanData\MS24'
files = dir;
[~,I] =  sort([files.datenum]);
cd(files(I(end)).name)
datapath = pwd;

% datapath = 'K:\IntanData\MS23\Peter_MS23_190821_153956';
sr = 20000;
downsample_n2 = 100;
channel_analog = [1,2,3,4];
num_channels_analog = 4;
channel_amp = 1;
num_channels_amp = 128;
warning off
rec_temperature = 1;
rec_wheelrunning = 0;
rec_spectrogram = 0;
figure(1); clf, hold off, xlabel('Time'),ylabel('Temperature (C)')
test = 1;
temperature = {[],[],[],[]};
colors = {'k','b','r','m'};
probe_labels = {'CA1 Left','CA1 Right','Room','MS Implant'};
probe_labels1 = {'CA1 Left','CA1 Right','MS Implant'};
probe_labels1 = {'CA1 Left','CA1 Right','Room','Heatpad'};
probe_labels1 = {'MS Implant','CA1 Right','CA1 Left','Room'};
temp_correction = [35.49, 33.87, 35.49 32.25]-35.49;
temp_correction = [-0.6, 0.1, 0., 0];
% temp_correction = [0, 0, 0, 0];
while test == 1
    fid = fopen(fullfile(datapath, 'analogin.dat'), 'r');
    % The last second of data from the file is loaded.
    fseek(fid,-num_channels_analog*sr,'eof'); 
    v = fread(fid, [num_channels_analog, sr], 'uint16');
    fclose(fid);
%     v = v * 0.0000374; % convert to volts
%     v = v * (3/2)*0.000050354; % convert to volts
    v = v *0.000050354; % convert to volts (analog channels)
    
    figure(1); clf, hold on
    for j =1:length(channel_analog)
%         v_downsample = mean(reshape(v(j,1:end-rem(length(v),downsample_n2)),downsample_n2,[]));
        temperature{j} = [temperature{j},mean((v(j,:)-1.25)/(0.00495))]; % Translating the voltage into temperature
        temperature{j}(temperature{j}>42) = nan;
        if any(j == [1,2,3])
        subplot(4,1,[1,2,3])
        plot((1:length(temperature{j}))/120,temperature{j}+temp_correction(j),'-','color',colors{j}), hold on
        title([num2str(temperature{j}(end),4) ' C'],'fontsize',13), xlabel('Time (min)'),ylabel('Temperature (C)'), grid on, 
        legend(probe_labels1,'Location','southwest')
%         ylim([20,39.])
        elseif j == 4
            subplot(4,1,4)
            plot((1:length(temperature{j}))/120,temperature{j},'-','color',colors{j})
            title([probe_labels1{3}, ': ' num2str(temperature{j}(end),3) ' C'],'fontsize',13), xlabel('Time (min)'),ylabel('Temperature (C)'), grid on, 
        else
            subplot(4,1,4)
             plot((1:length(temperature{j}))/120,temperature{j},'-','color',colors{j})
            title([probe_labels1{4}, ': ' num2str(temperature{j}(end),4) ' C'],'fontsize',13), xlabel('Time (min)'),ylabel('Temperature (C)'), grid on, 
        end
        
    end
    pause(0.5); % Pauses for 5 seconds before the file is read again. 
end
