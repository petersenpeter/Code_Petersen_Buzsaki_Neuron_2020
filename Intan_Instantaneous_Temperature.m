% Intan_Instantaneous_Temperature
% Reads the aux channel from a live intan recording
%
%
% by Peter Petersen
% petersen.peter@gmail.com

cd 'K:\IntanData\MS21'
files = dir;
[~,I] =  sort([files.datenum]);
cd(files(I(end)).name)
datapath = pwd;

%datapath = 'V:\IntanData\Peter_MS13\Peter_MS13_171121_124648';
sr = 20000;
downsample_n2 = 100;
channel_aux = 1;
num_channels_aux = 1;
channel_analog = 1;
num_channels_analog = 4;
channel_amp = 1;
num_channels_amp = 128;

rec_temperature = 1;
rec_wheelrunning = 0;
rec_spectrogram = 0;
figure(1); clf, hold off, xlabel('Time'),ylabel('Temperature (C)')
test = 1;
i = 0;
temperature = [];
wheel_speed = [];
spectro_peak = [];
wheel_rad = 14.86; % Radius of the wheel in cm

while test == 1
    figure(1); clf,
    if rec_temperature == 1
        if rec_wheelrunning == 1 | rec_spectrogram == 1
            subplot(2,2,1)
        end
        fid = fopen(fullfile(datapath, 'analogin.dat'), 'r');
        i = i+1;
        fseek(fid,-num_channels_aux*sr,'eof');
        v = fread(fid, [num_channels_aux, sr], 'uint16');
        fclose(fid);
%         v = v * 0.0000374; % convert to volts
        v = v * (3/2)*0.000050354; % convert to volts
        v_downsample = mean(reshape(v(channel_aux,1:end-rem(length(v),downsample_n2)),downsample_n2,[]));
        temperature = [temperature,mean((v_downsample-1.25)/(0.00495))];
        plot(temperature,'-k')
        title(['Temperature: ' num2str(temperature(end),4) ' C'],'fontsize',16), xlabel('Time (s)'),ylabel('Temperature (C)'), grid on, ylim([34.5,38.])
    end
    if rec_wheelrunning == 1
        if rec_temperature == 1 | rec_spectrogram == 1
            subplot(2,2,2)
        end
        fid = fopen(fullfile(datapath, 'analogin.dat'), 'r');
        i = i+1;
        fseek(fid,-2*num_channels_analog*sr,'eof');
        v = fread(fid, [num_channels_analog, sr], 'uint16');
        fclose(fid);
        v = v * 0.000050354; % convert to volts 0.0
        wheel_pos = mean(reshape(v(channel_analog,1:end-rem(length(v),downsample_n2)),downsample_n2,[]));
        wheel_pos_polar = unwrap(2*pi*(wheel_pos-0.36)/(2.98-min(wheel_pos)));
        
        wheel_velocity = sum(diff(wheel_rad*wheel_pos_polar));
        wheel_speed = [wheel_speed,wheel_velocity];
        plot(wheel_speed,'-k');
        title(['Wheel: ' num2str(round(wheel_speed(end)*10)/10,4) ' cm/s'],'fontsize',16), xlabel('Time (s)'),ylabel('Wheel Speed (cm/s)'); grid on
    end
    if rec_spectrogram == 1  
        if rec_temperature == 1 | rec_spectrogram == 1
            subplot(2,2,3)
        end
        fid = fopen(fullfile(datapath, 'amplifier.dat'), 'r');
        i = i+1;
        fseek(fid,-2*num_channels_amp*sr,'eof');
        v = fread(fid, [num_channels_amp, sr], 'int16');
        fclose(fid);
        v = v * 0.000050354; % convert to volts
        x = mean(reshape(v(channel_amp,1:end-rem(length(v),downsample_n2)),downsample_n2,[]));
        Fs = 100;
        %t = 0:1/Fs:1-1/Fs;
        N = length(x);
        xdft = fft(x);
        xdft = xdft(1:N/2+1);
        psdx = (1/(Fs*N)) * abs(xdft).^2;
        psdx(2:end-1) = 2*psdx(2:end-1);
        freq = 0:Fs/length(x):Fs/2;
        subplot(2,2,3)
        plot(freq,10*log10(psdx))
        title(['Powerspectrogram: '],'fontsize',16), xlabel('Frequency (Hz)'),ylabel('Power');
        xlim([0,20])
        subplot(2,2,4)
        [~,iii] = max(psdx);
        spectro_peak = [spectro_peak,freq(iii)];
        plot(spectro_peak,'-k'); xlabel('Time (s)'),ylabel('Frequency (Hz)'); grid on
    end
    pause(0.5);
end
