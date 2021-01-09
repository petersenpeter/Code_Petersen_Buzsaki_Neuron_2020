file_path = 'G:\ArduinoData\';
% Rat MS5: Digital amplifier with K-type thermocouple implantet in MS
% file_name = 'CoolTerm Capture 2016-08-04 12-53-01 v2.asc';
% file_name = 'CoolTerm Capture 2016-08-05 16-39-00.asc';
% file_name = 'CoolTerm Capture 2016-08-08 12-24-17.asc';

% Rat MS6: Analog amplifier with K-type thermocouple implantet in MS
%file_name = 'CoolTerm Capture 2016-09-15 11-14-40.asc'; % implantet temp
% file_name = 'CoolTerm Capture 2016-09-15 12-24-18.asc'; % room temp
% file_name = 'CoolTerm Capture 2016-09-15 14-09-27.asc'; % room temp
%file_name = 'CoolTerm Capture 2016-09-15 15-50-45.asc'; % room temp
%file_name = 'CoolTerm Capture 2016-09-16 10-39-46.asc'; % room temp
%file_name = 'CoolTerm Capture 2016-09-16 15-06-31.asc'; % room temp
%file_name = 'CoolTerm Capture 2016-09-16 16-10-46.asc'; % room temp
%file_name = 'CoolTerm Capture 2016-09-20 13-24-57.asc'; % room temp
% file_name = 'CoolTerm Capture 2016-09-21 13-31-18.asc'; % room temp
% file_name = 'CoolTerm Capture 2016-09-22 17-41-32.asc'; % room temp
%file_name = 'CoolTerm Capture 2016-09-22 18-02-12.asc'; % room temp
% MS8
file_name = 'CoolTerm Capture 2016-11-22 11-37-08.asc'; % room temp

delimiter = '\t';
formatSpec = '%s%f%[^\n\r]';
delimiter = {'\t',' '};
formatSpec = '%{yyyy-MM-dd}D%{HH:mm:ss}D%s%f%s%f%s%f%s%f%s%f%[^\n\r]';

fileID = fopen([file_path file_name],'r');
dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter,  'ReturnOnError', false);
fclose(fileID);
Timestamp = datenum(dataArray{:,2});
Temp = dataArray{:, 4};
figure; plot(Timestamp-Timestamp(1),Temp),datetick
xlabel('Time'), ylabel('Temperature (C)'), title(['Temperature profile: ' file_name])

%% % Read temperature from the Intan analog in

% Medial Septum Cooling project
% clear all
% datapath = '/Volumes/P/IntanData/'; % Location of the recording computer
% datapath = '/Volumes/TurtlePower/DataBank/Buzsakilab/';
datapath = 'D:\IntanData\Peter_MS12\';
% Recordings_MedialSeptum
% id = 19;

recording = 'Peter_MS12_170714_102142'; %recordings(id).name;
% recording = 'Peter_MS7_161008_134424';

Intan_rec_info = read_Intan_RHD2000_file_Peter([datapath, recording,'/']);
sr = Intan_rec_info.frequency_parameters.amplifier_sample_rate;

num_channels = length(Intan_rec_info.board_adc_channels); % ADC input info from header file
fileinfo = dir([datapath, recording,'/', 'analogin.dat']);
num_samples = fileinfo.bytes/(num_channels * 2); % uint16 = 2 bytes
fid = fopen([datapath, recording,'/', 'analogin.dat'], 'r');
v = fread(fid, [num_channels, num_samples], 'uint16');
fclose(fid);
v = v * 0.000050354; % convert to volts
downsample_n2 = 200;
%v_downsample = downsample(v,downsample_n2); %clear v;
%if num_channels .
v_downsample = mean(reshape(v(2,1:end-rem(length(v),downsample_n2)),downsample_n2,[]));
temperature = conv((v_downsample-1.25)/0.005,ones(1,50)/50,'same'); 
sr_temp = Intan_rec_info.frequency_parameters.amplifier_sample_rate/downsample_n2;
figure; plot([1:length(temperature)]/(sr/downsample_n2),temperature), axis tight, %ylim([22,40])
xlabel('Time'),ylabel('Temperature')

%% % Testing the cooling effect of peltier devices and dry ice
datapath = 'D:\IntanData\';
recordings = {'Peter_MS6_161006_123011','Peter_MS6_161006_134531'}; %recordings(id).name;

for i = 1:2
    Intan_rec_info = read_Intan_RHD2000_file_Peter([datapath, recordings{i},'/']);
    sr = Intan_rec_info.frequency_parameters.amplifier_sample_rate;
    
    num_channels = length(Intan_rec_info.board_adc_channels); % ADC input info from header file
    fileinfo = dir([datapath, recordings{i},'/', 'analogin.dat']);
    num_samples = fileinfo.bytes/(num_channels * 2); % uint16 = 2 bytes
    fid = fopen([datapath, recordings{i},'/', 'analogin.dat'], 'r');
    v = fread(fid, [num_channels, num_samples], 'uint16');
    fclose(fid);
    v = v * 0.000050354; % convert to volts
    downsample_n2 = 200;
    %v_downsample = downsample(v,downsample_n2); %clear v;
    %if num_channels .
    v_downsample = mean(reshape(v(1,1:end-rem(length(v),downsample_n2)),downsample_n2,[]));
    temperature = conv((v_downsample-1.25)/0.005,ones(1,50)/50,'same');
    sr_temp = Intan_rec_info.frequency_parameters.amplifier_sample_rate/downsample_n2;
    figure(1); plot([1:length(temperature)]/(sr/downsample_n2),temperature), hold on, axis tight, %ylim([22,40])
end
xlabel('Time'),ylabel('Temperature')
%% % Reading Temperature data from the OMEGA thermometer
%
datapath = 'G:\OmegaTemperatureData\'; % Windows Peter
datapath = 'P:\OmegaTemperatureData\'; % Windows PeterSetup
Recordings_OmegaTemperature;
format long
for i = 14:16
    id = i;
    [out] = ImportOmegaTemperature([datapath recordings(id).filename '.csv'],recordings(id).delimiter);
    rec_start = [datenum(recordings(id).filename,'yyyymmddHHMMSS')];
    events = [];
    for j = 1:length(recordings(id).events_timestamps)
        events(j) = datenum([recordings(id).filename(1:8),num2str(recordings(id).events_timestamps(j))],'yyyymmddHHMMSS')-rec_start;
    end
    events = events*24*60*60;
    figure(i)
    plot(out.sample,[out.temp1,out.temp2]'), legend(recordings(id).tempLocations),xlabel('Time'),ylabel('Temperature (C)'),title([recordings(id).comments,' (minimum T: ' num2str(min(out.temp1)) 'C)'])
    gridxy(events,'color',[0.8,0.8,0.8]), text(events+8,out.temp2(floor(events)+1),recordings(id).events_labels)
    if id == 12
        x1 = floor(events)+1;
        figure, plot([1:3001],out.temp1(x1(1):x1(1)+3000)), hold on
        plot([1:3001],out.temp1(x1(2):x1(2)+3000))
        plot([1:3001],abs(out.temp1(x1(2):x1(2)+3000)-out.temp1(x1(1):x1(1)+3000))), legend({'With graphene','Without graphene','Difference'}),xlabel('Time'),ylabel('Temperature (C)'),title('Dry Ice Cooling of water bath')
    end
end

%% % Reading temperature data from aux channel 1
datapath = 'F:\IntanData\Peter_MS13\';
% Recordings_MedialSeptum
% id = 19;

recording = 'Peter_MS13_171129_105507_concat'; % recordings(id).name;
% recording = 'Peter_MS7_161008_134424';
channel_aux = 1;
Intan_rec_info = read_Intan_RHD2000_file_Peter([datapath, recording,'/']);
sr = Intan_rec_info.frequency_parameters.aux_input_sample_rate*4;

num_channels = length(Intan_rec_info.aux_input_channels); % ADC input info from header file
fileinfo = dir([datapath, recording,'\', 'auxiliary.dat']);
num_samples = fileinfo.bytes/(num_channels * 2); % uint16 = 2 bytes
fid = fopen([datapath, recording,'/', 'auxiliary.dat'], 'r');
v = fread(fid, [num_channels, num_samples], 'uint16');
fclose(fid);
v = v *  0.0000374; % convert to volts
downsample_n2 = 100;
v_downsample = mean(reshape(v(channel_aux,1:end-rem(length(v),downsample_n2)),downsample_n2,[]));
temperature = nanconv((v_downsample-1.25+0.0125)/(0.005/1.1154),ones(1,50)/50,'edge');
sr_temp = sr/downsample_n2;
figure; plot([1:length(temperature)]/sr_temp,temperature), axis tight, %ylim([22,40])
xlabel('Time'),ylabel('Temperature')
