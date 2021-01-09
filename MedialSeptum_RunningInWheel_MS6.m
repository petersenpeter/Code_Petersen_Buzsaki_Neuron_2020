% Medial Septum Cooling project
clear all
% datapath = '/Volumes/P/IntanData/'; % Location of the recording computer
% datapath = '/Volumes/TurtlePower/DataBank/Buzsakilab/';
datapath = 'G:\IntanData\';
Recordings_MedialSeptum
id = 31;

recording= recordings(id).name;
Intan_rec_info = read_Intan_RHD2000_file_Peter([datapath, recording,'/']);
fname = [datapath, recording, '/amplifier.dat'];
nbChan = size(Intan_rec_info.amplifier_channels,2);
cooling = recordings(id).cooling;
cooling_onsets = recordings(id).cooling_onsets;
sr = Intan_rec_info.frequency_parameters.amplifier_sample_rate;
ch_lfp = recordings(id).ch_lfp;
ch_medialseptum = recordings(id).ch_medialseptum;
ch_hippocampus = recordings(id).ch_hippocampus;
time_frame = recordings(id).time_frame;
lfp_periods = 30*60; % in seconds
nb_lfp_periods = ceil((time_frame(2)-time_frame(1))/lfp_periods);
temp_ = dir(fname);
recording_length = round(temp_.bytes/sr/nbChan/2)-1;
cd([datapath, recording, '/'])
% %%
% theta_start = 1*60; % 77 min into the recording (Peter_nb2_160426_130349)
% % theta_start = 100*60;
% theta_step_length = 5*60;
% % theta_instances = [1:theta_step_length:86*60];
% theta_instances = [1:theta_step_length:recordings(id).time_frame(2)];
% theta_duration = 10;
% % lfp = zeros(1,sr*length(theta_instances)*thera_duration);
% 
% for i = 1:length(theta_instances)
%     disp(num2str(i))
%     lfp2(:,i) = LoadBinary(fname,'nChannels',nbChan,'channels',ch_lfp,'precision','int16','frequency',sr,'start',theta_instances(i),'duration',theta_duration);
% end
% lfp = lfp2(:);
% Fc2 = [100];
% [b1,a1]=butter(3,Fc2*2/sr,'low'); % 'high' pass filter (high,low,bandpass)
% lfp_filt = filtfilt(b1,a1,double(lfp(:)));
% downsample_n1 = 100;
% lfp_filt = downsample(lfp_filt,downsample_n1);
% t_axis = [1:length(lfp_filt)]/(sr/downsample_n1);

% %%
% % spectrogram(lfp_filt,5*sr/downsample_n,0,256,sr/downsample_n,'yaxis')
% freqlist = [3:0.1:15]; %10.^[0:0.03:1.5];
% [wt,freqlist,psi_array] = awt_freqlist(lfp_filt,sr/downsample_n1,freqlist);
% wt2 = ((abs(wt)'));
% figure(1); surf(t_axis/theta_duration,freqlist,wt2,'EdgeColor','None'), hold on
% view(0,90)
% set(gca,'yscale','log')
% set(gca,'YTick',([1 2 3 4 5 6 7 8 9 10 20 30]))
% axis tight
% zheight = max(max(wt2'));
% 
% plot3([1:length(theta_instances);1:length(theta_instances)],[freqlist(1),freqlist(end)],[zheight,zheight],'w')
% plot3(cooling'/theta_step_length,[freqlist(end),freqlist(end);freqlist(end),freqlist(end)],[zheight,zheight;zheight,zheight],'r','linewidth',2)
% % plot3([cooling_offset,cooling_offset]/theta_step_length,[freqlist(1),freqlist(end)],[zheight,zheight],'r','linewidth',2)
% xlabel('Time (20 sec intervals with 5 mins pauses)')
% ylabel('Frequency (Hz)')
% caxis([0 1500])

%% % Running wheel analysis
num_channels = length(Intan_rec_info.board_adc_channels); % ADC input info from header file
fileinfo = dir([datapath, recording,'/', 'analogin.dat']);
num_samples = fileinfo.bytes/(num_channels * 2); % uint16 = 2 bytes
fid = fopen([datapath, recording,'/', 'analogin.dat'], 'r');
v = fread(fid, [num_channels, num_samples], 'uint16');
fclose(fid);
v = v * 0.000050354; % convert to volts
downsample_n2 = 200;
wheel_pos = downsample(v(1,:),downsample_n2);
v_downsample = downsample(v(2,:),downsample_n2); clear v;
v_downsample(v_downsample<1.25) = 1.25;
temperature = conv((v_downsample-1.25)/0.005,ones(1,100)/100,'same'); 
save([datapath recording '/temperature.mat'],'temperature')
sr_wheel_pos = Intan_rec_info.frequency_parameters.amplifier_sample_rate/downsample_n2;
while sum((wheel_pos < 0.1))
    wheel_pos(find(wheel_pos < 0.1)) = wheel_pos(find(wheel_pos < 0.1)-1);
end
wheel_pos_polar = 2*pi*(wheel_pos-min(wheel_pos))/(max(wheel_pos)-min(wheel_pos));

%
calibration = 0;
if calibration == 1
    figure(2)
    ax1(1) = subplot(2,1,1); plot([1:length(wheel_pos_polar)]/sr_wheel_pos,wheel_pos_polar,'.'), hold on
    wheel_reset_left = find(diff(wheel_pos_polar)>2*pi*0.9)+1;
    wheel_reset_right = find(diff(wheel_pos_polar)<-2*pi*0.9);
    plot(wheel_reset_left/sr_wheel_pos,2*pi*ones(length(wheel_reset_left),1),'*r'),
    plot((wheel_reset_left-1)/sr_wheel_pos,zeros(length(wheel_reset_left),1),'*r')
    plot(wheel_reset_right/sr_wheel_pos,2*pi*ones(length(wheel_reset_right),1),'*b'),
    plot((wheel_reset_right-1)/sr_wheel_pos,zeros(length(wheel_reset_right),1),'*b')

    indices_chosen  = indices_right;
    [B,I] = sort([wheel_reset_left,wheel_reset_right]);
    indices_left = find(diff(I) == 1 & I(1:end-1) < length(wheel_reset_left) & diff(B) < 150);
    indices_right = (find(diff(I) == 1 & I(1:end-1) > length(wheel_reset_left)))-length(wheel_reset_left);
    plot(B(indices_left)/sr_wheel_pos,2*pi*ones(length(indices_left),1),'*y'),
    plot(B(indices_left+1)/sr_wheel_pos,zeros(length(indices_left),1),'*k')
    % plot(wheel_reset_right,2*pi*ones(length(wheel_reset_right),1),'*b'),
    % plot(wheel_reset_right-1,zeros(length(wheel_reset_right),1),'*b')
    p = [];
    figure(3),
    for i = 1:length(indices_left)
        x_span = length(B(indices_left(i)):B(indices_left(i)+1)-1);
        subplot(3,1,1)
        plot(wheel_pos_polar(B(indices_left(i)):B(indices_left(i)+1)-1)), hold on
        subplot(3,1,2)
        plot(wheel_pos_polar(B(indices_left(i)):B(indices_left(i)+1)-1),2*pi*[x_span:-1:1]/x_span), hold on
        p(i,:) = polyfit(wheel_pos_polar(B(indices_left(i)):B(indices_left(i)+1)-1),2*pi*[x_span:-1:1]/x_span,8);
    end
    plot(2*pi*[0,1],2*pi*[0,1],'k','linewidth',2)
    p2 = mean(p);
    x1 = wheel_pos_polar(B(indices_left(i)):B(indices_left(i)+1)-1);
    y1 = polyval(p2,x1);
    plot(x1,y1,'k','linewidth',2)
    subplot(3,1,3)
    for i = 1:length(indices_left)
        x_span = length(B(indices_left(i)):B(indices_left(i)+1)-1);
        plot(2*pi*[x_span:-1:1]/x_span,polyval(p2,wheel_pos_polar(B(indices_left(i)):B(indices_left(i)+1)-1))), hold on
    end
    plot(2*pi*[0,1],2*pi*[0,1],'k','linewidth',2)
    wheel_pos_polar_call = polyval(p2,wheel_pos_polar);
else
    wheel_pos_polar_call = wheel_pos_polar;
end
figure(2),ax1(1) = subplot(3,1,1)
plot([1:length(wheel_pos_polar_call)]/sr_wheel_pos,wheel_pos_polar_call,'.k')
ylabel('Position (rad)')

wheel_velocity = (diff(wheel_pos_polar_call));
while sum(abs(wheel_velocity) > 3/2*pi)
    wheel_velocity(find(wheel_velocity > pi)) = 2*pi-wheel_velocity(wheel_velocity > pi);
    wheel_velocity(find(wheel_velocity < -pi)) = 2*pi+wheel_velocity(wheel_velocity < -pi);
end
while sum(abs(wheel_velocity) > 0.2)
    wheel_velocity(find((wheel_velocity) > 0.2)) = wheel_velocity(find((wheel_velocity) > 0.2)-1);
    wheel_velocity(find((wheel_velocity) < -0.2)) = wheel_velocity(find((wheel_velocity) < -0.2)-1);
end
wheel_rad = 14.86; % Radius of the wheel in cm
wheel_velocity = sr_wheel_pos*wheel_rad*conv(wheel_velocity,gausswin(250)/sum(gausswin(250)),'same');
figure(2)
ax1(2) = subplot(3,1,2); plot([1:length(wheel_velocity)]/sr_wheel_pos,wheel_velocity)
ylabel('Velocity (cm/s)')
linkaxes(ax1,'x')
figure(2)
ax1(3) = subplot(3,1,3); plot([1:length(temperature)]/sr_wheel_pos,temperature)
xlabel('Time (s)'),ylabel('Temperature'),ylim([22,40])
linkaxes(ax1,'x'), axis tight
%% Correlating the speed of the wheel and temperature with the theta power and frequency
wheel_speed = abs(wheel_velocity);
speed_thres = 10;

x_start = find(diff(wheel_speed > speed_thres)==1);
x_stop = find(diff(wheel_speed > speed_thres)==-1);

wheel_periods = x_stop-x_start;
wheel_periods_min = 500;
wheel_periods2 = find(wheel_periods > wheel_periods_min);

temperature_cooling_thres = 26;
temperature_nocooling_thres = 35;
cooling_start = find(diff(temperature < temperature_cooling_thres)==1);
cooling_stop = find(diff(temperature < temperature_cooling_thres)==-1);
cooling = [cooling_start(1:end-1);cooling_stop(2:end)]/sr_wheel_pos;

nocooling_start = find(diff(temperature > temperature_nocooling_thres)==1);
nocooling_stop = find(diff(temperature > temperature_nocooling_thres)==-1);
nocooling = [nocooling_start;nocooling_stop]/sr_wheel_pos;

lfp_hippocampus = [];
wheel_speed_periods = [];
wheel_temperature_periods = [];
wavelets_lfp = [];
Fc2 = [50];
[b1,a1]=butter(3,Fc2*2/sr,'low'); % 'high' pass filter (high,low,bandpass)
downsample_n1 = 200;
freqlist = [4:0.1:12]; %10.^(0.4771:0.01:1.1761);
for i = 1:length(wheel_periods2)
    start = x_start(wheel_periods2(i))/sr_wheel_pos;
    duration = (x_stop(wheel_periods2(i))-x_start(wheel_periods2(i)))/sr_wheel_pos;
    lfp_wheel_temp = 0.000050354 * double(LoadBinary(fname,'nChannels',nbChan,'channels',ch_lfp,'precision','int16','frequency',sr,'start',start,'duration',duration));
    lfp_filt = filtfilt(b1,a1,lfp_wheel_temp);
    lfp_hippocampus{i} = downsample(lfp_filt,downsample_n1);
    wheel_speed_periods{i} = wheel_speed(x_start(wheel_periods2(i)):x_stop(wheel_periods2(i))-1);
    wheel_temperature_periods{i} = temperature(x_start(wheel_periods2(i)):x_stop(wheel_periods2(i))-1);
    [wt,~,~] = awt_freqlist(lfp_hippocampus{i},sr/downsample_n1,freqlist);
    wavelets_lfp{i} = abs(wt)';
end

%% % Analysing two running sessions with and without cooling
sessions = [6,7];%8,13 %13,24;
ix_state = {'With Cooling', 'Without Cooling'};
figure;
subplot(4,1,1), 
plot(cooling,[99,99],'b','linewidth',2), hold on
plot(nocooling,[99,99],'r','linewidth',2), legend({'Cooling'})
plot([1:length(wheel_speed)]/sr_wheel_pos,wheel_speed,'k')
plot([x_start(wheel_periods2);x_stop(wheel_periods2)]/sr_wheel_pos,[speed_thres;speed_thres],'o-'), 
plot([1:length(temperature)]/sr_wheel_pos,100*(temperature-20)/(38-20))
xlabel('Time (s)'),ylabel('Speed of running wheel (cm/s)'), title('Cooling experiment'), ylim([0,100])
for ix = 1:length(sessions)
    i = sessions(ix);
%     freqlist = 10.^(0.4771:0.02:1.1761);
%     [wt,freqlist,psi_array] = awt_freqlist(lfp_wheel{i},sr/downsample_n1,freqlist);
    t_axis = [1:length(lfp_hippocampus{i})]/(sr/downsample_n1);
    wt2 = zscore(wavelets_lfp{i});
    subplot(4,1,1)
    gridxy([x_start(wheel_periods2(i)),x_stop(wheel_periods2(i))]/sr_wheel_pos,'Color','m')
    
    ax2(1) = subplot(4,2,3+(ix-1)); plot((x_start(wheel_periods2(i)):x_stop(wheel_periods2(i)))/sr_wheel_pos-x_start(wheel_periods2(i))/sr_wheel_pos,wheel_speed(x_start(wheel_periods2(i)):x_stop(wheel_periods2(i))),'.r'),axis tight, xlabel('time (sec)')
    ax1 = ax2(1);
    ax1_pos = ax1.Position; % position of first axes
    ax3 = axes('Position',ax1_pos,'XAxisLocation','top','YAxisLocation','right','Color','none');
    line((x_start(wheel_periods2(i)):x_stop(wheel_periods2(i)))/sr_wheel_pos,wheel_speed(x_start(wheel_periods2(i)):x_stop(wheel_periods2(i))),'Parent',ax3,'color','w','linewidth',0.1)
    axis tight,ylabel('Speed (cm/s)'), title(ix_state{ix})
    ax2(2) = subplot(4,2,5+(ix-1)); 
    plot([1:length(lfp_hippocampus{i})]*downsample_n1/sr,lfp_hippocampus{i}), axis tight, title('Filtered LFP'), ylim([-0.2, 0.2])
    
    ax2(3) = subplot(4,2,7+(ix-1)); 
    surf(t_axis,freqlist,wt2,'EdgeColor','None'), axis tight, title('Spectrogram'), hold on
    plot3([t_axis(1),t_axis(end)],[6,6],[1000,1000],'k'),
    plot3([t_axis(1),t_axis(end)],[7,7],[1000,1000],'k'),
    plot3([t_axis(1),t_axis(end)],[8,8],[1000,1000],'k'),
    plot3([t_axis(1),t_axis(end)],[9,9],[1000,1000],'k'), hold off
    view(0,90)
    set(gca,'yscale','log')
    set(gca,'YTick',([1 2 3 4 5 6 7 8 9 10 12 20 30]))
    axis tight
    zheight = max(max(wt2));
    xlabel('Time (sec)')
    ylabel('Frequency (Hz)')
    caxis([-2 2])
    linkaxes(ax2,'x')
end
%% % Analysing all running times from the wheel to determine the powerspectrum of the lfp as a function of the running speed.
% Sessions with theta
if isempty(recordings(id).sessions_theta)
    sessions_theta = 1:length(wheel_periods2);
else
    sessions_theta = recordings(id).sessions_theta;
end
bins_speed = [10:4:75];
bins_speed_avg = mean([bins_speed(2:end);bins_speed(1:end-1)]);
bins_lfp = [4:0.1:12];%10.^(0.4771:0.01:1.1761);
sessions_all = 1:length(wheel_periods2);
cooling_wheel = x_start(wheel_periods2)/sr_wheel_pos;
% sessions_cooling = find((cooling_wheel > cooling(1,1) & cooling_wheel < cooling(1,2)) | (cooling_wheel > cooling(2,1) & cooling_wheel < cooling(2,2)));

sessions_cooling = [];
for i = 1:size(cooling,2)
    session_temp = find((cooling_wheel > cooling(1,i) & cooling_wheel < cooling(2,i)));
    if ~isempty(session_temp)
        sessions_cooling = [sessions_cooling,session_temp];
    end
end
sessions_nocooling = [];
for i = 1:size(nocooling,2)
    session_temp = find((cooling_wheel > nocooling(1,i) & cooling_wheel < nocooling(2,i)));
        if ~isempty(session_temp)
        sessions_nocooling = [sessions_nocooling,session_temp];
        end
end
%sessions_nocooling = sessions_all(~ismember([1:length(wheel_periods2)],sessions_cooling));

sessions_cooling2 = sessions_cooling(ismember(sessions_cooling,sessions_theta));
sessions_nocooling2 = sessions_nocooling(ismember(sessions_nocooling,sessions_theta));
% % % % % With Cooling
lfp_wavelets_combined = [];
for i = 1:length(bins_speed)-1
    lfp_wavelets_temp = [];
    for j= sessions_cooling2
        indices = find(wheel_speed_periods{j} > bins_speed(i) & wheel_speed_periods{j} <= bins_speed(i+1));
        if ~isempty(indices)
            lfp_wavelets_temp = [lfp_wavelets_temp;wavelets_lfp{j}(:,indices)'];
        end
%         horzcat(lfp_wheel{:})
    end
    if ~isempty(lfp_wavelets_temp)
        lfp_wavelets_combined(i,:) = mean(lfp_wavelets_temp);
    end
end
figure
subplot(3,1,1)
% surf(bins_lfp,bins_speed_avg,lfp_wavelets_combined,'EdgeColor','None'), view(0,90)
imagesc(bins_lfp,bins_speed_avg,lfp_wavelets_combined), set(gca,'Ydir','normal')
axis tight, title(['With Cooling (Temp < ' num2str(temperature_cooling_thres) 'C)']), xlabel('Powerspectrum (Hz)'),ylabel('Speed (cm/s)')
set(gca,'YTick',bins_speed), set(gca,'XTick',4:12), %set(gca,'xscale','log')
subplot(3,1,3)
stairs(bins_lfp,mean(lfp_wavelets_combined),'b'), hold on
% % % % % Without Cooling
lfp_wavelets_combined = [];
for i = 1:length(bins_speed)-1
    lfp_wavelets_temp = [];
    for j= sessions_nocooling2
        indices = find(wheel_speed_periods{j} > bins_speed(i) & wheel_speed_periods{j} <= bins_speed(i+1));
        lfp_wavelets_temp = [lfp_wavelets_temp;wavelets_lfp{j}(:,indices)'];
%         horzcat(lfp_wheel{:})
    end
    if ~isempty(lfp_wavelets_temp)
        lfp_wavelets_combined(i,:) = mean(lfp_wavelets_temp);
    end
end
subplot(3,1,2)
% surf(bins_lfp,bins_speed(1:end-1),lfp_wavelets_combined,'EdgeColor','None'), view(0,90)
imagesc(bins_lfp,bins_speed_avg,lfp_wavelets_combined), set(gca,'Ydir','normal')
axis tight, title(['Without Cooling (Temp > ' num2str(temperature_nocooling_thres) 'C)']), xlabel('Powerspectrum (Hz)'),ylabel('Speed (cm/s)')
set(gca,'YTick',bins_speed), set(gca,'XTick',4:12), axis tight, %set(gca,'xscale','log')
subplot(3,1,3)
stairs(bins_lfp,mean(lfp_wavelets_combined),'r'), axis tight, xlabel('Powerspectrum (Hz)')
legend('With Cooling','Without Cooling'), set(gca,'XTick',4:12), %set(gca,'xscale','log')
%% Theta frequency as a function of temperature
% 
% 
% 
disp('Importing hippocampus data')
Fc2 = [50];
[b1,a1]=butter(3,Fc2*2/sr,'low');
downsample_n1 = 200;
sr_lfp = sr/downsample_n1;
% freqlist = 10.^(0.6021:0.01:1.0792);
freqlist = 4:0.1:12;
% Hippocampus electrode
lfp_hippocampus2 = [];
wavelets_lfp = [];
for i = 1:nb_lfp_periods
    %start = cooling(i,1);
    %duration = cooling(i,2)-cooling(i,1);
    start = time_frame(1)+lfp_periods*(i-1);
    lfp_wheel_temp = 0.000050354 * double(LoadBinary(fname,'nChannels',nbChan,'channels',ch_lfp,'precision','int16','frequency',sr,'start',start,'duration',min(lfp_periods,time_frame(2)-time_frame(1)-(i-1)*lfp_periods)));
    lfp_filt = filtfilt(b1,a1,lfp_wheel_temp);
    lfp_hippocampus2{i} = downsample(lfp_filt,downsample_n1);
    [wt,~,~] = awt_freqlist(lfp_hippocampus2{i},sr_lfp,freqlist);
    wavelets_lfp{i} = abs(wt)';
end
lfp_hippocampus = [];
wt_hippocampus = [];
disp('Concatenating Hippocampus data')
for i = 1:length(wavelets_lfp)
    lfp_hippocampus = [lfp_hippocampus;lfp_hippocampus2{i}];
    wt_hippocampus = [wt_hippocampus,wavelets_lfp{i}];
end
% wt_full = zscore(wt_full);

% Medial Septum electrode
disp('Importing Medial Septum data')
lfp_medialseptum2 = [];
wavelets_MedialSeptum = [];
for i = 1:nb_lfp_periods
    %start = cooling(i,1);
    %duration = cooling(i,2)-cooling(i,1);
    start = time_frame(1)+lfp_periods*(i-1);
    lfp_wheel_temp = 0.000050354 * double(LoadBinary(fname,'nChannels',nbChan,'channels',ch_medialseptum,'precision','int16','frequency',sr,'start',start,'duration',min(lfp_periods,time_frame(2)-time_frame(1)-(i-1)*lfp_periods)));
    lfp_filt = filtfilt(b1,a1,lfp_wheel_temp);
    lfp_medialseptum2{i} = downsample(lfp_filt,downsample_n1);
    [wt,~,~] = awt_freqlist(lfp_medialseptum2{i},sr_lfp,freqlist);
    wavelets_MedialSeptum{i} = abs(wt)';
end
lfp_medialseptum = [];
wt_medialseptum = [];
disp('Concatenating Medial Septum data')
for i = 1:length(wavelets_MedialSeptum)
    lfp_medialseptum = [lfp_medialseptum;lfp_medialseptum2{i}];
    wt_medialseptum = [wt_medialseptum,wavelets_MedialSeptum{i}];
end

fname_digi = [datapath, recording, '/digitalin.dat'];
[digital_on,digital_off] = Process_IntanDigitalChannels(fname_digi);
camera_delay = []; %min(digital_on{1}(1),digital_off{1}(1))/sr;
save([datapath recording '/lfp_theta.mat'],'lfp_hippocampus','lfp_medialseptum','wt_hippocampus','wt_medialseptum','wavelets_lfp','freqlist','recordings','id','downsample_n1','Intan_rec_info','digital_on','digital_off','sr_lfp','camera_delay')
disp('lfp data import complete')
% % Importing temperature readings
load([datapath recording '/lfp_theta.mat'])
num_channels = length(Intan_rec_info.board_adc_channels); % ADC input info from header file
fileinfo = dir([datapath, recording,'/', 'analogin.dat']);
num_samples = fileinfo.bytes/(num_channels * 2); % uint16 = 2 bytes
fid = fopen([datapath, recording,'/', 'analogin.dat'], 'r');
v = fread(fid, [num_channels, num_samples], 'uint16');
fclose(fid);
v = v * 0.000050354; % convert to volts
downsample_n2 = 200;
wheel_pos = downsample(v(1,:),downsample_n2);
v_downsample = mean(reshape(v(2,1:end-mod(size(v,2),downsample_n2)),downsample_n2,[])); clear v;
v_downsample(v_downsample<1.25) = 1.25;
temperature = conv((v_downsample-1.25)/0.005,ones(1,100)/100,'same'); 
save([datapath recording '/temperature.mat'],'temperature')
sr_wheel_pos = Intan_rec_info.frequency_parameters.amplifier_sample_rate/downsample_n2;
%% % Running wheel analysis
while sum((wheel_pos < 0.1))
    wheel_pos(find(wheel_pos < 0.1)) = wheel_pos(find(wheel_pos < 0.1)-1);
end
wheel_pos_polar = 2*pi*(wheel_pos-min(wheel_pos))/(max(wheel_pos)-min(wheel_pos));
wheel_pos_polar_call = wheel_pos_polar;
wheel_velocity = (diff(wheel_pos_polar_call));
while sum(abs(wheel_velocity) > 3/2*pi)
    wheel_velocity(find(wheel_velocity > pi)) = 2*pi-wheel_velocity(wheel_velocity > pi);
    wheel_velocity(find(wheel_velocity < -pi)) = 2*pi+wheel_velocity(wheel_velocity < -pi);
end
while sum(abs(wheel_velocity) > 0.2)
    wheel_velocity(find((wheel_velocity) > 0.2)) = wheel_velocity(find((wheel_velocity) > 0.2)-1);
    wheel_velocity(find((wheel_velocity) < -0.2)) = wheel_velocity(find((wheel_velocity) < -0.2)-1);
end
wheel_rad = 14.86; % Radius of the wheel in cm
wheel_velocity = sr_wheel_pos*wheel_rad*conv(wheel_velocity,gausswin(250)/sum(gausswin(250)),'same');
wheel_speed = abs(wheel_velocity);
speed_thres = 20;

x_start = find(diff(wheel_speed > speed_thres)==1);
x_stop = find(diff(wheel_speed > speed_thres)==-1);

wheel_periods = x_stop-x_start;
wheel_periods_min = 500;
wheel_periods2 = find(wheel_periods > wheel_periods_min);

temperature_cooling_thres = 28;
temperature_nocooling_thres = 35;
cooling_start = find(diff(temperature < temperature_cooling_thres)==1);
cooling_stop = find(diff(temperature < temperature_cooling_thres)==-1);
cooling = [cooling_start(1:end-1);cooling_stop(2:end)]/sr_wheel_pos;
%cooling = [cooling_start;cooling_stop(2:end)]/sr_wheel_pos;
cooling_start = find(diff(temperature > temperature_nocooling_thres)==1);
cooling_stop = find(diff(temperature > temperature_nocooling_thres)==-1);
%nocooling = [cooling_start(1:end-1);cooling_stop(2:end)]/sr_wheel_pos;

temp_range = [floor(min(temperature(100:end-100))):0.5:min(ceil(max(temperature(100:end-100))),40)];
wt_range = [1:size(wt_hippocampus,2)];
temp_theta = [];
temp_theta_max = [];
for i = 1:(length(temp_range)-1)
    temp_index = find(temperature(wt_range) > temp_range(i) & temperature(wt_range) < temp_range(i+1) & wheel_speed(wt_range) > speed_thres);
    temp_theta(i,:) = mean(wt_hippocampus(:,temp_index),2);
    [~,temp_theta_max(i)] = max(mean(wt_hippocampus(:,temp_index),2));
end
% Hippocampus
figure
subplot(4,1,1)
plot(wt_range/sr_wheel_pos,wheel_speed(wt_range),'k'), hold on, axis tight, ylim([0,60])
ylabel('Speed (cm/s)')
title('Running speed (measured by wheel speed)')
subplot(4,1,2)
plot(wt_range/sr_wheel_pos,temperature(wt_range),'m'), axis tight, ylim([17,39])
xlabel('Time (seconds)')
ylabel('Temperature (C)')
title('Brain temperature during cooling (Medial Septum)')
subplot(2,3,4)
surf(temp_range(1:end-1),freqlist,temp_theta','EdgeColor','None'), hold on
view(0,90)
% set(gca,'YTick',([1 2 3 4 5 6 7 8 9 10 20 30]))
axis tight
ylim([freqlist(1), freqlist(end-3)])
caxis([0 max(max(temp_theta))])
ylabel('Theta frequency (Hz)')
title('Hippocampus')
xlabel('Brain temperature (C)')
subplot(2,3,6)
time = temp_range(find(temp_theta_max>2));
y1 = freqlist(temp_theta_max(find(temp_theta_max>2)));
plot(time,y1,'ob'), hold on
P = polyfit(time,y1,1);
yfit = P(1)*time+P(2);
plot(time,yfit,'b-');
text(22,8.8,['HPC slope: ', num2str(P(1),3)],'Color','blue'), axis tight
[r,p] = corr(time',y1')

% Medial Septum
temp_theta = [];
temp_theta_max = [];
for i = 1:(length(temp_range)-1)
    temp_index = find(temperature(wt_range) > temp_range(i) & temperature(wt_range) < temp_range(i+1) & wheel_speed(wt_range) > speed_thres);
    temp_theta(i,:) = mean(wt_medialseptum(:,temp_index),2);
    [~,temp_theta_max(i)] = max(mean(wt_medialseptum(:,temp_index),2));
end
subplot(2,3,5)
surf(temp_range(1:end-1),freqlist,temp_theta','EdgeColor','None'), hold on
view(0,90)
% set(gca,'YTick',([1 2 3 4 5 6 7 8 9 10 20 30]))
axis tight
ylim([freqlist(1), freqlist(end-3)])
%caxis([0 max(max(temp_theta))])
xlabel('Brain temperature (C)')
title('Medial Septum')

subplot(2,3,6)
time = temp_range(find(temp_theta_max>2));
y1 = freqlist(temp_theta_max(find(temp_theta_max>2)));
plot(time,y1,'*r')
P = polyfit(time,y1,1);
yfit = P(1)*time+P(2);
plot(time,yfit,'r--');
text(22,5.5,['MS slope: ', num2str(P(1),3)],'Color','red'), axis tight, hold off
[r,p] = corr(time',y1')
xlabel('Brain temperature (C)')
ylabel('Theta frequency (Hz)')
title('Cooling effects')
ylim([5,9.5])
%% % Calculating the coherence between MS and Hippocampus
%mscohere(lfp_hippocampus,lfp_medialseptum,[],[],[],sr_lfp);
time = [lfp_hippocampus,lfp_medialseptum]';
Fs = sr_lfp;
fMin = 4;
fMax = 12;
nbins = 40;
graphics = 1;
[ wave, f, t, coh, phases, raw, coi, scale, period, scalef ] = getWavelet( time, Fs, fMin, fMax, nbins, graphics );

temp_range = [floor(min(temperature(100:end-100))):0.5:ceil(max(temperature(100:end-100)))];
wt_range = [1:size(coh,2)];
temp_coh = [];
temp_coh_max = [];
for i = 1:(length(temp_range)-1)
    temp_index = find(temperature(wt_range) > temp_range(i) & temperature(wt_range) < temp_range(i+1) & wheel_speed(wt_range) > speed_thres);
    temp_coh(i,:) = mean(coh(:,temp_index),2);
    [~,temp_coh_max(i)] = max(mean(coh(:,temp_index),2));
end
figure
ax1(1) = subplot(6,1,1);
plot(wt_range/sr_wheel_pos,zscore(wheel_speed(wt_range)),'k'), hold on
plot(wt_range/sr_wheel_pos,zscore(temperature(wt_range)),'m'), axis tight
xlabel('Time (seconds)')
ylabel('Temperature (C)')
title('Brain temperature during cooling (Medial Septum)')
ax1(2) = subplot(6,1,[2,3]);
imagesc(wt_range/sr_lfp,freqlist,wt_hippocampus), hold on
set(gca,'YDir','normal'),axis tight
xlabel('Time')
ylabel('Frequency')
title('Wavelet spectrum')
caxis([0 max(max(wt_medialseptum))/5])
ax1(3) = subplot(6,1,4);
imagesc(t,f,coh), hold on
set(gca,'YDir','normal'),axis tight
xlabel('Time')
ylabel('Frequency')
title('Coherence')
linkaxes(ax1,'x')
% view(0,90)
% set(gca,'YTick',([1 2 3 4 5 6 7 8 9 10 20 30]))
% ylim([freqlist(1), freqlist(end-3)])
%caxis([0 max(max(temp_theta))])

subplot(3,2,5)
surf(temp_range(1:end-1),f,temp_coh','EdgeColor','None'), hold on
view(0,90)
% set(gca,'YTick',([1 2 3 4 5 6 7 8 9 10 20 30]))
axis tight
ylim([f(1), f(end)])
caxis([0 max(max(temp_coh))])
xlabel('Brain temperature (C)')
ylabel('Theta frequency (Hz)')
title('Coherence')

subplot(3,2,6)
time = temp_range(1:end-1);
y1 = f(temp_coh_max);
plot(time,y1,'*r'), hold on
%P = polyfit(x,y1,1);
% yfit = P(1)*x+P(2);
% plot(x,yfit,'r--');
% text(25,5.5,['MS slope: ', num2str(P(1),3)],'Color','red'), axis tight, hold off
[r,p] = corr(time',y1')
xlabel('Brain temperature (C)')
ylabel('Theta frequency (Hz)')
title('Cooling effect on the peak coherence')
ylim([5,9.5])
%% % Checking for time delay between the temperature and the change in theta frequency
[theta_power, theta_peak] = max(wt_medialseptum(5:end,:));
[theta_power3, theta_peak3] = max(wt_hippocampus(5:end,:));
% indexes = find(wheel_speed(range) > speed_thres)
temperature1 = mean(reshape(temperature(wt_range),[sr_lfp/10,length(wt_range)/sr_lfp*10]));
wheel_speed1 = mean(reshape(wheel_speed(wt_range),[sr_lfp/10,length(wt_range)/sr_lfp*10]));

theta_peak1 = mean(reshape(theta_peak,[sr_lfp/10,length(wt_range)/sr_lfp*10]));
theta_power1 = mean(reshape(theta_power,[sr_lfp/10,length(wt_range)/sr_lfp*10]));

theta_peak2 = mean(reshape(theta_peak3,[sr_lfp/10,length(wt_range)/sr_lfp*10]));
theta_power2 = mean(reshape(theta_power3,[sr_lfp/10,length(wt_range)/sr_lfp*10]));

indexes = find(temperature1 < 36);
maxlag = 10*10; % 10 Hz sampling rate
[acor1,lag] = xcorr(theta_peak1(indexes),temperature1(indexes),maxlag);
[bcor1,lag] = xcorr(theta_power1(indexes),temperature1(indexes),maxlag);
[acor2,lag] = xcorr(theta_peak2(indexes),temperature1(indexes),maxlag);
[bcor2,lag] = xcorr(theta_power2(indexes),temperature1(indexes),maxlag);
figure
subplot(1,2,1)
plot(lag/10,zscore(acor1),'b'), hold on
plot(lag/10,zscore(acor2),'r'),
legend({'Medial Septum','Hippocapus'}), gridxy(0)
xlabel('Time (seconds)')
ylabel('xcorr')
title('Cooling theta frequency delay')

subplot(1,2,2)
plot(lag/10,zscore(bcor1),'b'), hold on
plot(lag/10,zscore(bcor2),'r'),
legend({'Medial Septum','Hippocapus'}), gridxy(0)
xlabel('Time (seconds)')
ylabel('xcorr')
title('Cooling theta amplitude delay')

%% % Exporting mat files with temperature and speed for the state editor
disp('Exporting mat files for TheStateEditor')
StateEditor_Temperature = downsample(temperature,sr_lfp);
StateEditor_Temperature = StateEditor_Temperature(1:recording_length);
StateEditor_WheelSpeed =  downsample(wheel_speed,sr_lfp);
StateEditor_WheelSpeed = StateEditor_WheelSpeed(1:recording_length);
StateEditor_RunningSpeed_Temperature = [StateEditor_Temperature;StateEditor_WheelSpeed];
save([datapath recording '/StateEditor_RunningSpeed_Temperature.mat'],'StateEditor_RunningSpeed_Temperature')
save([datapath recording '/StateEditor_RunningSpeed.mat'],'StateEditor_WheelSpeed')
save([datapath recording '/StateEditor_Temperature.mat'],'StateEditor_Temperature')
disp('Exporting mat files for TheStateEditor complete!')
%% % Reading in the accelerometer data
fileinfo = dir([datapath, recording,'/', 'auxiliary.dat']);
num_channels = 3;
num_samples = fileinfo.bytes/(num_channels); % uint16 = 2 bytes
fid = fopen([datapath, recording,'/', 'auxiliary.dat'], 'r');
v = fread(fid, [num_channels, num_samples], 'uint16');
fclose(fid);
v = v * 0.000050354; % convert to volts
move_threshold = 0.02;
downsample_n2 = 200;
sr_downsample = sr/downsample_n2;
acc_data = [];
acc_data2 = [];
for i = 1:num_channels
    acc_data(i,:) = conv(v(i,:),ones(1,downsample_n2),'same')/(downsample_n2);
end
acc_data2 = conv(abs(diff(downsample(sum(acc_data),downsample_n2))),ones(1,100),'same')/100;
clear v, clear acc_data,
figure
plot([1:length(acc_data2)]/sr_downsample,acc_data2), hold on, plot([1,length(acc_data2)]/sr_downsample,[move_threshold,move_threshold])

movement = find(acc_data2>move_threshold);
save([datapath recording '/movement.mat'],'movement','acc_data2')
%% % Plot
% load([datapath recording '/lfp_theta.mat'])
% load([datapath recording '/temperature.mat'])
% % load([datapath recording '/movement.mat'])
cooling_onsets = recordings(id).cooling_onsets;
wt_range = [1:size(wt_hippocampus,2)];
% movement2 = movement(find(movement<wt_range(end)));
% temperature2 = conv(temperature,ones(1,100),'same')/100;
% temp1 = temperature(movement2);
[~,theta1] = max(wt_hippocampus);
conv_width = 1000;
theta_plot = conv(freqlist(theta1(theta1>5 & theta1<55)),ones(1,conv_width),'same')/conv_width;
figure(1)
subplot(3,2,1)
plot(wt_range(theta1>5 & theta1<55)/sr_lfp,theta_plot,'k'), ylim([5.5,8.5])
title('Theta across full session'), ylabel('Frequency (Hz)'),xlabel('Time (s)'), axis tight
subplot(3,2,2)
plot([wt_range]/sr_lfp,temperature(wt_range),'r'), hold on
% plot(movement2/sr_lfp,conv(temp1,ones(1,conv_width),'same')/conv_width,'m'), ylim([min(temperature(200:end-200)),max(temperature(200:end-200))])
title('Temperature across full session'), ylabel('Temperature (C)'),xlabel('Time (s)'), axis tight
subplot(3,1,2:3)
plot(wt_range(theta1>5 & theta1<55)/sr_lfp,zscore(conv(freqlist(theta1(theta1>5 & theta1<55)),ones(1,conv_width),'same')/conv_width),'-k'), hold on
gridxy(cooling_onsets,'color',[0.8,0.8,0.8])
%plot(movement2/sr_lfp,zscore(conv(temp1,ones(1,conv_width),'same')/conv_width),'.m')
plot([wt_range]/sr_lfp,zscore(temperature(wt_range)),'r','linewidth',2), hold on, axis tight, ylim([-3,3])
title('Temperature and Theta frequency across full session'), ylabel('Z-scored'),xlabel('Time (s)')
%%
temp_sessions = [];
theta_sessions = [];
theta_sessions_indexes = [];
t_before = -300;
t_after = 900;
bin_size = 10;
t_binning = [t_before:bin_size:t_after];
theta_binned = [];
theta_binned_av = [];
figure(2)
for i = 1:length(cooling_onsets)
    cool_window = [sr_lfp*cooling_onsets(i)+t_before*sr_lfp,sr_lfp*cooling_onsets(i)+t_after*sr_lfp];
    % Temperature
    temp_sessions(i,:) = temperature(cool_window(1):cool_window(2));
    % Theta frequency
    [~,theta_sessions_temp] = max(wt_hippocampus(:,cool_window(1):cool_window(2)));
    % movement_indexes = find(movement2 >cool_window(1) & movement2 < cool_window(2));
    theta_indexes = find(theta_sessions_temp>2 & theta_sessions_temp<55);
    % theta_indexes_temp = ismember(theta_indexes,movement2(movement_indexes)-cool_window(1));
    % theta_sessions_indexes{i} = theta_indexes(theta_indexes_temp);
    theta_sessions_indexes{i} = theta_indexes;
    %theta_sessions_indexes{i}(theta_sessions_indexes{i}==0) = [];
    theta_sessions{i} = theta_sessions_temp(theta_sessions_indexes{i});
    % Binning the data
    for j = 1:length(t_binning)-1
        temp_bin = find((theta_sessions_indexes{i} > (t_binning(j)-t_before)*sr_lfp ) & (theta_sessions_indexes{i} < (t_binning(j+1)-t_before)*sr_lfp));
        if ~isempty(temp_bin)
            theta_binned(i,j) = mean(freqlist(theta_sessions{i}(temp_bin)));
        else
            theta_binned(i,j) = 0;
        end
        if i == length(cooling_onsets)
            theta_binned_av(j) = mean(theta_binned(theta_binned(:,j)~=0,j));
        end
    end
    % Plotting theta
    ax1(1) = subplot(3,2,2);
    plot(theta_sessions_indexes{i}/sr_lfp+t_before,conv(freqlist(theta_sessions{i}),ones(1,conv_width),'same')/conv_width,'-'), hold on
end
title(['Theta peak frequency (Rat ' recording(7:9) ' ' recording(11:end-7) ')']), ylabel('Frequency (Hz)'),xlabel('Time (s)')
ylim([5,8.5]), gridxy(0,'color',[0.8,0.8,0.8]), hold off
ax1(2) = subplot(3,2,1);
plot([t_before*sr_lfp:t_after*sr_lfp]/sr_lfp,temp_sessions), hold on, axis tight
plot([t_before*sr_lfp:t_after*sr_lfp]/sr_lfp,mean(temp_sessions),'k','linewidth',2),gridxy(0,'color',[0.8,0.8,0.8])
ylabel('Temp (C)'),xlabel('Time (s)')
ax1(3) = subplot(3,2,3);
plot(t_binning(1:end-1)+bin_size/2,theta_binned,'-'), hold on
plot(t_binning(1:end-1)+bin_size/2,theta_binned_av,'-k')
ylim([5,8]),gridxy(0,'color',[0.8,0.8,0.8]), hold off
title('Binned Theta'), ylabel('Frequency (Hz)'),xlabel('Time (s)')
ax1(4) = subplot(3,2,4);
plot(t_binning(1:end-1)+bin_size/2,zscore(theta_binned_av),'-k'), hold on
plot([t_before*sr_lfp:t_after*sr_lfp]/sr_lfp,zscore(mean(temp_sessions)),'k','linewidth',2),gridxy(0,'color',[0.8,0.8,0.8]), hold off
title('Temperature vs Theta frequency'), ylabel('Z-scored'),xlabel('Time (s)')
linkaxes([ax1],'x'), xlim([-250,900]),gridxy(0,'color',[0.8,0.8,0.8])

% Fitting the temperature decay to estimate the time constant
t_diff1 = 60; % In seconds
temp_ave = mean(temp_sessions);
[temp_min,temp_index] = min(temp_ave);
Temp = temp_ave((-t_before+t_diff1)*sr_lfp:temp_index)-temp_min;
time = (1:length(Temp))/sr_lfp;
f = fit(time',Temp','exp1')
t_constant =-1/f.b;
figure(2) 
subplot(3,2,1)
title(['Temperature (Tau = ' num2str(t_constant,3) 's)'])
subplot(3,2,5)
plot(f,time,Temp)
xlabel('Time (s)'),ylabel('Temperature (C)'), axis tight, title(['Fitting Temperature (Shift: ' num2str(t_diff1) ' s)'])

t_diff2 = -2; % In seconds
theta_temp_index = round(temp_index/sr_lfp/bin_size);
theta = theta_binned_av(-round(t_before/bin_size)+t_diff2:theta_temp_index+t_diff2)-mean(theta_binned_av(theta_temp_index-7:theta_temp_index));
time2 = (1:length(theta))*bin_size-bin_size;
f2 = fit(time2',theta','exp1')
t_constant2 =-1/f2.b
subplot(3,2,3)
title(['Binned Theta (Tau = ' num2str(t_constant2,3) 's)'])
subplot(3,2,6)
plot(f2,time2,theta,'-o')
xlabel('Time (s)'),ylabel('Frequency (Hz)'), axis tight, title(['Fitting Theta (Shift: ' num2str(t_diff2*bin_size) 's)'])
