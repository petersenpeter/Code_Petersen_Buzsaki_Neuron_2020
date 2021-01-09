% Medial Septum Cooling project
clear all
% datapath = '/Volumes/P/IntanData/'; % Location of the recording computer
% datapath = '/Volumes/TurtlePower/DataBank/Buzsakilab/';
datapath = 'G:\IntanData\';
Recordings_MedialSeptum
id = 25;

recording= recordings(id).name;
Intan_rec_info = read_Intan_RHD2000_file_Peter([datapath, recording,'/']);
fname = [datapath, recording, '/amplifier.dat'];
nbChan = recordings(id).nbChan;
cooling = recordings(id).cooling;
sr = Intan_rec_info.frequency_parameters.amplifier_sample_rate;
lfp_channel = recordings(id).lfp;
theta_start = 1*60; % 77 min into the recording (Peter_nb2_160426_130349)
% theta_start = 100*60;
theta_step_length = 5*60;
% theta_instances = [1:theta_step_length:86*60];
theta_instances = [1:theta_step_length:recordings(id).time_frame(2)];
theta_duration = 10;
% lfp = zeros(1,sr*length(theta_instances)*thera_duration);
for i = 1:length(theta_instances)
    disp(num2str(i))
    lfp2(:,i) = LoadBinary(fname,'nChannels',nbChan,'channels',lfp_channel,'precision','int16','frequency',sr,'start',theta_instances(i),'duration',theta_duration);
end
lfp = lfp2(:);
%%
Fc2 = [100];
[b1,a1]=butter(3,Fc2*2/sr,'low'); % 'high' pass filter (high,low,bandpass)
lfp_filt = filtfilt(b1,a1,double(lfp(:)));
downsample_n1 = 100;
lfp_filt = downsample(lfp_filt,downsample_n1);
t_axis = [1:length(lfp_filt)]/(sr/downsample_n1);

%%
% spectrogram(lfp_filt,5*sr/downsample_n,0,256,sr/downsample_n,'yaxis')
freqlist = [3:0.2:15]; %10.^[0:0.03:1.5];
[wt,freqlist,psi_array] = awt_freqlist(lfp_filt,sr/downsample_n1,freqlist);
wt2 = ((abs(wt)'));
figure(1); surf(t_axis/theta_duration,freqlist,wt2,'EdgeColor','None'), hold on
view(0,90)
set(gca,'yscale','log')
set(gca,'YTick',([1 2 3 4 5 6 7 8 9 10 20 30]))
axis tight
zheight = max(max(wt2'));

plot3([1:length(theta_instances);1:length(theta_instances)],[freqlist(1),freqlist(end)],[zheight,zheight],'w')
plot3(cooling'/theta_step_length,[freqlist(end),freqlist(end);freqlist(end),freqlist(end)],[zheight,zheight;zheight,zheight],'r','linewidth',2)
% plot3([cooling_offset,cooling_offset]/theta_step_length,[freqlist(1),freqlist(end)],[zheight,zheight],'r','linewidth',2)
xlabel('Time (20 sec intervals with 5 mins pauses)')
ylabel('Frequency (Hz)')
caxis([0 1500])

%% % Running wheel analysis
num_channels = length(Intan_rec_info.board_adc_channels); % ADC input info from header file
fileinfo = dir([datapath, recording,'/', 'analogin.dat']);
num_samples = fileinfo.bytes/(num_channels * 2); % uint16 = 2 bytes
fid = fopen([datapath, recording,'/', 'analogin.dat'], 'r');
v = fread(fid, [num_channels, num_samples], 'uint16');
fclose(fid);
v = v * 0.000050354; % convert to volts
downsample_n2 = 200;
wheel_pos = downsample(v,downsample_n2); clear v;
sr_wheel_pos = Intan_rec_info.frequency_parameters.amplifier_sample_rate/downsample_n2;
while sum((wheel_pos < 0.1))
    wheel_pos(find(wheel_pos < 0.1)) = wheel_pos(find(wheel_pos < 0.1)-1);
end
wheel_pos_polar = 2*pi*(wheel_pos-min(wheel_pos))/(max(wheel_pos)-min(wheel_pos));
%%
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
figure(2),subplot(2,1,1)
plot([1:length(wheel_pos_polar_call)]/sr_wheel_pos,wheel_pos_polar_call,'.k')
xlabel('Time (s)'),ylabel('Position (rad)')

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
ax1(2) = subplot(2,1,2); plot([1:length(wheel_velocity)]/sr_wheel_pos,wheel_velocity)
xlabel('Time (s)'),ylabel('Velocity (cm/s)')
linkaxes(ax1,'x')
%% Correlating the speed of the wheel with the theta power and frequency
wheel_speed = abs(wheel_velocity);
speed_thres = 10;
x_start = find(diff(wheel_speed > speed_thres)==1);
x_stop = find(diff(wheel_speed > speed_thres)==-1);

wheel_periods = x_stop-x_start;
wheel_periods_min = 500;
wheel_periods2 = find(wheel_periods > wheel_periods_min);

lfp_wheel = [];
wheel_speed_periods = [];
lfp_wavelets = [];
Fc2 = [50];
[b1,a1]=butter(3,Fc2*2/sr,'low'); % 'high' pass filter (high,low,bandpass)
downsample_n1 = 200;
freqlist = 10.^(0.4771:0.015:1.1761);
for i = 1:length(wheel_periods2)
    start = x_start(wheel_periods2(i))/sr_wheel_pos;
    duration = (x_stop(wheel_periods2(i))-x_start(wheel_periods2(i)))/sr_wheel_pos;
    lfp_wheel_temp = 0.000050354 * double(LoadBinary(fname,'nChannels',nbChan,'channels',lfp_channel,'precision','int16','frequency',sr,'start',start,'duration',duration));
    lfp_filt = filtfilt(b1,a1,lfp_wheel_temp);
    lfp_wheel{i} = downsample(lfp_filt,downsample_n1);
    wheel_speed_periods{i} = wheel_speed(x_start(wheel_periods2(i)):x_stop(wheel_periods2(i))-1);
    [wt,~,~] = awt_freqlist(lfp_wheel{i},sr/downsample_n1,freqlist);
    lfp_wavelets{i} = abs(wt)';
end
figure, plot([1:length(lfp_wheel_temp)]/sr,lfp_wheel_temp), hold on
plot([1:length(lfp_wheel{i})]/sr_wheel_pos,lfp_wheel{i})
legend('unfiltered lfp', 'filtered lfp')

%% % Analysing two running sessions with and without cooling
sessions = [11,12];%8,13 %13,24;
ix_state = {'With Cooling', 'Without Cooling'};
figure;
subplot(4,1,1), plot([1:length(wheel_speed)]/sr_wheel_pos,wheel_speed,'k'), hold on
plot([x_start(wheel_periods2);x_stop(wheel_periods2)]/sr_wheel_pos,[speed_thres;speed_thres],'o-'), 
plot(recordings(id).cooling',[100,100],'b','linewidth',3), legend('Cooling periods')
xlabel('Time (s)'),ylabel('Speed of running wheel (cm/s)'), title('Cooling experiment')
for ix = 1:length(sessions)
    i = sessions(ix);
%     freqlist = 10.^(0.4771:0.02:1.1761);
%     [wt,freqlist,psi_array] = awt_freqlist(lfp_wheel{i},sr/downsample_n1,freqlist);
    t_axis = [1:length(lfp_wheel{i})]/(sr/downsample_n1);
    wt2 = zscore(lfp_wavelets{i});
    subplot(4,1,1)
    gridxy([x_start(wheel_periods2(i)),x_stop(wheel_periods2(i))]/sr_wheel_pos,'Color','m')
    
    ax2(1) = subplot(4,2,3+(ix-1)); plot((x_start(wheel_periods2(i)):x_stop(wheel_periods2(i)))/sr_wheel_pos-x_start(wheel_periods2(i))/sr_wheel_pos,wheel_speed(x_start(wheel_periods2(i)):x_stop(wheel_periods2(i))),'.r'),axis tight, xlabel('time (sec)')
    ax1 = ax2(1);
    ax1_pos = ax1.Position; % position of first axes
    ax3 = axes('Position',ax1_pos,'XAxisLocation','top','YAxisLocation','right','Color','none');
    line((x_start(wheel_periods2(i)):x_stop(wheel_periods2(i)))/sr_wheel_pos,wheel_speed(x_start(wheel_periods2(i)):x_stop(wheel_periods2(i))),'Parent',ax3,'color','w','linewidth',0.1)
    axis tight,ylabel('Speed (cm/s)'), title(ix_state{ix})
    ax2(2) = subplot(4,2,5+(ix-1)); 
    plot([1:length(lfp_wheel{i})]*downsample_n1/sr,lfp_wheel{i}), axis tight, title('Filtered LFP'), ylim([-0.2, 0.2])
    
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
sessions_theta = recordings(id).sessions_theta;

bins_speed = [15:5:75];
bins_speed_avg = mean([bins_speed(2:end);bins_speed(1:end-1)]);
bins_lfp = 10.^(0.4771:0.015:1.1761);
sessions_all = 1:length(wheel_periods2);
cooling_wheel = x_start(wheel_periods2)/sr_wheel_pos;
% sessions_cooling = find((cooling_wheel > cooling(1,1) & cooling_wheel < cooling(1,2)) | (cooling_wheel > cooling(2,1) & cooling_wheel < cooling(2,2)));
sessions_cooling = find((cooling_wheel > cooling(1,1) & cooling_wheel < cooling(1,2)));
sessions_nocooling = sessions_all(~ismember([1:length(wheel_periods2)],sessions_cooling));

sessions_cooling2 = sessions_cooling(ismember(sessions_cooling,sessions_theta));
sessions_nocooling2 = sessions_nocooling(ismember(sessions_nocooling,sessions_theta));
% % % % % With Cooling
lfp_wavelets_combined = [];
for i = 1:length(bins_speed)-1
    lfp_wavelets_temp = [];
    for j= sessions_cooling2
        indices = find(wheel_speed_periods{j} > bins_speed(i) & wheel_speed_periods{j} <= bins_speed(i+1));
        if ~isempty(indices)
            lfp_wavelets_temp = [lfp_wavelets_temp;lfp_wavelets{j}(:,indices)'];
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
axis tight, title('With Cooling'), xlabel('Powerspectrum (Hz)'),ylabel('Speed (cm/s)')
set(gca,'YTick',bins_speed), set(gca,'XTick',3:14), set(gca,'xscale','log')
subplot(3,1,3)
stairs(bins_lfp,mean(lfp_wavelets_combined),'b'), hold on
% % % % % Without Cooling
lfp_wavelets_combined = [];
for i = 1:length(bins_speed)-1
    lfp_wavelets_temp = [];
    for j= sessions_nocooling2
        indices = find(wheel_speed_periods{j} > bins_speed(i) & wheel_speed_periods{j} <= bins_speed(i+1));
        lfp_wavelets_temp = [lfp_wavelets_temp;lfp_wavelets{j}(:,indices)'];
%         horzcat(lfp_wheel{:})
    end
    if ~isempty(lfp_wavelets_temp)
        lfp_wavelets_combined(i,:) = mean(lfp_wavelets_temp);
    end
end
subplot(3,1,2)
% surf(bins_lfp,bins_speed(1:end-1),lfp_wavelets_combined,'EdgeColor','None'), view(0,90)
imagesc(bins_lfp,bins_speed_avg,lfp_wavelets_combined), set(gca,'Ydir','normal')
axis tight, title('Without Cooling'), xlabel('Powerspectrum (Hz)'),ylabel('Speed (cm/s)')
set(gca,'YTick',bins_speed), set(gca,'XTick',3:14), set(gca,'xscale','log'), axis tight
subplot(3,1,3)
stairs(bins_lfp,mean(lfp_wavelets_combined),'r'), axis tight, xlabel('Powerspectrum (Hz)')
legend('With Cooling','Without Cooling'), set(gca,'xscale','log'), set(gca,'XTick',3:14)
