% Cooling during sleep
clear all
% datapath = '/Volumes/P/IntanData/'; % Location of the recording computer
% datapath = '/Volumes/TurtlePower/DataBank/Buzsakilab/';
datapath = 'G:\IntanData\';
Recordings_MedialSeptum
id = 21;
recording = recordings(id).name;
cd([datapath, recording, '/'])
Intan_rec_info = read_Intan_RHD2000_file_Peter([datapath, recording,'/']);
fname = [datapath, recording, '/amplifier.dat'];
nbChan = recordings(id).nbChan;
cooling = recordings(id).cooling;
sr = Intan_rec_info.frequency_parameters.amplifier_sample_rate;
lfp_channel = recordings(id).lfp;
time_frame = recordings(id).time_frame; 
lfp_periods = 30*60; % in seconds
nb_lfp_periods = ceil((time_frame(2)-time_frame(1))/lfp_periods);
temp_ = dir(fname);
recording_length = round(temp_.bytes/sr/nbChan/2)-1;
disp('Loading complete')
%%
disp('Importing lfp data')
Fc2 = [50];
[b1,a1]=butter(3,Fc2*2/sr,'low');
downsample_n1 = 200;
sr_lfp = sr/downsample_n1;
freqlist = 10.^(0.4771:0.01:1.1761);

for i = 1:nb_lfp_periods
    i
    %start = cooling(i,1);
    %duration = cooling(i,2)-cooling(i,1);
    start = time_frame(1)+lfp_periods*(i-1);
    lfp_wheel_temp = 0.000050354 * double(LoadBinary(fname,'nChannels',nbChan,'channels',lfp_channel,'precision','int16','frequency',sr,'start',start,'duration',min(lfp_periods,time_frame(2)-time_frame(1)-(i-1)*lfp_periods)));
    lfp_filt = filtfilt(b1,a1,lfp_wheel_temp);
    lfp_wheel{i} = downsample(lfp_filt,downsample_n1);
    [wt,~,~] = awt_freqlist(lfp_wheel{i},sr_lfp,freqlist);
    lfp_wavelets{i} = abs(wt)';
end
fname_digi = [datapath, recording, '/digitalin.dat'];
[digital_on,digital_off] = Process_IntanDigitalChannels(fname_digi);
camera_delay = min(digital_on{1}(1),digital_off{1}(1))/sr;
save([datapath recording '/lfp_cooling_sessions.mat'],'lfp_wheel','lfp_wavelets','freqlist','recordings','id','downsample_n1','Intan_rec_info','digital_on','digital_off','sr_lfp','camera_delay')
disp('Lfp data import complete')

% %% % Plotting the lfp and spectrogram analysis
% plots = 0;
% if plots == 1
%     load([datapath recording '/lfp_cooling_sessions.mat'])
%     sessions = [1:nb_lfp_periods];
%     figure;
%     for ix = 1:length(sessions)
%         i = sessions(ix);
%         t_axis = [1:length(lfp_wheel{i})]/sr_lfp+cooling(ix,1);
%         wt2 = zscore(lfp_wavelets{i});
%         ax2(2) = subplot(2,size(cooling,1),ix);
%         plot([1:length(lfp_wheel{i})]/sr_lfp+cooling(ix,1),lfp_wheel{i}), axis tight, title('Filtered LFP'), ylim([-0.2, 0.2])
%         
%         ax2(3) = subplot(2,size(cooling,1),size(cooling,1)+ix);
%         surf(t_axis,freqlist,wt2,'EdgeColor','None'), axis tight, title('Spectrogram'), hold on
%         plot3([t_axis(1),t_axis(end)],[6,6],[1000,1000],'k'),
%         plot3([t_axis(1),t_axis(end)],[7,7],[1000,1000],'k'),
%         plot3([t_axis(1),t_axis(end)],[8,8],[1000,1000],'k'),
%         plot3([t_axis(1),t_axis(end)],[9,9],[1000,1000],'k'), hold off
%         view(0,90)
%         set(gca,'yscale','log')
%         set(gca,'YTick',([1 2 3 4 5 6 7 8 9 10 12 20 30]))
%         axis tight
%         zheight = max(max(wt2));
%         xlabel('Time (sec)')
%         ylabel('Frequency (Hz)')
%         caxis([-2 2])
%         linkaxes(ax2,'x')
%     end
% end

%% Importing video and getting the tracking data
disp('Importing video for movement analysis')
tracking_data_path = 'G:\CameraData\'; % Windows work computer path
% tracking_data_path = '/Volumes/TurtlePower/DataBank/Buzsakilab/CameraData/VGAR3 16-07-15/'; % Macmini path
% tracking_data_filebase = 'fc2_save_2016-07-15-130447-0000';
tracking_data_filebase = recordings(id).tracking_file;

file1 = [tracking_data_path tracking_data_filebase];
vidObj = VideoReader(file1);

video_frame_diff = [];
video_frame = readFrame(vidObj);
figure(1)
imagesc(video_frame), hold on
[X_video,Y_video] = ginput(2);
plot(X_video,Y_video,'m','linewidth',2)
close(1)

X = [floor(X_video(1)):ceil(X_video(2))];
Y = [floor(Y_video(1)):ceil(Y_video(2))];
i = 0;
while hasFrame(vidObj)
    i = i+1;
    if mod(i,100)==0
        if i~=100
            backSp = repmat('\b',[1 length([num2str(i-100) ' frames processed'])]);
            fprintf(backSp)
        end
        fprintf('%i frames processed',i)
    end
    video_frame_pre = video_frame;
    video_frame = readFrame(vidObj);
    video_frame(1:2,1:4,:) = 0;
    video_frame_diff1 = video_frame(Y,X,:)-video_frame_pre(Y,X,:);
    video_frame_diff(i) = sum(video_frame_diff1(:));
end
save([datapath recording '/trackingWithCamera.mat'],'video_frame_diff','vidObj')
disp('Video import complete')
%%
disp('Perfoming movement analysis on video')
load([datapath recording '/trackingWithCamera.mat'])
load([datapath recording '/lfp_cooling_sessions.mat'])
gausswin_width = floor(vidObj.FrameRate)*30;
video_frame_diff2 = zscore(conv(video_frame_diff,gausswin(gausswin_width)/sum(gausswin(gausswin_width)),'same'));
video_frame_diff2 = video_frame_diff2-min(video_frame_diff2(gausswin_width:end-gausswin_width));

figure
subplot(2,1,2)
hist(video_frame_diff2,linspace(0,max(video_frame_diff2),1000)), hold on
xlim([0,3])
subplot(2,1,1)
time_axis = digital_off{1}(1:length(video_frame_diff2))/sr; % (1:length(video_frame_diff2))/vidObj.FrameRate;
plot(time_axis,video_frame_diff2), axis tight, ylim([0,3])
[~,video_threshold] = ginput(1);
test = find(video_frame_diff2<video_threshold);
test(find(test<gausswin_width))= [];
test(find(test>length(video_frame_diff2)-gausswin_width))= [];
hold on, plot(time_axis(test),video_frame_diff2(test),'.')
plot(recordings(id).cooling',[median(video_frame_diff2),median(video_frame_diff2)],'b','linewidth',2), legend({'CameraNoise','Rest','Cooling periods'})
subplot(2,1,2)
plot(video_threshold,0,'or')
subplot(2,1,1)
test2 = video_frame_diff2<video_threshold;
% test(find(test2<100))= [];
% test(find(test>length(video_frame_diff2)-gausswin_width/3))= [];
quiet = [];
quiet(:,1) = find(diff(test2) == 1);
quiet(:,2) = find(diff(test2) == -1)
quiet = [quiet(1:end-1,1),quiet(2:end,2)];
sleep_min_time = 30; % in seconds
quiet(find (time_axis(quiet(:,2))-time_axis(quiet(:,1))<sleep_min_time),:)= [];
plot(time_axis(quiet),[mean(video_frame_diff2),mean(video_frame_diff2)],'r','linewidth',2)
save([datapath recording '/quiet.mat'],'quiet','video_frame_diff2','time_axis')

motion_from_camera2 = resample(video_frame_diff2,1000,round(1000*vidObj.FrameRate));
StateEditor_motion_from_camera = [zeros(1,round(camera_delay)), motion_from_camera2,zeros(1,length(recording_length)-round(camera_delay)-length(motion_from_camera2))];
save([datapath recording '/StateEditor_MotionFromCamera.mat'],'StateEditor_motion_from_camera')

cooling_periods12 = (1:recording_length);
StateEditor_cooling_periods = zeros(1,recording_length);
for i = 1:size(cooling,1)
    StateEditor_cooling_periods(find(cooling_periods12>cooling(i,1) & cooling_periods12<cooling(i,2))) = 1;
end
save([datapath recording '/StateEditor_CoolingPeriods.mat'],'StateEditor_cooling_periods')
disp('Video analysis complete')
%% % Comparing the movement data from the camera with the eeg data
disp('Analysing LFP data and extracting theta')
load([datapath recording '/lfp_cooling_sessions.mat'])
load([datapath recording '/trackingWithCamera.mat'])
load([datapath recording '/quiet.mat'])
load([datapath recording '/StateEditor_CoolingPeriods.mat'])
load([datapath recording '/' recording '_SleepScore.mat'])
sr_lfp = sr/downsample_n1;
t_axis = [];
for i = 1:nb_lfp_periods
    t_axis = [t_axis,(1:length(lfp_wheel{i}))/(sr_lfp)+lfp_periods*(i-1)];
end

vector_cooling = zeros(1,length(t_axis));
vector_quiet = zeros(1,length(t_axis));
vector_quiet_and_cool = zeros(1,length(t_axis));

for i = 1:size(quiet,1)
    vector_quiet(find(t_axis>time_axis(quiet(i,1)) & t_axis<time_axis(quiet(i,2)))) = 1;
end

for i = 1:size(cooling,1)
    vector_cooling(find(t_axis>cooling(i,1) & t_axis<cooling(i,2))) = 1;
end
vector_quiet_and_cool = vector_quiet .* vector_cooling;

wt2 = zscore(lfp_wavelets{1});
wt_full = [];
for i = 1:length(lfp_wavelets)
    wt_full = [wt_full,lfp_wavelets{i}];
end
wt_full = zscore(wt_full);
disp_periods = (3600*sr_lfp:4200*sr_lfp);
t_axis2 = t_axis(disp_periods);
figure
surf(t_axis2,freqlist,wt_full(:,disp_periods),'EdgeColor','None'), axis tight, title('Spectrogram'), hold on,
view(0,90)
set(gca,'yscale','log')
set(gca,'YTick',([1 2 3 4 5 6 7 8 9 10 12 20 30]))
axis tight
% zheight = max(max(wt2));
xlabel('Time (sec)')
ylabel('Frequency (Hz)')
caxis([min(min(wt_full))/2, max(max(wt_full))/10])

%figure, plot(freqlist(13:42),mean(wt2(10:42,disp_periods),2))
vector_theta = [];
theta_max_all = [];
theta_i2 = [];
freqlist_selected = 12:42;
freqlist2 = freqlist(freqlist_selected);
for k = 1:size(lfp_wavelets,2)
    wt2 = zscore(lfp_wavelets{k});
    wt3 = zeros(size(wt2,2),length(freqlist_selected));
    for i = 1:length(freqlist_selected)
        wt3(:,i) = conv(wt2(freqlist_selected(i),:),ones(1,5*sr_lfp)/(5*sr_lfp),'same');
    end
    [theta_max,theta_i] = max(wt3');
    theta_max2 = (theta_max > 1);
    vector_theta = [vector_theta, theta_max2];
    theta_max_all = [theta_max_all, theta_max];
    theta_i2  = [theta_i2, theta_i];
end
vector_quiet_cooling_and_theta = vector_quiet .* vector_cooling .* vector_theta;
vector_quiet_nocooling_and_theta = vector_quiet .* (1-vector_cooling) .* vector_theta;

vector_theta_state = zeros(1,length(t_axis));
for i = 1:size(StateIntervals.REMstate,1)
    vector_theta_state(sr_lfp*StateIntervals.REMstate(i,1):sr_lfp*StateIntervals.REMstate(i,2)) = 1;
end
vector_theta_state = vector_theta_state(1:length(t_axis));
vector_cooling_theta_state = vector_theta_state .* vector_cooling;
vector_nocooling_theta_state = vector_theta_state .* (1-vector_cooling);
save([datapath recording '/vectors.mat'],'vector_quiet_cooling_and_theta','vector_theta','vector_cooling','vector_quiet','theta_max_all','theta_i2','freqlist2','vector_quiet_nocooling_and_theta','vector_cooling_theta_state','vector_nocooling_theta_state')

StateEditor_theta_periods = zeros(1,recording_length);
for i = 1:size(StateEditor_theta_periods,2)
    theta_periods2 = find(t_axis(logical(vector_theta)) > i-1 &  t_axis(logical(vector_theta)) < i);
    if isempty(theta_periods2)
        StateEditor_theta_periods(i) = freqlist2(1);
    else
        theta_periods3 = freqlist2(theta_i2(logical(vector_theta)));
        StateEditor_theta_periods(i) = mean(theta_periods3(theta_periods2));
    end
end
save([datapath recording '/StateEditor_Theta.mat'],'StateEditor_theta_periods')
StateEditor_CoolingAndTheta = [StateEditor_cooling_periods;StateEditor_theta_periods];
save([datapath recording '/StateEditor_CoolingAndTheta.mat'],'StateEditor_CoolingAndTheta')
disp('Complete analysing LFP data and extracting theta')
%%
load([datapath recording '\vectors.mat'])
load([datapath recording '\' recording '_SleepScore.mat'])
figure
ax1(1) = subplot(2,1,1);
plot(t_axis,vector_cooling,'-k','linewidth',2), title('States'), hold on
plot(t_axis,vector_quiet*0.8,'.m')
plot(t_axis,vector_theta*0.6,'.')
plot(t_axis,vector_quiet_cooling_and_theta*0.5,'.r')
plot(t_axis,vector_quiet_nocooling_and_theta*0.5,'.g')
plot(t_axis,vector_cooling_theta_state*0.4,'.r')
plot(t_axis,vector_nocooling_theta_state*0.4,'.g')
plot(StateIntervals.REMstate',ones(size(StateIntervals.REMstate'))*0.4,'k'),axis tight

legend({'Cooling','Quiet','Theta','Cooling, quiet & theta','No cooling but quiet and theta','REM (TheStateEditor)'})
ax1(2) = subplot(2,1,2);
% time_axis_video = (1:length(video_frame_diff2))/vidObj.FrameRate;
time_axis = digital_off{1}(1:length(video_frame_diff2))/sr;
plot(time_axis,video_frame_diff2,'-m'), title('Movement and theta'), xlim([0 t_axis(end)]), hold on
plot(t_axis(logical(vector_theta)),freqlist2(theta_i2(logical(vector_theta)))), axis tight, ylim([0,8])
plot(t_axis,(vector_quiet-0.1)*4,'.m')
plot(t_axis,vector_quiet_cooling_and_theta*3,'.r')
plot(t_axis,vector_quiet_nocooling_and_theta*3,'.g')
plot(t_axis,vector_cooling_theta_state*2.8,'.r')
plot(t_axis,vector_nocooling_theta_state*2.8,'.g')
plot(StateIntervals.REMstate',ones(size(StateIntervals.REMstate'))*2.8,'k')
linkaxes(ax1,'x'), xlabel('Time (s)')

%%
disp('Exporting theta for TheStateEditor...')
test = find(vector_theta);
test1 = test/sr_lfp;
theta_frames_intan = [];
theta_frames_video = [];
for i = 1:(length(time_axis)-1)
    i
    test3 = find(test1 > time_axis(i) & test1 < time_axis(i+1));
    theta_frames_intan = [theta_frames_intan,ones(1,length(test3))*i];
    theta_frames_video = [theta_frames_video,test(test3)];
end
test3 = find(time_axis(end) < test1 & (time_axis(end)+time_axis(end)-time_axis(end-1)) > test1);

theta_frames_intan = [theta_frames_intan,ones(1,length(test3))*i];
theta_frames_video = [theta_frames_video,test(test3)];
save([datapath recording '/theta_frames.mat'],'theta_frames_intan','theta_frames_video')
disp('Complete exporting theta for TheStateEditor')
%% % Plotting theta and amplitude for cooling and noncooling sessions
load([datapath recording '/vectors.mat'])
X1 = freqlist2(theta_i2(logical(vector_quiet_cooling_and_theta)));
Y1 = theta_max_all(logical(vector_quiet_cooling_and_theta));
% X1 = freqlist2(theta_i2(logical(vector_cooling_theta_state)));
% Y1 = theta_max_all(logical(vector_cooling_theta_state));
Bx = freqlist2;
By = linspace(1,max(theta_max_all),30);
N = hist2d(X1,Y1,Bx,By);

figure
subplot(3,1,1)
surf(Bx,By,N,'EdgeColor','None'), hold on
view(0,90)
% set(gca,'YTick',([1 2 3 4 5 6 7 8 9 10 20 30]))
axis tight
xlim([freqlist2(1), freqlist2(end-5)])
caxis([0 max(max(N(:,3:end)))])
xlabel('Frequency (Hz)')
ylabel('Amplitude')
title('Theta during REM with cooling')

subplot(3,1,2)
X2 = freqlist2(theta_i2(logical(vector_quiet_nocooling_and_theta)));
Y2 = theta_max_all(logical(vector_quiet_nocooling_and_theta));
% X2 = freqlist2(theta_i2(logical(vector_nocooling_theta_state)));
% Y2 = theta_max_all(logical(vector_nocooling_theta_state));
N2 = hist2d(X2,Y2,Bx,By);
surf(Bx,By,N2,'EdgeColor','None'), hold on
view(0,90)
axis tight
xlim([freqlist2(1), freqlist2(end-5)])
caxis([0 max(max(N2(:,2:end)))])
xlabel('Frequency (Hz)')
ylabel('Amplitude')
title('Theta during REM without cooling')

subplot(3,2,5)
hist1 = hist(X1,Bx);
hist2 = hist(X2,Bx);
stairs(Bx,hist1/sum(X1)), hold on
stairs(Bx,hist2/sum(X2)), xlim([freqlist2(1), freqlist2(end-5)])
[r,p] = kstest2(hist1/sum(X1),hist2/sum(X2));
title(['Frequency (KS-test r=', num2str(r),',p=',num2str(p),')']), xlabel('Frequency'),legend({'with cooling','without cooling'})
subplot(3,2,6)
hist1 = hist(Y1,By);
hist2 = hist(Y2,By);
stairs(By,hist1/sum(Y1)), hold on
stairs(By,hist2/sum(Y2)), xlim([1,ceil(max((theta_max_all)))])
[r,p] = kstest2(hist1/sum(Y1),hist2/sum(Y2));
title(['Amplitude (KS-test r=', num2str(r),',p=',num2str(p), ')']), xlabel('Amplitude')

%% % Plotting theta frequency and amplitude vs movements
load([datapath recording '/theta_frames.mat'])
load([datapath recording '\' recording '_SleepScore.mat'])
X1 = freqlist2(theta_i2(theta_frames_video));
Y1 = video_frame_diff2(theta_frames_intan);
Bx = freqlist2;
By = linspace(min(video_frame_diff2(theta_frames_intan)),max(video_frame_diff2(theta_frames_intan)),50);
N = hist2d(X1,Y1,Bx,By);
figure
subplot(2,1,1)
surf(Bx,By,N,'EdgeColor','None'), hold on
view(0,90)
axis tight
xlim([freqlist2(1), freqlist2(end-5)])
caxis([0 max(max(N(:,3:end)))])
xlabel('Frequency (Hz)')
ylabel('Video movement')
title('Theta frequency vs movement')

subplot(2,1,2)
X2 = theta_max_all(theta_frames_video);
Y2 = video_frame_diff2(theta_frames_intan);
Bx = linspace(0.8,max(theta_max_all),30);
By = linspace(min(video_frame_diff2(theta_frames_intan)),max(video_frame_diff2(theta_frames_intan)),50);
N2 = hist2d(X2,Y2,Bx,By);
surf(Bx,By,N2,'EdgeColor','None'), hold on
view(0,90)
axis tight
xlim([0.8, max(theta_max_all)])
caxis([0 max(max(N2(:,2:end)))])
xlabel('Theta amplitude')
ylabel('Video movement')
title('Theta amplitude vs movement')
