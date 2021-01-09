% Batch analysis

% % % % % % % % % % % % % % % % % % % % % % % % % % % %
% Task performance and Running speed

clear all, close all
MedialSeptum_Recordings

% Control sets
% MS10:
% MS12:
% MS13:
% MS14: 107
% MS18: 116,141,142
% MS21: 143,154,159
% MS22: 144

datasets_MS10 = [58,59,60,61,62,66,71,72]; % 8 sessions. More? 182
datasets_MS12 = [77,78,79,80,81,82,83,84,86,179]; % 10 sessions.
datasets_MS13 = [90,91,92,93,94,173,174,175]; % 8 sessions. More? 176,178
datasets_MS14 = [108];
datasets_MS21 = [122,126,140,146,147,149,151,152,153,155,156,157,158,160,161]; % 15 sessions
datasets_MS22 = [123,127,139,145,162,163,164,165,166,167,168,169]; % 12 sessions

% New sessions (11-09-2019): 
% MS13: 173 OK,174 OK,175 OK,176 tracking missing,178,
% MS10: 181 no temperature?, 182 no temperature?
% MS14: ids 102 102 103 104 105 106 107 108
datasets = [datasets_MS10,datasets_MS12,datasets_MS13,datasets_MS21,datasets_MS22];
animalid2 = {datasets_MS10,datasets_MS12,datasets_MS13,datasets_MS21,datasets_MS22};
% datasets = [datasets_MS10];
% animalid2 = {datasets_MS10};
animal_names = {'MS10','MS12','MS13','MS21','MS22'};
animalid = [];
for i = 1:length(animalid2)
    animalid = [animalid,i*ones(1,length(animalid2{i}))];
end

% MS14: ids 102 102 103 104 105 106 107 108
trials_speed(1,5).cooling = [];
trials_speed(1,5).nocooling = [];
trials_batch = {};
for k = 1:length(datasets)
    recording = recordings(datasets(k));
    disp(['Processing ', num2str(k),'/',num2str(length(datasets)),' recordingID = ' num2str(datasets(k)),', basename = ' recording.name])
    cd(fullfile(datapath, recording.animal_id, recording.name))
    
    % Behavior
    load([recording.name, '.animal.behavior.mat'])
    
    % Cooling
    load([recording.name, '.cooling.manipulation.mat'])
    
    % Trials
    load([recording.name, '.trials.behavior.mat'])
    
    
    trials_batch.error{k} = trials.error;
    trials_batch.cooling{k} = trials.cooling;
    for j = 1 :  trials.total
        trials_batch.temperature{k}(j) = mean(animal.temperature(trials.start(j):trials.end(j)));
        trials_batch.speed{k}(j) = mean(animal.speed(trials.start(j):trials.end(j)));
    end
    
    temp1 = 0; temp2 = 0;
    for j = 1:size(cooling.cooling,2)
        temp1 = temp1 + length(find(animal.time(trials.start(trials.error)) > cooling.cooling(1,j) & animal.time(trials.start(trials.error)) < cooling.cooling(2,j)));
        temp2 = temp2 + length(find(animal.time(trials.start) > cooling.cooling(1,j) & animal.time(trials.start) < cooling.cooling(2,j)));
    end
    trials.Cooling_error_ratio = 100*temp1/temp2;
    trials.NoCooling_error_ratio_before = 100*length(find(animal.time(trials.start(trials.error)) < cooling.cooling(1)))/length(find(animal.time(trials.start) < cooling.cooling(1)));
    temp1 = 0; temp2 = 0;
    for j = 1:size(cooling.cooling,2)-1
        temp1 = temp1 + length(find(animal.time(trials.start(trials.error)) > cooling.cooling(2,j)+20 &  animal.time(trials.start(trials.error)) < cooling.cooling(1,j+1)));
        temp2 = temp2 + length(find(animal.time(trials.start) > cooling.cooling(2,j)+20 & animal.time(trials.start) < cooling.cooling(1,j+1)+20));
    end
    temp1 = temp1 + length(find(animal.time(trials.start(trials.error)) > cooling.cooling(2,end)+20));
    temp2 = temp2 + length(find(animal.time(trials.start) > cooling.cooling(2,end)+20));
    trials.NoCooling_error_ratio_after = 100*temp1/temp2;
    trials_errors(:,k) = [trials.NoCooling_error_ratio_before,trials.Cooling_error_ratio,trials.NoCooling_error_ratio_after];
    
    figure,
    bar(1, trials.NoCooling_error_ratio_before, 'red'), hold on
    bar(2, trials.Cooling_error_ratio, 'blue')
    bar(3, trials.NoCooling_error_ratio_after, 'red')
    xticks([1, 2, 3]), xticklabels({'Pre Cooling','Cooling','Post cooling'}),ylabel('Percentage of errors'),title([num2str(datasets(k)),': Error trials (%)']),axis tight,
    xlim([0,4]), ylim([0,30])
    
    % % % % % % % % % % % % % % % % % % % % % % % % % % % %
    % % Running speed
    temp1 = [];
    for j = 1:size(cooling.cooling,2)
        temp1 = [temp1, find(animal.time(trials.all) > cooling.cooling(1,j) & animal.time(trials.all) < cooling.cooling(2,j))];
    end
    trials_speed(animalid(k)).cooling = [trials_speed(animalid(k)).cooling,animal.speed(trials.all(temp1))];
    
    temp1 = [];
    temp1 = find(animal.time(trials.all) < cooling.cooling(1));
    for j = 1:size(cooling.cooling,2)-1
        temp1 = [temp1, find(animal.time(trials.all) > cooling.cooling(2,j)+20 &  animal.time(trials.all) < cooling.cooling(1,j+1))];
    end
    temp1 = [temp1, find(animal.time(trials.all) > cooling.cooling(2,end)+20)];
    trials_speed(animalid(k)).nocooling = [trials_speed(animalid(k)).nocooling,animal.speed(trials.all(temp1))];
    trials_speed(animalid(k)).name = recording.animal_id;
    
    % % % % % % % % % % % % % % % % % % % % % % % % % % % %
    % Theta
    theta.sr_freq = 10;
    InstantaneousTheta = calcInstantaneousTheta(recording);
    if length(InstantaneousTheta.timestamps)~= length(InstantaneousTheta.ThetaInstantFreq{recording.ch_theta})
        InstantaneousTheta = calcInstantaneousTheta(recording,'forceReload',true);
    end
%     if ~isfield(InstantaneousTheta,'signal_time')
%         InstantaneousTheta = calcInstantaneousTheta(recording,'forceReload',true);
%     end
    theta.phase = InstantaneousTheta.signal_phase{recording.ch_theta};
    theta.phase2 = InstantaneousTheta.signal_phase2{recording.ch_theta};
    theta.freq = InstantaneousTheta.signal_freq{recording.ch_theta};
    theta.power = InstantaneousTheta.signal_power{recording.ch_theta};
    theta.time = InstantaneousTheta.signal_time;
    disp('Loading theta per trial')
    for j = 1:trials.total
        idx = find(trials.trials2 == j & animal.speed > 30);
        trials_batch.theta_freq{k}(j) = nanmean(interp1(InstantaneousTheta.timestamps,InstantaneousTheta.ThetaInstantFreq{recording.ch_theta},animal.time(idx)));
        trials_batch.theta_power{k}(j) = nanmean(interp1(theta.time,theta.power,animal.time(idx)));
    end
%     load('trialsGamma.mat')
%     trials_batch.lowGamma_power{k} = trialsGamma.power.lowGamma;
%     trials_batch.midGamma_power{k} = trialsGamma.power.midGamma;
    trials_batch.start{k} = animal.time(round(mean([trials.start;trials.end])));
    trials_batch.cooling_onset(k) = cooling.onsets(1);
end

figure,
subplot(2,1,1)
plot([1,2,3], trials_errors,'.-','markersize',8), hold on
xticks([1, 2, 3]), xticklabels({'Pre Cooling','Cooling','Post cooling'}),ylabel('Percentage of errors'),title('Error trials (%)'),axis tight,
[p1,h1] = signrank(trials_errors(1,:),trials_errors(2,:));
[p2,h2] = signrank(trials_errors(2,:),trials_errors(3,:));
[p3,h3] = signrank(trials_errors(1,:),trials_errors(3,:));
text(1.2,40,[num2str(p1),',  ',num2str(h1)],'Rotation',45);
text(2.8,40,[num2str(p2),'  ,',num2str(h2)],'Rotation',45);
text(2,40,[num2str(p3),',  ',num2str(h3)],'Rotation',45);
xlim([0,4]),ylim([0,40])
subplot(2,1,2)
boxplot(trials_errors')
xticks([1, 2, 3]), xticklabels({'Pre Cooling','Cooling','Post cooling'}),ylabel('Percentage of errors')

figure
bins_speed_size = 3;
bins_speed = [0:bins_speed_size:160];
for k = 1:size(trials_speed,2)
    subplot(size(trials_speed,2),1,k)
    temp1 = histcounts(trials_speed(k).cooling,'BinEdges',bins_speed,'Normalization','probability');
    temp2 = histcounts(trials_speed(k).nocooling,'BinEdges',bins_speed,'Normalization','probability');
    plot(bins_speed(1:end-1)+bins_speed_size/2,temp1,'b'), hold on,
    plot(bins_speed(1:end-1)+bins_speed_size/2,temp2,'r')
    [h,p] = kstest2(trials_speed(k).cooling,trials_speed(k).nocooling); ylim([0,0.1])
    xlabel('Speed (cm/s)'), ylabel('Probability'), title(['Speed during the trials (Animal ', num2str(trials_speed(k).name), ', h,p= ', num2str(h),', ',num2str(p) , ')']),xlim([bins_speed(1),bins_speed(end)])
end

figure
plt1 = [];
colors = {'r','b','g','m','c'};
for k = 1:length(unique(animalid))
    
    subplot(1,4,1); hold on
    plot([1,2,3],trials_errors(:,animalid==k),colors{k}), hold on
    xticks([1, 2, 3]), xticklabels({'Pre','Cooling','Post'}),ylabel('Percentage of errors'),title('Error rate, sessions'),axis tight, xlim([0.5, 3.5]), ylim([0,40])
    
    subplot(1,4,2); hold on
    trials_errors_mean = mean(trials_errors(:,animalid==k)');
    trials_errors_std = sem(trials_errors(:,animalid==k)');
    patch([1,2,3,3,2,1], [trials_errors_mean+trials_errors_std,flip(trials_errors_mean-trials_errors_std)],colors{k},'EdgeColor','none','FaceAlpha',.2)
    plt1(k) = plot([1,2,3],trials_errors_mean,colors{k}); hold on
    xticks([1, 2, 3]), xticklabels({'Pre','Cooling','Post'}), ylim([0,40]), xlim([0.5, 3.5])
    testdata = trials_errors(:,animalid==k)';
    [p1,h1] = signrank(testdata(:,1),testdata(:,2));
    [p2,h2] = signrank(testdata(:,2),testdata(:,3));
    [p3,h3] = signrank(testdata(:,1),testdata(:,3));
    text(1,42-3*k,['1,2: ',num2str(p1),',  ',num2str(h1)]);
    text(1.5,42-3*k-1,['2,3: ',num2str(p2),'  ,',num2str(h2)]);
    text(2,42-3*k-2,['1,3: ',num2str(p3),',  ',num2str(h3)]);
end

legend(plt1, animal_names), title('Animal specific error rate')
subplot(1,4,3), hold on
trials_errors_mean = mean(trials_errors');
trials_errors_std = sem(trials_errors');
patch([1,2,3,3,2,1], [trials_errors_mean+trials_errors_std,flip(trials_errors_mean-trials_errors_std)],'k','EdgeColor','none','FaceAlpha',.2)
plt1(k) = plot([1,2,3],trials_errors_mean,'k'); hold on
xticks([1, 2, 3]), xticklabels({'Pre','Cooling','Post'}), title('Error rate across animals'), ylim([0,40]), xlim([0.5, 3.5])
subplot(1,4,4), hold on
boxplot(trials_errors')
xticks([1, 2, 3]), xticklabels({'Pre Cooling','Cooling','Post cooling'}),ylabel('Percentage of errors'), ylim([0,40]), xlim([0.5, 3.5])


%%
% Error rate during the task (trial dependent)
% trials_batch.theta_freq
figure, 
imagesc(animalid')


error_continuous  = nan(length(trials_batch.error),max(max([trials_batch.error{:}]),280));
error_continuous_smooth  = nan(length(trials_batch.error),max(max([trials_batch.error{:}]),280));
speed_continuous = nan(length(trials_batch.error),max([trials_batch.error{:}]));
temperature_continuous = nan(length(trials_batch.error),max([trials_batch.error{:}]));
theta_freq_continuous = nan(length(trials_batch.error),max([trials_batch.error{:}]));
theta_power_continuous = nan(length(trials_batch.error),max([trials_batch.error{:}]));
% lowGamma_power_continuous = nan(length(trials_batch.error),max([trials_batch.error{:}]));
% midGamma_power_continuous = nan(length(trials_batch.error),max([trials_batch.error{:}]));
time_start_continuous = nan(length(trials_batch.error),max([trials_batch.error{:}]));

for i = 1:length(trials_batch.error)
    trial_offset = 65-find(trials_batch.cooling{i}==2,1);
    if ~isempty(intersect(find(trials_batch.cooling{i}==3),find(trials_batch.cooling{i}==2)-1))
        trial_end = intersect(find(trials_batch.cooling{i}==3),find(trials_batch.cooling{i}==2)-1);
    else
        trial_end = length(trials_batch.cooling{i});
    end
    error_continuous(i,[1:length(trials_batch.speed{i}(1:trial_end))]+trial_offset) = 0;
    trials_batch4 = trials_batch.error{i}(find(trials_batch.error{i}<trial_end));
    error_continuous(i,trials_batch4 + trial_offset) = 1;
    error_continuous_smooth(i,:) = nanconv(error_continuous(i,:),gausswin(11)'/sum(gausswin(11)),'same');
    speed_continuous(i,[1:length(trials_batch.speed{i}(1:trial_end))]+trial_offset) = trials_batch.speed{i}(1:trial_end);
    temperature_continuous(i,[1:length(trials_batch.temperature{i}(1:trial_end))]+trial_offset) = trials_batch.temperature{i}(1:trial_end);
    theta_freq_continuous(i,[1:length(trials_batch.theta_freq{i}(1:trial_end))]+trial_offset) = trials_batch.theta_freq{i}(1:trial_end);
    theta_power_continuous(i,[1:length(trials_batch.theta_power{i}(1:trial_end))]+trial_offset) = trials_batch.theta_power{i}(1:trial_end);
%     lowGamma_power_continuous(i,[1:length(trials_batch.lowGamma_power{i}(1:trial_end))]+trial_offset) = trials_batch.lowGamma_power{i}(1:trial_end);
%     midGamma_power_continuous(i,[1:length(trials_batch.midGamma_power{i}(1:trial_end))]+trial_offset) = trials_batch.midGamma_power{i}(1:trial_end);
    
    time_start_continuous(i,[1:length(trials_batch.start{i}(1:trial_end))]+trial_offset) = trials_batch.start{i}(1:trial_end)-trials_batch.cooling_onset(i);
end

index_end = 200;
error_continuous = error_continuous(:,1:index_end);
error_continuous_smooth = error_continuous_smooth(:,1:index_end);
speed_continuous = speed_continuous(:,1:index_end);
temperature_continuous = temperature_continuous(:,1:index_end);
theta_freq_continuous = theta_freq_continuous(:,1:index_end);
theta_power_continuous = theta_power_continuous(:,1:index_end); theta_power_continuous = theta_power_continuous - nanmean(theta_power_continuous,2);
% lowGamma_power_continuous = lowGamma_power_continuous(:,1:index_end); lowGamma_power_continuous = (lowGamma_power_continuous - nanmean(lowGamma_power_continuous,2))./(nanstd(lowGamma_power_continuous')'*ones(1,index_end));
% midGamma_power_continuous = midGamma_power_continuous(:,1:index_end); midGamma_power_continuous = (midGamma_power_continuous - nanmean(midGamma_power_continuous,2))./(nanstd(midGamma_power_continuous')'*ones(1,index_end));

time_start_continuous = time_start_continuous(:,1:index_end); 
% time_start_continuous = (time_start_continuous - nanmean(time_start_continuous,2))./(nanstd(time_start_continuous')'*ones(1,index_end));

% Error rate, speed and temperature
xlimits = [30,160];
figure,
subplot(2,2,1)
myLogicalMask = ~isnan(error_continuous);
imagesc(error_continuous, 'AlphaData', myLogicalMask), colormap(parula), xlim(xlimits)
% colormap([0.9,0.9,0.9;0,0,0])
title('Errors')
subplot(2,2,2)
imagesc(error_continuous_smooth, 'AlphaData', myLogicalMask), title('Errors'), xlim(xlimits)
subplot(2,2,3)
imagesc(speed_continuous, 'AlphaData', myLogicalMask), title('Speed'), colorbar, clim([42,120]), xlim(xlimits)
subplot(2,2,4)
imagesc(temperature_continuous, 'AlphaData', myLogicalMask), title('Temperature'), colorbar, clim([15,40]), xlim(xlimits)

figure,
subplot(2,1,1)
imagesc(time_start_continuous, 'AlphaData', myLogicalMask), title('Trial start'), colorbar, %clim([15,40]), 
xlim(xlimits)
subplot(2,1,2)
plot_mean = nanmean(time_start_continuous); plot_mean(isnan(plot_mean)) = 0;
plot_std = nanstd(time_start_continuous); plot_std(isnan(plot_std)) = 0;
t_axis = 1:length(plot_mean);
patch([t_axis,flip(t_axis)], [plot_mean+plot_std,flip(plot_mean-plot_std)],'black','EdgeColor','none','FaceAlpha',.3), hold on
plot(plot_mean,'k'), title('Theta power'), axis tight, xlim(xlimits)

% Theta
figure,
subplot(2,2,1), colormap(parula)
imagesc(theta_freq_continuous, 'AlphaData', myLogicalMask), title('Theta freq'), colorbar, clim([5.5,9.5]), xlim(xlimits)
subplot(2,2,2)
imagesc(theta_power_continuous, 'AlphaData', myLogicalMask), title('Theta power'), colorbar, clim([-30,25]), xlim(xlimits)
subplot(2,2,3)
plot_mean = nanmean(theta_freq_continuous);  plot_mean(isnan(plot_mean)) = 0;
plot_std = nanstd(theta_freq_continuous); plot_std(isnan(plot_std)) = 0;
t_axis = 1:length(plot_mean);
patch([t_axis,flip(t_axis)], [plot_mean+plot_std,flip(plot_mean-plot_std)],'black','EdgeColor','none','FaceAlpha',.3), hold on
plot(plot_mean,'m'), title('Theta freq'), axis tight, xlim(xlimits)
subplot(2,2,4)
plot_mean = nanmean(theta_power_continuous); plot_mean(isnan(plot_mean)) = 0;
plot_std = nanstd(theta_power_continuous); plot_std(isnan(plot_std)) = 0;
t_axis = 1:length(plot_mean);
patch([t_axis,flip(t_axis)], [plot_mean+plot_std,flip(plot_mean-plot_std)],'black','EdgeColor','none','FaceAlpha',.3), hold on
plot(plot_mean,'k'), title('Theta power'), axis tight, xlim(xlimits)

% % Gamma low and mid
% figure,
% subplot(2,2,1)
% plot(lowGamma_power_continuous'), title('Gamma low'), colorbar, clim([4.5,9.5]), xlim(xlimits)
% subplot(2,2,2)
% plot(midGamma_power_continuous'), title('Gamma mid'), colorbar, clim([-30,25]), xlim(xlimits)
% subplot(2,2,3)
% plot_mean = nanmean(lowGamma_power_continuous);  plot_mean(isnan(plot_mean)) = 0;
% plot_std = nanstd(lowGamma_power_continuous); plot_std(isnan(plot_std)) = 0;
% t_axis = 1:length(plot_mean);
% patch([t_axis,flip(t_axis)], [plot_mean+plot_std,flip(plot_mean-plot_std)],'black','EdgeColor','none','FaceAlpha',.3), hold on
% plot(plot_mean,'m'), title('Gamma low power (35-80Hz)'), axis tight, xlim(xlimits)
% subplot(2,2,4)
% plot_mean = nanmean(midGamma_power_continuous); plot_mean(isnan(plot_mean)) = 0;
% plot_std = nanstd(midGamma_power_continuous); plot_std(isnan(plot_std)) = 0;
% t_axis = 1:length(plot_mean);
% patch([t_axis,flip(t_axis)], [plot_mean+plot_std,flip(plot_mean-plot_std)],'black','EdgeColor','none','FaceAlpha',.3), hold on
% plot(plot_mean,'k'), title('Gamma mid power (60-140Hz)'), axis tight, xlim(xlimits)

figure,
subplot(2,3,1)
plot_mean = nanmean(error_continuous_smooth); plot_mean(isnan(plot_mean)) = 0;
plot_std = nanstd(error_continuous_smooth); plot_std(isnan(plot_std)) = 0;
t_axis = 1:length(plot_mean);
patch([t_axis,flip(t_axis)], [plot_mean+plot_std,flip(plot_mean-plot_std)],'black','EdgeColor','none','FaceAlpha',.3), hold on
plot(plot_mean,'r'), title('Errors'), axis tight, xlim(xlimits)

subplot(2,3,2)
plot_mean = nanmean(temperature_continuous); plot_mean(isnan(plot_mean)) = 0;
plot_std = nanstd(temperature_continuous); plot_std(isnan(plot_std)) = 0;
t_axis = 1:length(plot_mean);
patch([t_axis,flip(t_axis)], [plot_mean+plot_std,flip(plot_mean-plot_std)],'black','EdgeColor','none','FaceAlpha',.3), hold on
plot(plot_mean,'b'), title('Temperature'), axis tight,, xlim(xlimits)

subplot(2,3,3)
plot_mean = nanmean(speed_continuous); plot_mean(isnan(plot_mean)) = 0;
plot_std = nanstd(speed_continuous); plot_std(isnan(plot_std)) = 0;
t_axis = 1:length(plot_mean);
patch([t_axis,flip(t_axis)], [plot_mean+plot_std,flip(plot_mean-plot_std)],'black','EdgeColor','none','FaceAlpha',.3), hold on
plot(plot_mean,'g'), title('Speed'), axis tight, xlim(xlimits)

subplot(2,1,2)
plot(nanzscore(nanmean(error_continuous_smooth)),'r'), hold on
plot(nanzscore(nanmean(temperature_continuous)),'b'),
plot(nanzscore(nanmean(speed_continuous)),'g'),
plot(nanzscore(nanmean(theta_freq_continuous)),'m'),
plot(nanzscore(nanmean(theta_power_continuous)),'k'),
% plot(nanzscore(nanmean(lowGamma_power_continuous)),'--b'),
% plot(nanzscore(nanmean(midGamma_power_continuous)),'--k'),
% legend({'Error','Temperature','Speed','Theta freq','Theta power','Low gamma (35-80Hz)','Mid gamma (60-140Hz)'}), axis tight
legend({'Error','Temperature','Speed','Theta freq','Theta power'}), axis tight
plot(nanzscore(nanmean(nanconv(error_continuous_smooth,gausswin(11)'/sum(gausswin(11)),'edge'))),'r'), hold on
xlim(xlimits)

% VsPlots for all session
figure
subplot(3,2,1)
plot(nanmean(temperature_continuous),nanmean(error_continuous_smooth),'.'), xlabel('Temp'), ylabel('Error rate')
subplot(3,2,2)
plot(nanmean(theta_freq_continuous),nanmean(error_continuous_smooth),'.'), xlabel('Theta freq'), ylabel('Error rate')
subplot(3,2,3)
plot(nanmean(temperature_continuous),nanmean(theta_freq_continuous),'.'), xlabel('Temp'), ylabel('Theta freq')
subplot(3,2,4)
plot(nanmean(temperature_continuous),nanmean(theta_power_continuous),'.'), xlabel('Temp'), ylabel('Theta power')
subplot(3,2,5)
plot(nanmean(speed_continuous),nanmean(error_continuous_smooth),'.'), xlabel('Speed'), ylabel('Error rate')
subplot(3,2,6)
plot(nanmean(theta_power_continuous),nanmean(error_continuous_smooth),'.'), xlabel('Theta power'), ylabel('Error rate')

figure
subplot(3,2,1)
plot((temperature_continuous),(speed_continuous),'.k'), xlabel('Temp'), ylabel('Speed (cm/s)')
subplot(3,2,2)
plot((theta_freq_continuous),(error_continuous_smooth),'.k'), xlabel('Theta frequency (Hz)'), ylabel('Error rate')
subplot(3,2,3)
plot((temperature_continuous),(theta_freq_continuous),'.k'), xlabel('Temperature (C)'), ylabel('Theta freq')
subplot(3,2,4)
plot((temperature_continuous),(theta_power_continuous),'.k'), xlabel('Temp'), ylabel('Theta power')
subplot(3,2,5)
plot((speed_continuous),(error_continuous_smooth),'.k'), xlabel('Speed'), ylabel('Error rate')
subplot(3,2,6)
plot((theta_power_continuous),(error_continuous_smooth),'.k'), xlabel('Theta power'), ylabel('Error rate')


% VsPlots grouped by animals
figure
R1 = []; R2 = []; R3 = []; R4 = []; R5 = []; R6 = [];
P1 = []; P2 = []; P3 = []; P4 = []; P5 = []; P6 = [];
Q1 = []; Q2 = []; Q3 = []; Q4 = []; Q5 = []; Q6 = [];
for k = 1:length(unique(animalid))
    subplot(3,2,1), hold on
    x = nanmean(temperature_continuous(animalid==k,:));
    y1 = nanmean(error_continuous_smooth(animalid==k,:));
    [R1(k),P1(k),Q1(k),RL1,RU1] = plotWithFit(x,y1,colors{k});
    
    xlabel('Temp'), ylabel('Error rate')
    subplot(3,2,2), hold on
    x = nanmean(theta_freq_continuous(animalid==k,:));
    y1 = nanmean(error_continuous_smooth(animalid==k,:));
    [R2(k),P2(k),Q2(k),RL2,RU2] = plotWithFit(x,y1,colors{k});
    xlabel('Theta freq'), ylabel('Error rate')
    subplot(3,2,3), hold on
    x = nanmean(temperature_continuous(animalid==k,:));
    y1 = nanmean(theta_freq_continuous(animalid==k,:));
    [R3(k),P3(k),Q3(k),RL3,RU3] = plotWithFit(x,y1,colors{k});
    xlabel('Temp'), ylabel('Theta freq')
    subplot(3,2,4), hold on
    x = nanmean(temperature_continuous(animalid==k,:));
    y1 = nanmean(theta_power_continuous(animalid==k,:));
    [R4(k),P4(k),Q4(k),RL4,RU4] = plotWithFit(x,y1,colors{k});
    xlabel('Temp'), ylabel('Theta power')
    subplot(3,2,5), hold on
    x = nanmean(speed_continuous(animalid==k,:));
    y1 = nanmean(error_continuous_smooth(animalid==k,:));
    [R5(k),P5(k),Q5(k),RL5,RU5] = plotWithFit(x,y1,colors{k});
    xlabel('Speed'), ylabel('Error rate')
    subplot(3,2,6), hold on
    x = nanmean(theta_power_continuous(animalid==k,:));
    y1 = nanmean(error_continuous_smooth(animalid==k,:));
    [R6(k),P6(k),Q6(k),RL6,RU6] = plotWithFit(x,y1,colors{k});
    xlabel('Theta power'), ylabel('Error rate')
end

% VsPlots grouped by animals
figure
plot1 = []; plot2 = []; plot3 = []; plot4 = []; plot5 = []; plot6 = [];
for k = 1:length(unique(animalid))
    subplot(3,2,1), hold on
    x = (temperature_continuous(animalid==k,:));
    y1 = (error_continuous_smooth(animalid==k,:));
    x_bins = [20:2:40];
    [x_bins1,plot1{k}] = plotErrorBarsFromScatter(x,y1,x_bins);
    
    xlabel('Temp'), ylabel('Error rate')
    subplot(3,2,2), hold on
    x = (theta_freq_continuous(animalid==k,:));
    y1 = (error_continuous_smooth(animalid==k,:));
    x_bins = [6:.2:10];
    [x_bins2,plot2{k}] = plotErrorBarsFromScatter(x,y1,x_bins);
    xlabel('Theta freq'), ylabel('Error rate')
    subplot(3,2,3), hold on
    x = temperature_continuous(animalid==k,:);
    y1 = (theta_freq_continuous(animalid==k,:));
    x_bins = [20:2:40];
    [x_bins3,plot3{k}] = plotErrorBarsFromScatter(x,y1,x_bins);
    xlabel('Temp'), ylabel('Theta freq')
    subplot(3,2,4), hold on
    x = (temperature_continuous(animalid==k,:));
    y1 = (theta_power_continuous(animalid==k,:));
    x_bins = [20:2:40];
    [x_bins4,plot4{k}] = plotErrorBarsFromScatter(x,y1,x_bins);
    xlabel('Temp'), ylabel('Theta power')
    subplot(3,2,5), hold on
    x = (speed_continuous(animalid==k,:));
    y1 = (error_continuous_smooth(animalid==k,:));
    x_bins = [30:5:110];
    [x_bins5,plot5{k}] = plotErrorBarsFromScatter(x,y1,x_bins);
    xlabel('Speed'), ylabel('Error rate')
    subplot(3,2,6), hold on
    x = (theta_power_continuous(animalid==k,:));
    y1 = (error_continuous_smooth(animalid==k,:));
    x_bins = [-15:15];
    [x_bins6,plot6{k}] = plotErrorBarsFromScatter(x,y1,x_bins);
    xlabel('Theta power'), ylabel('Error rate')
end
figure
subplot(3,2,1), hold on
errorbar(x_bins1,nanmean(vertcat(plot1{:})),nansem(vertcat(plot1{:})),'Color','k'), ylim([0,0.4])
xlabel('Temp'), ylabel('Error rate')
subplot(3,2,2), hold on
errorbar(x_bins2,nanmean(vertcat(plot2{:})),nansem(vertcat(plot2{:})),'Color','k'), ylim([0,0.4])
xlabel('Theta freq'), ylabel('Error rate')
subplot(3,2,3), hold on
errorbar(x_bins3,nanmean(vertcat(plot3{:})),nansem(vertcat(plot3{:})),'Color','k')
xlabel('Temp'), ylabel('Theta freq')
subplot(3,2,4), hold on
errorbar(x_bins4,nanmean(vertcat(plot4{:})),nansem(vertcat(plot4{:})),'Color','k')
xlabel('Temp'), ylabel('Theta power')
subplot(3,2,5), hold on
errorbar(x_bins5,nanmean(vertcat(plot5{:})),nansem(vertcat(plot5{:})),'Color','k'), ylim([0,0.4])
xlabel('Speed'), ylabel('Error rate')
subplot(3,2,6), hold on
errorbar(x_bins6,nanmean(vertcat(plot6{:})),nansem(vertcat(plot6{:})),'Color','k'), ylim([0,0.4])
xlabel('Theta power'), ylabel('Error rate')

% VsPlots for all session
figure
subplot(3,2,1), x_bins = [20:2:40];
plotErrorBarsFromScatter((temperature_continuous),(error_continuous_smooth),x_bins), xlabel('Temp'), ylabel('Error rate')
subplot(3,2,2), x_bins = [6:.2:10];
plotErrorBarsFromScatter((theta_freq_continuous),(error_continuous_smooth),x_bins), xlabel('Theta freq'), ylabel('Error rate')
subplot(3,2,3), x_bins = [20:2:40];
plotErrorBarsFromScatter((temperature_continuous),(theta_freq_continuous),x_bins), xlabel('Temp'), ylabel('Theta freq')
subplot(3,2,4), x_bins = [20:2:40];
plotErrorBarsFromScatter((temperature_continuous),(theta_power_continuous),x_bins), xlabel('Temp'), ylabel('Theta power')
subplot(3,2,5), x_bins = [30:5:110];
plotErrorBarsFromScatter((speed_continuous),(error_continuous_smooth),x_bins), xlabel('Speed'), ylabel('Error rate')
subplot(3,2,6), x_bins = [-15:15];
plotErrorBarsFromScatter((theta_power_continuous),(error_continuous_smooth),x_bins), xlabel('Theta power'), ylabel('Error rate')

figure,
for i =1:5
    subplot(2,2,1)
    plot([1],R3(i),'o'), hold on, ylim([-0.5,1])
    subplot(2,2,2)
    plot([1]*2,R4(i),'o'), xlim([0,3]), title('R values'), hold on, ylim([-0.5,1])
    subplot(2,2,3)
    plot([1],log10(P3(i)),'o'), hold on, ylim([-100,0])
    subplot(2,2,4)
    plot([1]*2,log10(P4(i)),'o'), xlim([0,3]), title('P values'), hold on, ylim([-100,0])
    legend(animal_names)
end

x = nanmean(temperature_continuous);
y1 = nanmean(theta_freq_continuous);
[R33,P33,Q33,RL33,RU33] = plotWithFit(x,y1,colors{k});
x = nanmean(temperature_continuous);
y1 = nanmean(theta_power_continuous);
[R44,P44,Q44,RL44,RU44] = plotWithFit(x,y1,colors{k});

figure,
subplot(1,3,1)
plot([1,2,3,4]',[R1;R2;R5;R6],'.','markersize',15), ylabel('Correlation'), hold on, title('Correlations with error rate')
plot(1,[R1(find(P1<0.001))],'ok','markersize',8)
plot(2,[R2(find(P2<0.001))],'ok','markersize',8)
% plot(3,[R5(find(P5<0.001))],'ok','markersize',8)
plot(4,[R6(find(P6<0.001))],'ok','markersize',8)

plot(1,[R1(find(P1<0.05))],'sk','markersize',8)
plot(2,[R2(find(P2<0.05))],'sk','markersize',8)
plot(3,[R5(find(P5<0.05))],'sk','markersize',8)
plot(4,[R6(find(P6<0.05))],'sk','markersize',8)

xticks([1:4]), xticklabels({'Temperature','Theta frequency','Running speed','Theta power'}), xtickangle(45)
subplot(1,3,2)
plot([1,2,3,4]',[Q1;Q2;Q5;Q6],'.','markersize',8), title('Slope')
xticks([1:4]), xticklabels({'Temp vs Error','Theta freq vs Error','Speed vs Error rate','Theta power vs Error rate'}), xtickangle(45)
subplot(1,3,3)
plot([1,2,3,4]',log10([P1;P2;P5;P6]),'.','markersize',8), title('P-values')
xticks([1:4]), xticklabels({'Temp vs Error','Theta freq vs Error','Speed vs Error rate','Theta power vs Error rate'}), xtickangle(45)

[rho1,pva1] = corr([nanmean(error_continuous_smooth(:,10:160));nanmean(temperature_continuous(:,10:160));nanmean(speed_continuous(:,10:160));nanmean(theta_freq_continuous(:,10:160));nanmean(theta_power_continuous(:,10:160))]');
[rho2,pva2] = partialcorr([nanmean(error_continuous_smooth(:,10:160));nanmean(temperature_continuous(:,10:160));nanmean(speed_continuous(:,10:160));nanmean(theta_freq_continuous(:,10:160));nanmean(theta_power_continuous(:,10:160))]');

figure,
subplot(2,2,1)
imagesc(abs(rho1)), colormap(parula), clim([0,1])
datatypes = {'error rate','temperature','speed','theta freq','theta power'};
xticks([1:5]), xticklabels(datatypes), yticks([1:5]), yticklabels(datatypes), title('Correlation'), xtickangle(45)
subplot(2,2,3)
imagesc(log10(pva1)), colormap(flip(gray)), clim(log10([0.0001,1]))
datatypes = {'error rate','temperature','speed','theta freq','theta power'};
xticks([1:5]), xticklabels(datatypes), yticks([1:5]), yticklabels(datatypes), title('P value'), xtickangle(45)
subplot(2,2,2)
imagesc(abs(rho2)), colormap(parula), clim([0,1])
xticks([1:5]), xticklabels(datatypes), yticks([1:5]), yticklabels(datatypes), title('Partial correlation'), xtickangle(45),colorbar
subplot(2,2,4)
imagesc(log10(pva2)), colormap(flip(gray)), clim(log10([0.0001,1]))
xticks([1:5]), xticklabels(datatypes), yticks([1:5]), yticklabels(datatypes), title('P value'), xtickangle(45),colorbar

%% % Measures vs time
figure,
subplot(2,2,1)
plot(time_start_continuous', temperature_continuous'), axis tight, xlim([-200,1000]), ylim([15,39]), xlabel('Time (s)'), ylabel('Temperature (C)')
subplot(2,2,2)
plot(time_start_continuous', theta_freq_continuous'-nanmean(theta_freq_continuous(:,1:64)')), axis tight, xlim([-200,1000]), xlabel('Time (s)'), ylabel('Theta freq (Hz)'), %ylim([6,9.5]),
subplot(2,2,3)
plot(time_start_continuous', theta_power_continuous'), axis tight, xlim([-200,1000]), xlabel('Time (s)'), ylabel('Theta power'), %ylim([15,39])
subplot(2,2,4)
plot(nanmean(time_start_continuous)', nanmean(error_continuous_smooth)'), axis tight, xlim([-200,1000]), xlabel('Time (s)'), ylabel('Error rate'), %ylim([15,39])

figure,
subplot(2,2,2)
plot(temperature_continuous', theta_freq_continuous'-nanmean(theta_freq_continuous(:,1:64)'),'o-'), axis tight, xlabel('Temperature (C)'), ylabel('Theta freq (Hz)'), %ylim([6,9.5]),
subplot(2,2,3)
plot(temperature_continuous', theta_power_continuous'), axis tight, xlabel('Temperature (C)'), ylabel('Theta power'), %ylim([15,39])
subplot(2,2,4)
plot(nanmean(temperature_continuous)', nanmean(error_continuous_smooth)'), axis tight, xlabel('Temperature (C)'), ylabel('Error rate'), %ylim([15,39])

% plot(nanzscore(nanmean(error_continuous_smooth)),'r'), hold on
% plot(nanzscore(nanmean(temperature_continuous)),'b'),
% plot(nanzscore(nanmean(speed_continuous)),'g'),
% plot(nanzscore(nanmean(theta_freq_continuous)),'m'),
% plot(nanzscore(nanmean(theta_power_continuous)),'k'),
% plot(nanzscore(nanmean(lowGamma_power_continuous)),'--b'),
% plot(nanzscore(nanmean(midGamma_power_continuous)),'--k'),

%% % % % % % % % % % % % % % % % % % % % % % % % % % % %
% Phase precession
clear all, close all
MedialSeptum_Recordings
% MS10: 61 (Peter_MS10_170317_153237_concat) OK
%       62 (Peter_MS10_170314_163038) % gamma id: 62 OK
%       63 (Peter_MS10_170315_123936) 
%       64 (Peter_MS10_170307_154746_concat) OK
% MS12: 78 (Peter_MS12_170714_122034_concat) OK
%       79 (Peter_MS12_170715_111545_concat) OK
%       80 (Peter_MS12_170716_172307_concat) OK
%       81 (Peter_MS12_170717_111614_concat) OK
%       83 (Peter_MS12_170719_095305_concat) OK
% MS13: 92 (Peter_MS13_171129_105507_concat) OK
%       93 (Peter_MS13_171130_121758_concat) OK
%       88 (Peter_MS13_171110_163224_concat) No good cooling behavior
%       91 (Peter_MS13_171128_113924_concat) OK
%       94 (Peter_MS13_171201_130527_concat) OK
% MS21: 126 (Peter_MS21_180629_110332_concat) OK
%       140 (Peter_MS21_180627_143449_concat) OK
%       143 (Peter_MS21_180719_155941_concat, control)
%       149 (Peter_MS21_180625_153927_concat) OK
%       153 (Peter_MS21_180712_103200_concat) OK
%       151 (Peter_MS21_180628_155921_concat) OK
%       154 (Peter_MS21_180719_122733_concat, control, duplicated)
%       159 (Peter_MS21_180807_122213_concat, control)
% MS22: 139 (Peter_MS22_180628_120341_concat) OK
%       127 (Peter_MS22_180629_110319_concat) OK
%       144 (Peter_MS22_180719_122813_concat, control)
%       168 (Peter_MS22_180720_110055_concat) OK 
%       166 (Peter_MS22_180711_112912_concat) OK 

% datasets = [78,79,80,81];
% datasets = [126, 140, 139, 127, 78, 79, 80, 81, 92, 93]; % 62
% animalID = [3,3,4,4,1,1,1,1,2,2];
datasets = [61,64,78,79,80,81,83,91,92,93,126,140,149,153,151,139,127,166]; % 62 63, 94,168
animalID = [1,1,  2,2,2,2,2,  3,3,3  4,4,4,4,4,  5,5,5];
clear out1
kk = 1;
for k = 1:length(datasets)
    recording = recordings(datasets(k));
    recording.sr_lfp = recording.sr/16;
    disp(['Processing ', num2str(k),'/',num2str(length(datasets)),' recordingID = ' num2str(datasets(k)),', basename = ' recording.name])
    sr = recording.sr;
    cd(fullfile(datapath, recording.animal_id, recording.name))
    
    disp('Loading animal structure')
    load([recording.name,'.animal.behavior.mat'])

    % Trials
    disp('Loading trials')
    load([recording.name,'.trials.behavior.mat'])
    
    disp('Loading spikes')
    spikes = loadSpikes('clusteringpath',recording.SpikeSorting.path,'clusteringformat',recording.SpikeSorting.method,'basename',recording.name);

    if isfield(recording.SpikeSorting,'polar_theta_placecells')
        for i = 1:spikes.numcells
            if spikes.cluID(i) <= length(recording.SpikeSorting.polar_theta_placecells)
                if ~isempty(recording.SpikeSorting.polar_theta_placecells{spikes.cluID(i)})
                    spikes.PhasePrecession{i}.placefields_polar_theta = recording.SpikeSorting.polar_theta_placecells{spikes.cluID(i)};
                end
            end
        end
    end
    if isfield(recording.SpikeSorting,'center_arm_placecells')
        for i = 1:spikes.numcells
            if spikes.cluID(i) <= length(recording.SpikeSorting.center_arm_placecells)
                if ~isempty(recording.SpikeSorting.center_arm_placecells{spikes.cluID(i)})
                    spikes.PhasePrecession{i}.placefields_center_arm = recording.SpikeSorting.center_arm_placecells{spikes.cluID(i)};
                end
            end
        end
    end
    disp('Checking place field stability')
    out1(kk) = plotPlaceFieldStability(spikes,trials);
    kk = kk + 1;
end

%%
out_fieldnames = fieldnames(out1);
out = [];
for j = 1:length(out1)
    if j == 1
        for i = 1:length(out_fieldnames)-3
            out.(out_fieldnames{i}) = out1(j).(out_fieldnames{i});
        end
    else
        for i = 1:length(out_fieldnames)-3
            out.(out_fieldnames{i}) = [out.(out_fieldnames{i}),out1(j).(out_fieldnames{i})];
        end
        
    end
end

hist_bins = [-1:0.1:1];
figure,
subplot(3,4,1)
histogram(out.r_oscillation_freq,hist_bins), title('oscillation freq'), xlim([-1, 1]), ylabel('Temperature'), hold on
histogram(out.r_oscillation_freq(find(out.p_oscillation_freq > 0.05)),hist_bins)
plot(mean(out.r_oscillation_freq(find(out.p_oscillation_freq < 0.05))),0,'v','linewidth',2)
subplot(3,4,2)
histogram(out.r_slope1,hist_bins), title('slope'), xlim([-1, 1]), hold on
histogram(out.r_slope1(find(out.p_slope1 > 0.05)),hist_bins)
plot(mean(out.r_slope1(find(out.p_slope1 < 0.05))),0,'v','linewidth',2)
subplot(3,4,3)
histogram(out.r_theta_cycles,hist_bins), title('theta cycles'), xlim([-1, 1]), hold on
histogram(out.r_theta_cycles(find(out.p_theta_cycles > 0.05)),hist_bins)
plot(mean(out.r_theta_cycles(find(out.p_theta_cycles < 0.05))),0,'v','linewidth',2)
subplot(3,4,4)
histogram(out.r_spikecount,hist_bins), title('spikecount'), xlim([-1, 1]), hold on
histogram(out.r_spikecount(find(out.p_spikecount > 0.05)),hist_bins)
plot(mean(out.r_spikecount(find(out.p_spikecount < 0.05))),0,'v','linewidth',2)

% Theta
subplot(3,4,5)
histogram(out.r1_oscillation_freq,hist_bins), title('oscillation freq'), xlim([-1, 1]), ylabel('Theta'), hold on
histogram(out.r1_oscillation_freq(find(out.p1_oscillation_freq > 0.05)),hist_bins)
plot(mean(out.r1_oscillation_freq(find(out.p1_oscillation_freq < 0.05))),0,'v','linewidth',2)

subplot(3,4,6)
histogram(out.r1_slope1,hist_bins), title('slope'), xlim([-1, 1]), hold on
histogram(out.r1_slope1(find(out.p1_slope1 > 0.05)),hist_bins)
plot(mean(out.r1_slope1(find(out.p1_slope1 < 0.05))),0,'v','linewidth',2)

subplot(3,4,7)
histogram(out.r1_theta_cycles,hist_bins), title('theta cycles'), xlim([-1, 1]), hold on
histogram(out.r1_theta_cycles(find(out.p1_theta_cycles > 0.05)),hist_bins)
plot(mean(out.r1_theta_cycles(find(out.p1_theta_cycles < 0.05))),0,'v','linewidth',2)

subplot(3,4,8)
histogram(out.r1_spikecount,hist_bins), title('spikecount'), xlim([-1, 1]), hold on
histogram(out.r1_spikecount(find(out.p1_spikecount > 0.05)),hist_bins)
plot(mean(out.r1_spikecount(find(out.p1_spikecount < 0.05))),0,'v','linewidth',2)

% Speed
subplot(3,4,9)
histogram(out.r2_oscillation_freq,hist_bins), title('oscillation freq'), xlim([-1, 1]), ylabel('Speed'), hold on
histogram(out.r2_oscillation_freq(find(out.p2_oscillation_freq > 0.05)),hist_bins)
plot(mean(out.r2_oscillation_freq(find(out.p2_oscillation_freq < 0.05))),0,'v','linewidth',2)

subplot(3,4,10)
histogram(out.r2_slope1,hist_bins), title('slope'), xlim([-1, 1]), hold on
histogram(out.r2_slope1(find(out.p2_slope1 > 0.05)),hist_bins)
plot(mean(out.r2_slope1(find(out.p2_slope1 < 0.05))),0,'v','linewidth',2)

subplot(3,4,11)
histogram(out.r2_theta_cycles,hist_bins), title('theta cycles'), xlim([-1, 1]), hold on
histogram(out.r2_theta_cycles(find(out.p2_theta_cycles > 0.05)),hist_bins)
plot(mean(out.r2_theta_cycles(find(out.p2_theta_cycles < 0.05))),0,'v','linewidth',2)

subplot(3,4,12)
histogram(out.r2_spikecount,hist_bins), title('spikecount'), xlim([-1, 1]), hold on
histogram(out.r2_spikecount(find(out.p2_spikecount > 0.05)),hist_bins)
plot(mean(out.r2_spikecount(find(out.p2_spikecount < 0.05))),0,'v','linewidth',2)

figure, subplot(2,2,1), plot(out.r_oscillation_freq,out.r2_oscillation_freq,'.','linewidth',2), xlim([-1,1]), ylim([-1,1]), hold on, plot([-1,1],[-1,1],'-')
xlabel('Temperature'), ylabel('Running speed'),title('place cells oscillation correlations'), grid on

subplot(2,2,2)
histogram(out.r3_slope1,hist_bins), title('Theta ratio vs precession slope'), xlim([-1, 1]), hold on
histogram(out.r3_slope1(find(out.p3_slope1 > 0.05)),hist_bins)
plot(mean(out.r3_slope1(find(out.p3_slope1 < 0.05))),0,'v','linewidth',2)

subplot(2,2,3)
histogram(out.r3_speed,hist_bins), title('Theta ratio vs speed'), xlim([-1, 1]), hold on
histogram(out.r3_speed(find(out.p3_speed > 0.05)),hist_bins)
plot(mean(out.r3_speed(find(out.p3_speed < 0.05))),0,'v','linewidth',2)

subplot(2,2,4), plot(out.r3_slope1,out.r3_speed,'.','linewidth',2), xlim([-1,1]), ylim([-1,1]), hold on, plot([-1,1],[-1,1],'-')
xlabel('Precession slope'), ylabel('Running speed'),title('Oscillation freq/LFP theta correlations'), grid on

%% % Figures with trial-wise plots

trials_continuous_slope1  = nan(220,200);
trials_continuous_temperature  = nan(220,200);
trials_continuous_spikecount  = nan(220,200);
trials_continuous_speed  = nan(220,200);
trials_continuous_theta  = nan(220,200);
trials_continuous_theta_cycles  = nan(220,200);
trials_continuous_oscillation_freq  = nan(220,200);
kk = 1;

for j = 1:size(out1,2)
%     trial_offset = 65-find(trials_batch.cooling{i}==2,1);
    for i = 1:size(out1(j).tri,2)
        tri = out1(j).tri{i};
        tri.slope1 = abs(tri.slope1);
        tri.slope1(tri.slope1>0.06) = nan;
        trial_offset = 65-find(tri.temperature<36.5,1);
        if ~isempty(trial_offset)
        trials_continuous_slope1(kk,[1:length(tri.slope1)]+trial_offset) = tri.slope1;
        trials_continuous_temperature(kk,[1:length(tri.slope1)]+trial_offset) = tri.temperature;
        trials_continuous_spikecount(kk,[1:length(tri.slope1)]+trial_offset) = tri.spikecount;
        trials_continuous_speed(kk,[1:length(tri.slope1)]+trial_offset) = tri.speed;
        trials_continuous_theta(kk,[1:length(tri.slope1)]+trial_offset) = tri.theta;
        trials_continuous_theta_cycles(kk,[1:length(tri.slope1)]+trial_offset) = tri.theta_cycles;
        trials_continuous_oscillation_freq(kk,[1:length(tri.slope1)]+trial_offset) = tri.oscillation_freq;
        kk = kk + 1;
        end
    end
end

% Images
figure, 
subplot(3,3,1), imagesc(trials_continuous_slope1), title('Slope'), xlabel('Trials')
subplot(3,3,2), imagesc(trials_continuous_temperature), title('Temperature'), xlabel('Trials')
subplot(3,3,3), imagesc(trials_continuous_spikecount), title('Spike count'), xlabel('Trials')
subplot(3,3,4), imagesc(trials_continuous_speed), title('Speed'), xlabel('Trials')
subplot(3,3,5), imagesc(trials_continuous_theta), title('Theta'), xlabel('Trials')
subplot(3,3,6), imagesc(trials_continuous_theta_cycles), title('theta cycles'), xlabel('Trials')
subplot(3,3,7), imagesc(trials_continuous_oscillation_freq), title('oscillation_freq'), xlabel('Trials')

% Aligned average
figure
subplot(3,3,1)
plot_mean = nanmean(trials_continuous_slope1);  plot_mean(isnan(plot_mean)) = 0;
plot_std = nanstd(trials_continuous_slope1); plot_std(isnan(plot_std)) = 0;
t_axis = 1:length(plot_mean);
patch([t_axis,flip(t_axis)], [plot_mean+plot_std,flip(plot_mean-plot_std)],'black','EdgeColor','none','FaceAlpha',.3), hold on
plot(plot_mean,'m'), title('slope'), axis tight
subplot(3,3,2)
plot_mean = nanmean(trials_continuous_temperature);  plot_mean(isnan(plot_mean)) = 0;
plot_std = nanstd(trials_continuous_temperature); plot_std(isnan(plot_std)) = 0;
t_axis = 1:length(plot_mean);
patch([t_axis,flip(t_axis)], [plot_mean+plot_std,flip(plot_mean-plot_std)],'black','EdgeColor','none','FaceAlpha',.3), hold on
plot(plot_mean,'m'), title('temperature'), axis tight
subplot(3,3,3)
plot_mean = nanmean(trials_continuous_spikecount);  plot_mean(isnan(plot_mean)) = 0;
plot_std = nanstd(trials_continuous_spikecount); plot_std(isnan(plot_std)) = 0;
t_axis = 1:length(plot_mean);
patch([t_axis,flip(t_axis)], [plot_mean+plot_std,flip(plot_mean-plot_std)],'black','EdgeColor','none','FaceAlpha',.3), hold on
plot(plot_mean,'m'), title('spikecount'), axis tight
subplot(3,3,4)
plot_mean = nanmean(trials_continuous_speed);  plot_mean(isnan(plot_mean)) = 0;
plot_std = nanstd(trials_continuous_speed); plot_std(isnan(plot_std)) = 0;
t_axis = 1:length(plot_mean);
patch([t_axis,flip(t_axis)], [plot_mean+plot_std,flip(plot_mean-plot_std)],'black','EdgeColor','none','FaceAlpha',.3), hold on
plot(plot_mean,'m'), title('speed'), axis tight
subplot(3,3,5)
plot_mean = nanmean(trials_continuous_theta);  plot_mean(isnan(plot_mean)) = 0;
plot_std = nanstd(trials_continuous_theta); plot_std(isnan(plot_std)) = 0;
t_axis = 1:length(plot_mean);
patch([t_axis,flip(t_axis)], [plot_mean+plot_std,flip(plot_mean-plot_std)],'black','EdgeColor','none','FaceAlpha',.3), hold on
plot(plot_mean,'m'), title('theta'), axis tight
subplot(3,3,6)
plot_mean = nanmean(trials_continuous_theta_cycles);  plot_mean(isnan(plot_mean)) = 0;
plot_std = nanstd(trials_continuous_theta_cycles); plot_std(isnan(plot_std)) = 0;
t_axis = 1:length(plot_mean);
patch([t_axis,flip(t_axis)], [plot_mean+plot_std,flip(plot_mean-plot_std)],'black','EdgeColor','none','FaceAlpha',.3), hold on
plot(plot_mean,'m'), title('theta_cycles'), axis tight
subplot(3,3,7)
plot_mean = nanmean(trials_continuous_oscillation_freq);  plot_mean(isnan(plot_mean)) = 0;
plot_std = nanstd(trials_continuous_oscillation_freq); plot_std(isnan(plot_std)) = 0;
t_axis = 1:length(plot_mean);
patch([t_axis,flip(t_axis)], [plot_mean+plot_std,flip(plot_mean-plot_std)],'black','EdgeColor','none','FaceAlpha',.3), hold on
plot(plot_mean,'m'), title('oscillation_freq'), axis tight

colors = {'-r','-b','-g','-m','-c'};

kk = 1;
fig1 = figure('pos',[50 50 900 800]),
for j = 1:size(out1,2)
    
    color1 = colors{animalID(j)};
    color1 = '.';
    for i = 1:size(out1(j).tri,2)
        tri = out1(j).tri{i};
        tri.slope1 = abs(tri.slope1);
        tri.slope1(tri.slope1>0.06) = nan;
        subplot(3,3,1), hold on, plot(tri.temperature,abs(tri.slope1),color1,'markersize',8), xlabel('Temperature'), ylabel('Slope'), axis tight
        subplot(3,3,2), hold on, plot(tri.temperature,tri.theta,color1,'markersize',8), xlabel('Temperature'), ylabel('Theta freq'), axis tight
        subplot(3,3,3), hold on, plot(tri.temperature,tri.oscillation_freq,color1,'markersize',8), xlabel('Temperature'), ylabel('Oscillation freq'), axis tight
        subplot(3,3,4), hold on, plot(tri.temperature,tri.theta_cycles,color1,'markersize',8), xlabel('Temperature'), ylabel('Theta cycles'), axis tight
        subplot(3,3,5), hold on, plot(tri.temperature,tri.speed,color1,'markersize',8), xlabel('Temperature'), ylabel('Speed'), axis tight,
        subplot(3,3,6), hold on, plot(tri.temperature,tri.spikecount,color1,'markersize',8), xlabel('Temperature'), ylabel('Spike Count'), axis tight
        subplot(3,3,7), hold on, plot(tri.speed,tri.oscillation_freq-tri.theta,color1,'markersize',8), ylabel('theta offset'), xlabel('Speed'), axis tight
        subplot(3,3,8), hold on, plot(tri.theta,abs(tri.slope1),color1), xlabel('Theta freq'), ylabel('Precession slope'), axis tight
        subplot(3,3,9), hold on, plot(tri.theta,tri.oscillation_freq,color1), xlabel('Theta freq'), ylabel('Oscillation freq'), axis tight
        kk = kk + 1;
    end
%     saveas(fig1,['K:\Dropbox\Buzsakilab Postdoc\MatlabFigures\TrialwiseUnits\','VsPlots_animal', num2str(animalID(j)), '_session',num2str(j),'.png'])
end

subplot(3,3,1),
x = (trials_continuous_temperature);
y1 = (trials_continuous_slope1);
[R1,P1,Q1] = plotWithFit2(x(:)',y1(:)','k');
title(['R,P,Q:' num2str(R1),', ',num2str(P1),', ',num2str(Q1)])
subplot(3,3,2),
x = (trials_continuous_temperature);
y1 = (trials_continuous_theta);
[R1,P1,Q1] = plotWithFit2(x(:)',y1(:)','k');
title(['R,P,Q:' num2str(R1),', ',num2str(P1),', ',num2str(Q1)])
subplot(3,3,3),
x = (trials_continuous_temperature);
y1 = (trials_continuous_oscillation_freq);
[R1,P1,Q1] = plotWithFit2(x(:)',y1(:)','k');
title(['R,P,Q:' num2str(R1),', ',num2str(P1),', ',num2str(Q1)])
subplot(3,3,4),
x = (trials_continuous_temperature);
y1 = (trials_continuous_theta_cycles);
[R1,P1,Q1] = plotWithFit2(x(:)',y1(:)','k');
title(['R,P,Q:' num2str(R1),', ',num2str(P1),', ',num2str(Q1)])
subplot(3,3,5),
x = (trials_continuous_temperature);
y1 = (trials_continuous_speed);
[R1,P1,Q1] = plotWithFit2(x(:)',y1(:)','k');
title(['R,P,Q:' num2str(R1),', ',num2str(P1),', ',num2str(Q1)])
subplot(3,3,6),
x = (trials_continuous_temperature);
y1 = (trials_continuous_spikecount);
[R1,P1,Q1] = plotWithFit2(x(:)',y1(:)','k');
title(['R,P,Q:' num2str(R1),', ',num2str(P1),', ',num2str(Q1)])

subplot(3,3,7),
x = (trials_continuous_speed);
y1 = (trials_continuous_oscillation_freq-trials_continuous_theta);
[R1,P1,Q1] = plotWithFit2(x(:)',y1(:)','k');
title(['R,P,Q:' num2str(R1),', ',num2str(P1),', ',num2str(Q1)])
subplot(3,3,8),
x = (trials_continuous_theta);
y1 = (trials_continuous_slope1);
[R1,P1,Q1] = plotWithFit2(x(:)',y1(:)','k');
title(['R,P,Q:' num2str(R1),', ',num2str(P1),', ',num2str(Q1)])
subplot(3,3,9),
x = (trials_continuous_theta);
y1 = (trials_continuous_oscillation_freq);
[R1,P1,Q1] = plotWithFit2(x(:)',y1(:)','k');
title(['R,P,Q:' num2str(R1),', ',num2str(P1),', ',num2str(Q1)])

% Theta freq and oscillation freq 
figure
for j = 1:size(out1,2)
    color1 = colors{animalID(j)};
    color1 = '.';
    for i = 1:size(out1(j).tri,2)
        tri = out1(j).tri{i};
        tri.slope1 = abs(tri.slope1);
        tri.slope1(tri.slope1>0.06) = nan;
        subplot(2,2,1), 
        hold on, plot(tri.temperature,tri.theta,'.','color',[0.5 0.5 0.5],'markersize',8), xlabel('Temperature'), ylabel('Theta freq'), axis tight
        hold on, plot(tri.temperature,tri.oscillation_freq,'.','color',[0.9 0.2 0.2],'markersize',8), xlabel('Temperature'), ylabel('Oscillation freq'), axis tight
        subplot(2,2,2), 
        hold on, plot(tri.speed,tri.theta,'.','color',[0.5 0.5 0.5],'markersize',8), xlabel('Animal speed'), ylabel('Theta freq'), axis tight
        hold on, plot(tri.speed,tri.oscillation_freq,'.','color',[0.9 0.2 0.2],'markersize',8), xlabel('Animal speed'), ylabel('Oscillation freq'), axis tight
        kk = kk + 1;
    end
%     saveas(fig1,['C:\Users\peter\Dropbox\Buzsakilab Postdoc\Medial Septum Cooling Project\TrialwiseUnits\','VsPlots_animal', num2str(animalID(j)), '_session',num2str(j),'.png'])
end

subplot(2,2,1), 
x = (trials_continuous_temperature);
y1 = (trials_continuous_theta);
[R1,P1,Q1] = plotWithFit2(x(:)',y1(:)','k');
title(['R,P,Q:' num2str(R1),', ',num2str(P1),', ',num2str(Q1)])
Xedges = [18:2:38];
x2 = x(:);
y2 = y1(:);
plot_mean = [];
plot_std = [];
for i = 1:length(Xedges)-1
    idx = find(x2 > Xedges(i) & x2 < Xedges(i+1));
    plot_mean(i) = nanmean(y2(idx));
    plot_std(i) = nanstd(y2(idx));
end
errorbar(Xedges(1:end-1)+1,plot_mean,plot_std)
subplot(2,2,3), 
x = (trials_continuous_temperature);
y1 = (trials_continuous_oscillation_freq);
[R1,P1,Q1] = plotWithFit2(x(:)',y1(:)','b');
title(['R,P,Q:' num2str(R1),', ',num2str(P1),', ',num2str(Q1)])
x2 = x(:);
y2 = y1(:);
plot_mean = [];
plot_std = [];
for i = 1:length(Xedges)-1
    idx = find(x2 > Xedges(i) & x2 < Xedges(i+1));
    plot_mean(i) = nanmean(y2(idx));
    plot_std(i) = nanstd(y2(idx));
end
errorbar(Xedges(1:end-1)+1,plot_mean,plot_std)

subplot(2,2,2), 
x = (trials_continuous_speed);
y1 = (trials_continuous_theta);
[R1,P1,Q1] = plotWithFit2(x(:)',y1(:)','k');
title(['R,P,Q:' num2str(R1),', ',num2str(P1),', ',num2str(Q1)])
% Xedges = [18:2:38];
Xedges = [40:10:140];
x2 = x(:);
y2 = y1(:);
plot_mean = [];
plot_std = [];
for i = 1:length(Xedges)-1
    idx = find(x2 > Xedges(i) & x2 < Xedges(i+1));
    plot_mean(i) = nanmean(y2(idx));
    plot_std(i) = nanstd(y2(idx));
end
errorbar(Xedges(1:end-1)+5,plot_mean,plot_std)
x = (trials_continuous_speed);
y1 = (trials_continuous_oscillation_freq);
[R1,P1,Q1] = plotWithFit2(x(:)',y1(:)','b');
title(['R,P,Q:' num2str(R1),', ',num2str(P1),', ',num2str(Q1)])
x2 = x(:);
y2 = y1(:);
plot_mean = [];
plot_std = [];
for i = 1:length(Xedges)-1
    idx = find(x2 > Xedges(i) & x2 < Xedges(i+1));
    plot_mean(i) = nanmean(y2(idx));
    plot_std(i) = nanstd(y2(idx));
end
errorbar(Xedges(1:end-1)+5,plot_mean,plot_std)

subplot(2,2,3), 
X1 = trials_continuous_temperature(:);
Y1 = trials_continuous_theta(:);
Y2 = trials_continuous_oscillation_freq(:);
Xedges = [18:0.5:38];
Yedges = [6:0.2:15];
[N1,Xedges,Yedges] = histcounts2(X1,Y1,Xedges,Yedges);
[N2,Xedges,Yedges] = histcounts2(X1,Y2,Xedges,Yedges);
colorplot = zeros(length(Yedges)-1,length(Xedges)-1,3);
colorplot(:,:,1) = N1'/max(max(N1))*4;
colorplot(:,:,2) = N2'/max(max(N2))*4;
% colorplot(:,:,3) = N2'/300;
imagesc(1-colorplot), set(gca,'YDIR','normal')

subplot(2,2,4), 
X1 = trials_continuous_speed(:);
Y1 = trials_continuous_theta(:);
Y2 = trials_continuous_oscillation_freq(:);
Xedges = [40:5:140];
Yedges = [6:0.2:15];
[N1,Xedges,Yedges] = histcounts2(X1,Y1,Xedges,Yedges);
[N2,Xedges,Yedges] = histcounts2(X1,Y2,Xedges,Yedges);
colorplot = zeros(length(Yedges)-1,length(Xedges)-1,3);
colorplot(:,:,1) = N1'/max(max(N1))*2;
colorplot(:,:,2) = N2'/max(max(N2))*2;
% colorplot(:,:,3) = N2'/300;
imagesc(1-colorplot), set(gca,'YDIR','normal')


% Next plot
figure('pos',[50 50 900 800]),
for j = 1:size(out1,2)
    color1 = colors{animalID(j)};
for i = 1:size(out1(j).tri,2)
    tri = out1(j).tri{i};
    tri.slope1 = abs(tri.slope1);
    tri.slope1(tri.slope1>0.06) = nan;
    subplot(3,3,1), hold on, plot(tri.temperature,color1,'markersize',8), ylabel('Temperature'), axis tight, grid on
    subplot(3,3,2), hold on, plot(tri.theta,color1,'markersize',8), ylabel('LFP theta frequency'), axis tight, grid on,
    subplot(3,3,3), hold on, plot(tri.speed,color1,'markersize',8), ylabel('Running speed'), axis tight, grid on
    subplot(3,3,4), hold on, plot(abs(tri.slope1),color1,'markersize',8), ylabel('Precession slope'), axis tight, grid on,
    %                 plot([1,trials.total-trials2average],abs([slope2,slope2]),'--b','linewidth',1)
    subplot(3,3,5), hold on,
    plot(tri.theta,'-k','markersize',8),  ylabel('Field oscillation freq'), axis tight, grid on, hold on
    plot(tri.oscillation_freq,color1,'markersize',8), legend({'Theta','Oscillation freq'})
    subplot(3,3,6), hold on, plot(tri.theta_cycles,color1,'markersize',8), ylabel('Theta cycles'), axis tight, grid on
    subplot(3,3,7), hold on, plot(tri.oscillation_freq./tri.theta,color1,'markersize',8), ylabel('frequency ratio'), hold on
    subplot(3,3,8), hold on, plot(tri.oscillation_freq-tri.theta,color1,'markersize',8), hold on
    ylabel('frequency difference'), axis tight, grid on, legend({'Oscillation freq','ACG-freq theta'})
    subplot(3,3,9), hold on, plot(tri.spikecount,color1,'markersize',8), ylabel('Spike count'), axis tight, xlabel('Trials'), grid on
end
end
%% % Monosynaptic connections from CCGs
% MS10: 61, 62, 63, 64 % gamma id: 62
% MS12: 78 (Peter_MS12_170714_122034_concat), 79 (Peter_MS12_170715_111545_concat), 80 (Peter_MS12_170716_172307_concat), 81 (Peter_MS12_170717_111614_concat),
% MS13: 92 (Peter_MS13_171129_105507_concat), 93 (Peter_MS13_171130_121758_concat)
% MS14:
% MS18:
% MS21: 126 (Peter_MS21_180629_110332_concat), 140 (Peter_MS21_180627_143449_concat), 143 (control: Peter_MS21_180719_155941_concat)
% MS22: 139 (Peter_MS22_180628_120341_concat), 127 (Peter_MS22_180629_110319_concat), 144 (control: Peter_MS22_180719_122813_concat)
close all
temp = {'Peter_MS21_180629_110332_concat','Peter_MS21_180627_143449_concat','Peter_MS22_180628_120341_concat','Peter_MS22_180629_110319_concat','Peter_MS13_171129_105507_concat','Peter_MS13_171130_121758_concat','Peter_MS12_170714_122034_concat','Peter_MS12_170715_111545_concat','Peter_MS12_170716_172307_concat','Peter_MS12_170717_111614_concat'};
ids = [126, 140, 139, 127, 78, 79, 80, 81, 92, 93]; % 62
ids = [61,64,78,79,80,81,83,91,92,93,94,126,140,149,153,151,139,127,168,166]; % 62 63
MedialSeptum_Recordings
ccg_mean = [];
ccg_std = [];
ccg_peak = [];
ccg_peak_time = [];
ccg_peak_strenth = [];
transmissionProbability = [];
postSynapticRate = [];
ccg_single_out1 = [];
ccg_single_out2 = [];
waveforms1 = {};
waveforms2 = {};
nSpikes1 = {};
nSpikes2 = {};

for i = 3:length(ids)
    id = ids(i)
    recording = recordings(id);
    recording.datapath = datapath;
    monosynOut = monosynapticCoolingEffect(recording);
    ccg_mean = [ccg_mean; monosynOut.ccg_mean];
    ccg_std = [ccg_std; monosynOut.ccg_std];
    ccg_peak = [ccg_peak; monosynOut.ccg_peak];
    ccg_peak_time = [ccg_peak_time; monosynOut.ccg_peak_time];
    ccg_peak_strenth = [ccg_peak_strenth; monosynOut.ccg_peak_strenth];
    transmissionProbability = [transmissionProbability; monosynOut.trans];
    postSynapticRate = [postSynapticRate; monosynOut.postRate];
    ccg_single_out1 = [ccg_single_out1, monosynOut.ccg_single_out1];
    ccg_single_out2 = [ccg_single_out2, monosynOut.ccg_single_out2];
    waveforms1{i} = monosynOut.waveforms1;
    waveforms2{i} = monosynOut.waveforms2;
    nSpikes1{i} = monosynOut.nSpikes1;
    nSpikes2{i} = monosynOut.nSpikes2;
end

pairs_to_keep = find(ccg_peak_strenth(:,1)>5 & ccg_peak_strenth(:,2)>5 & abs(diff(ccg_peak_time'))'<1 & transmissionProbability(:,1)>0.005 & transmissionProbability(:,2)>0.005);
ccg_mean = ccg_mean(pairs_to_keep,:);
ccg_std = ccg_std(pairs_to_keep,:);
ccg_peak = ccg_peak(pairs_to_keep,:);
ccg_peak_time = ccg_peak_time(pairs_to_keep,:);
ccg_peak_strenth = ccg_peak_strenth(pairs_to_keep,:);
ccg_peak_diff = diff(ccg_peak_time')';
transmissionProbability = transmissionProbability(pairs_to_keep,:);
postSynapticRate = postSynapticRate(pairs_to_keep,:);
ccg_single_out1 = ccg_single_out1(:,pairs_to_keep);
ccg_single_out2 = ccg_single_out2(:,pairs_to_keep);

figure(1000)
subplot(2,3,1), hold on
plot(ccg_peak_time(:,1),ccg_peak_strenth(:,1),'.r','markersize',12), hold on
plot(ccg_peak_time(:,2),ccg_peak_strenth(:,2),'.b','markersize',12),
plot(ccg_peak_time',ccg_peak_strenth','-k'),
xlabel('peak time (ms)'), ylabel('peak strength'), grid on
subplot(2,3,2), hold on
plot(ccg_peak_diff',mean(ccg_peak_strenth'),'.r','markersize',12), hold on, xlim([-1,1])
xlabel('peak difference (ms)'), ylabel('peak strength (STDs)'), grid on
subplot(2,3,3), hold on
temp1 = histcounts(transmissionProbability(:,1),[0:0.01:0.5],'normalization','probability');
temp2 = histcounts(transmissionProbability(:,2),[0:0.01:0.5],'normalization','probability');
plot([0.01:0.01:0.5],temp1,'-r')
plot([0.01:0.01:0.5],temp2,'-b')
% plot(transmissionProbability(:,1),ccg_peak_strenth(:,1),'.r','markersize',12), hold on
% plot(transmissionProbability(:,2),ccg_peak_strenth(:,2),'.b','markersize',12),
% plot(transmissionProbability',ccg_peak_strenth','-k'),
xlabel('Transmission Probability'), %ylabel('peak strength'),
grid on, xlim([0,0.5]), %set(gca, 'XScale', 'log')
% subplot(2,4,4)
% histogram(postSynapticRate(:,2)./postSynapticRate(:,1),[0:0.1:5],'normalization','probability'), grid on, %xlim([-1,1]), hold on
subplot(2,3,4), hold on
plot(ccg_peak_time(:,1),transmissionProbability(:,1),'.r','markersize',12), hold on
plot(ccg_peak_time(:,2),transmissionProbability(:,2),'.b','markersize',12),
plot(ccg_peak_time',transmissionProbability','-k'),
xlabel('peak time (ms)'), ylabel('Transmission Probability'), grid on
subplot(2,3,5), hold on
histogram(ccg_peak_diff,[-1:0.05:1]+0.025,'normalization','probability'), grid on, xlim([-1,1]), hold on
idx = find(abs(ccg_peak_diff)<0.4);
[h,p] = ttest(ccg_peak_diff(idx));
plot(mean(ccg_peak_diff(idx)),0,'vk')
% pd = fitdist(ccg_peak_diff(idx),'Normal');
% x_values = [-1:0.05:1]+0.025;
% y = pdf(pd,x_values);
% plot(x_values,y/21,'LineWidth',2)
text(0.2,0.15,['mean = ' num2str(mean(ccg_peak_diff(idx))),', step = 0.05'])
xlabel('peak difference (ms)')
title(['t-test: k=' num2str(h),',p=' num2str(p)])
subplot(2,3,6), hold on
histogram((diff(transmissionProbability')'),([-0.2:0.01:0.2]+0.005),'normalization','probability'), grid on, xlim([-0.2,0.2])
xlabel('Relative change in transmission probability')
% subplot(2,4,8), hold on
% plot(diff(transmissionProbability')',postSynapticRate(:,2)./postSynapticRate(:,1),'.'), grid on

x_data = [1:size(ccg_single_out1,1)]*0.05;
y_data1 = ccg_single_out1./max(ccg_single_out1);
y_data2 = ccg_single_out2./max(ccg_single_out2);
y_data1_mean = mean(y_data1');
y_data1_std = std(y_data1');
y_data2_mean = mean(y_data2');
y_data2_std = std(y_data2');

figure
subplot(1,2,1)
plot(x_data, y_data1,'r'), hold on
plot(x_data, y_data2,'b'), axis tight, xlabel('Time (ms)'), title('EPSPs')

subplot(1,2,2), hold on
patch([x_data,flip(x_data)], [y_data1_mean+y_data1_std,flip(y_data1_mean-y_data1_std)],'r','EdgeColor','none','FaceAlpha',.2)
plot(x_data, y_data1_mean, 'color', 'r','linewidth',2), grid on

patch([x_data,flip(x_data)], [y_data2_mean+y_data2_std,flip(y_data2_mean-y_data2_std)],'b','EdgeColor','none','FaceAlpha',.2)
plot(x_data, y_data2_mean, 'color', 'b','linewidth',2), grid on, axis tight, xlabel('Time (ms)'), title('Average EPSP')

%% % Waveform estimates
waveforms_raw1 = {};
waveforms_raw2 = {};
waveforms_filt1 = {};
waveforms_filt2 = {};

for i = 1:length(waveforms1)
    waveforms_raw1 = [waveforms_raw1,waveforms1{i}.rawWaveform];
    waveforms_raw2 = [waveforms_raw2,waveforms2{i}.rawWaveform];
    waveforms_filt1 = [waveforms_filt1,waveforms1{i}.filtWaveform];
    waveforms_filt2 = [waveforms_filt2,waveforms2{i}.filtWaveform];
end
waveforms_raw1 = vertcat(waveforms_raw1{:})';
waveforms_raw2 = vertcat(waveforms_raw2{:})';
waveforms_filt1 = vertcat(waveforms_filt1{:})';
waveforms_filt2 = vertcat(waveforms_filt2{:})';
waveforms_nSpikes1 = [nSpikes1{:}];
waveforms_nSpikes2 = [nSpikes2{:}];
idx = find(waveforms_nSpikes1>500 & waveforms_nSpikes2>500);

waveforms_raw1 = waveforms_raw1(:,idx);
waveforms_raw2 = waveforms_raw2(:,idx);
waveforms_filt1 = waveforms_filt1(:,idx);
waveforms_filt2 = waveforms_filt2(:,idx);

figure,
subplot(2,3,1)
plot(waveforms_filt1./range(waveforms_filt1),'r'), hold on
plot(waveforms_filt2./range(waveforms_filt1),'b'), axis tight,
title('Filtered waveforms'), ylabel('Normalized to first state'), xlabel('Samples')
subplot(2,3,2)
plot(range(waveforms_filt1),range(waveforms_filt2),'.'), refline, %set(gca, 'XScale', 'log'), set(gca, 'YScale', 'log')
axis equal, title('Filt waveforms range'), xlabel('State1'), ylabel('State2'), xlim([0,800]), ylim([0,800])
subplot(2,3,3)
plot(sum(abs(waveforms_filt1)),sum(abs(waveforms_filt2)),'.'), refline, %set(gca, 'XScale', 'log'), set(gca, 'YScale', 'log')
axis equal, title('Filt waveforms difference'), xlabel('State1'), ylabel('State2'), xlim([0,2000]), ylim([0,2000])
subplot(2,3,4)
data1 = range(waveforms_filt2)./range(waveforms_filt1);
histogram(data1,[0.7:0.01:1.3]), grid on, hold on
[h,p] = ttest(data1);
plot(nanmean(data1),0,'vk')
title(['t-test: k=' num2str(h),',p=' num2str(p)]), xlabel(['waveform range, mean=', num2str(nanmean(data1))])
subplot(2,3,5)
data2 = sum(abs(waveforms_filt2)-abs(waveforms_filt1))./sum(abs(waveforms_filt1));
histogram(data2,[-0.3:0.01:0.3]), grid on, hold on
plot(nanmean(data2),0,'vk')
[h,p] = ttest(data2);
title(['t-test: k=' num2str(h),',p=' num2str(p)]), xlabel(['waveform difference, mean=', num2str(nanmean(data2))])

%% Interneuron/Pyramidal rate change by theta/temperature

close all
temp = {'Peter_MS21_180629_110332_concat','Peter_MS21_180627_143449_concat','Peter_MS22_180628_120341_concat','Peter_MS22_180629_110319_concat','Peter_MS13_171129_105507_concat','Peter_MS13_171130_121758_concat','Peter_MS12_170714_122034_concat','Peter_MS12_170715_111545_concat','Peter_MS12_170716_172307_concat','Peter_MS12_170717_111614_concat'};
ids = [126, 140, 139, 127, 78, 79, 80, 81, 92, 93]; % 62
ids = [78,79,80,81,83,91,92,93,94,126,140,149,153,151,139,127,168,166]; % 62 63, 61,64,
MedialSeptum_Recordings
slope1 = [];
slope2 = [];
slope3 = [];
putativeCellType = {};
k = 1;
k2 = 1;
fig_num = 0;
% celltype = 'Interneuron';
celltype = 'Pyramidal';
for iii = 1:length(ids)
    iii
    id = ids(iii);
    recording = recordings(id);
    recording.datapath = datapath;
    cd(fullfile(recording.datapath, recording.animal_id, recording.name))
    spikes = loadSpikes('clusteringpath',recording.SpikeSorting.path,'clusteringformat',recording.SpikeSorting.method,'basename',recording.name);
    load(fullfile(recording.SpikeSorting.path,[recording.name,'.cell_metrics.cellinfo.mat']));
    [CellIndexes,cell_metrics] = loadCellMetrics('cell_metrics',cell_metrics,'putativeCellType',{celltype});
    for i = 1:length(CellIndexes)
        
        spkdiff = diff(spikes.times{CellIndexes(i)});
        spkdiff = [spkdiff;spkdiff(end)];
        %     spkdiff = 1./diff(spikes.times{InterneuronIndexes(i)});
        indx = find(spikes.speed{CellIndexes(i)} >10 & spikes.speed{CellIndexes(i)} <140 &  spikes.pos_linearized{CellIndexes(i)}>0);
        y1 = 1./nanconv(spkdiff(indx),ones(40,1)/40,'edge');
        %         y1 = nanconv(spkdiff(indx),gausswin(50)/sum(gausswin(50)),'edge');
        
        if length(indx)>300
            if rem(k,5)==1
                if k>1
                    if strcmp(celltype,'Pyramidal')
                        saveas(fig,fullfile('K:\Dropbox\Buzsakilab Postdoc\MatlabFigures\PyramidalCellResponse',['PyramidalCell_',num2str(fig_num),'.png']));
                    else
                        saveas(fig,fullfile('K:\Dropbox\Buzsakilab Postdoc\MatlabFigures\InterneuronResponse',['Interneurons_',num2str(fig_num),'.png']));
                    end
                end
                fig = figure('Position', [100 100 2000 1200]);
                fig_num = fig_num+1;
            end
            if rem(k,5)== 0
                column = 4*3;
            else
                column = (rem(k,5)-1)*3;
            end
            % % % %
            % Temperature
            subplot(5,3,1+column)
            x = spikes.temperature{CellIndexes(i)}(indx);
            [R,P] = corrcoef(x,y1);
            P1 = polyfit(x,y1,1); yfit = P1(1)*x+P1(2);
            if P(2,1)<0.05
                color = '.b';
                slope1(k) = P1(1);
            else
                color = '.k';
                slope1(k) = nan;
            end
            plot(x, y1,color), grid on, axis tight, hold on
            plot(x,yfit,'-k');
            title(['Unit #' num2str(CellIndexes(i))])
            xlabel(['R,P: ' num2str(R(2,1),3),',', num2str(P(2,1),3)])
            ylabel([num2str(P1(1),3),' Hz/degree']), ylim([0,min([200,max(y1)])]),
            putativeCellType{k} = cell_metrics.putativeCellType{CellIndexes(i)};
            
            % % % %
            % Theta
            subplot(5,3,2+column)
            x = spikes.theta_freq{CellIndexes(i)}(indx);
            [R,P] = corrcoef(x,y1);
            P1 = polyfit(x,y1,1); yfit = P1(1)*x+P1(2);
            if P(2,1)<0.05
                color = '.r';
                slope2(k) = P1(1);
            else
                color = '.k';
                slope2(k) = nan;
            end
            plot(x, y1,color), grid on, axis tight, hold on
            plot(x,yfit,'-k');
            title(cell_metrics.putativeCellType{CellIndexes(i)})
            xlabel(['R,P: ' num2str(R(2,1),3),',', num2str(P(2,1),3)])
            ylabel([num2str(P1(1),3),' Hz/degree']), ylim([0,min([200,max(y1)])]),
            
            % % % %
            % Speed
            subplot(5,3,3+column)
            x = spikes.speed{CellIndexes(i)}(indx);
            [R,P] = corrcoef(x,y1);
            P1 = polyfit(x,y1,1); yfit = P1(1)*x+P1(2);
            if P(2,1)<0.05
                color = '.m';
                slope3(k) = P1(1);
            else
                color = '.k';
                slope3(k) = nan;
            end
            plot(x, y1,color), grid on, axis tight, hold on
            plot(x,yfit,'-k');
            title(recording.name)
            xlabel(['R,P: ' num2str(R(2,1),3),',', num2str(P(2,1),3)])
            ylabel([num2str(P1(1),3),' Hz/degree']), ylim([0,min([200,max(y1)])]),
            
            k = k+1;
        end
    end
end

idx_narrow = find(strcmp(putativeCellType,'Narrow Interneuron'));
idx_wide = find(strcmp(putativeCellType,'Wide Interneuron'));
bin1 = [-1:0.05:1];
figure, subplot(1,3,1)
plt1 = histcounts(slope1,bin1);
% plt1 = histcounts(slope1(idx_narrow),bin1);
% plt2 = histcounts(slope1(idx_wide),bin1);
stairs(bin1(1:end-1),plt1), hold on, grid on
% stairs(bin1(1:end-1),plt2)
xlabel('Slope (Rate (Hz)/ temperature (degree C))'), title('Temperature')

subplot(1,3,2)
bin2 = [-6:0.2:6];
plt1 = histcounts(slope2,bin2);
% plt1 = histcounts(slope2(idx_narrow),bin2);
% plt2 = histcounts(slope2(idx_wide),bin2);
stairs(bin2(1:end-1),plt1), hold on, grid on
% stairs(bin2(1:end-1),plt2), legend({'Narrow','Wide'})
xlabel('Slope (Rate (Hz)/ theta (Hz))'), title('Theta')

subplot(1,3,3)
bin3 = [-0.3:0.01:0.3];
plt1 = histcounts(slope3,bin3);
% plt1 = histcounts(slope3(idx_narrow),bin3);
% plt2 = histcounts(slope3(idx_wide),bin3);
stairs(bin3(1:end-1),plt1), hold on, grid on
% stairs(bin3(1:end-1),plt2), legend({'Narrow','Wide'})
xlabel('Slope (Rate (Hz)/ speed (m/s))'), title('Speed')

%% % Gamma band analysis
datasets_MS10 = [58,59,60,61,62,66,71,72]; % 64, 63, 70
datasets_MS12 = [77,78,79,80,81,82,84]; % 83
datasets_MS13 = [91,92,93,94]; % 90 88
datasets_MS21 = [122,126,140,147,151,152,149,146,156,157,161,155,158,160]; % 148, 150 (duplicated)
datasets_MS22 = [139,127,163,164,145,165,166,169,162,167,168]; % 123
datasets = [datasets_MS10,datasets_MS12,datasets_MS13,datasets_MS21,datasets_MS22];

% datasets = [126, 140, 139, 127, 78, 79, 80, 81, 92, 93]; % 62
for iSession=3%1:length(datasets)
    datasets(iSession)
    recording = recordings(datasets(iSession));
    disp(['Processing ', num2str(iSession),'/',num2str(length(datasets)),' recordingID = ' num2str(datasets(iSession)),', basename = ' recording.name])
    basepath = fullfile(datapath, recording.animal_id, recording.name);
    cd(basepath)
    winopen(basepath);
%     [session, basename, basepath, clusteringpath] = db_set_path('session',recording.name);
%     classification_DeepSuperficial(session)
end


%% % Phase amplitude coupling
clear all, close all
MedialSeptum_Recordings
datasets_MS10 = [60,62,71,72];
datasets_MS12 = [77,78,79,80,81,82];
datasets_MS13 = [92,93,94];
datasets_MS21 = [147,149,140,151,126,152,155,156];
datasets_MS22 = [163,164,165,139,127,166,167,168,169,170];
datasets = [datasets_MS10,datasets_MS12,datasets_MS13,datasets_MS21,datasets_MS22];

phaserange = [4.6:0.2:12];
amprange = [16:2:100,100:4:200];
state_labels = {'PreCooling','Cooling','PostCooling'};
% ID 145->
for iSession=1:length(datasets)
    datasets(iSession)
    recording = recordings(datasets(iSession));
    disp(['Processing ', num2str(iSession),'/',num2str(length(datasets)),' recordingID = ' num2str(datasets(iSession)),', basename = ' recording.name])
    [session, basename, basepath, clusteringpath] = db_set_path('session',recording.name);
    
    sessionInfo = bz_getSessionInfo(pwd, 'noPrompts', true);
    %     xml = LoadXml(fullfile([recording.name, '.xml']));
    %     [session, basename, basepath, clusteringpath] = db_set_path('session',recording.name);
    load('trials.mat')
    load('animal.mat')
    thetaGammaPhaseCoupling = {};
    if exist([recording.name, '.ThetaGammaPhaseCoupling.channelInfo.mat'])
        load([recording.name, '.ThetaGammaPhaseCoupling.channelInfo.mat']);
    end
    
    
    channel = session.channelTags.GammaPhaseCoupling.channels;
    iShank = session.channelTags.GammaPhaseCoupling.spikeGroups;
    % %     for iShank = 1:size(sessionInfo.AnatGrps,2)
    %         disp(['Processing shank ' num2str(iShank),'/' num2str(size(sessionInfo.AnatGrps,2))])
    %         channel = sessionInfo.AnatGrps(iShank).Channels(end);
    %         channel = session.extracellular.spikeGroups.channels{iShank}(end);
    
    comod4 = {};
    climits = [];
    comod5 = {};
    climits2 = [];
    
    for iStates = 1:3
        trials_idx = setdiff(find(trials.cooling==iStates),trials.error);
        intervals = [animal.time(trials.start(trials_idx));animal.time(trials.end(trials_idx))]';
        lfp = bz_GetLFP(channel-1,'basepath',basepath,'intervals',intervals);
        lfp2 = lfp(1);
        lfp2.duration = sum(vertcat(lfp.duration));
        lfp2.data = vertcat(lfp.data);
        lfp2.timestamps = vertcat(lfp.timestamps);
        [comod4{iStates}] = bz_ModIndex(lfp2,phaserange,amprange,0);
        climits(iStates) = max(max(comod4{iStates}(8:end,:)));
        %                 [comod5{iStates}] = bz_ModIndex2(lfp2,phaserange,amprange,0);
        %                 climits2(iStates) = max(max(comod5{iStates}(8:end,:)));
    end
    fig = figure('Position', [50 50 1800 800]);
    for iStates = 1:3
        figure(fig)
        subplot(1,3,iStates)
        imagesc(phaserange(1:end-1),amprange(1:end-1),comod4{iStates})
        colormap jet, hold on, xlabel('Frequency phase'); ylabel('Frequency amplitude'), axis xy
        title(state_labels{iStates})
        clim([0,0.9*max(climits)])
        %                 subplot(2,3,iStates+3)
        %                 imagesc(phaserange(1:end-1),amprange(1:end-1),comod5{iStates})
        %                 colormap jet, hold on, xlabel('Frequency phase'); ylabel('Frequency amplitude'), axis xy
        %                 title(state_labels{iStates})
        %                 clim([0,0.9*max(climits2)])
        if iStates==3
            colorbar
        end
        drawnow
    end
    thetaGammaPhaseCoupling.comod4{channel} = comod4;
    thetaGammaPhaseCoupling.phaserange{channel} = phaserange;
    thetaGammaPhaseCoupling.amprange{channel} = amprange;
    thetaGammaPhaseCoupling.shank{channel} = iShank;
    %             save([recording.name, '.ThetaGammaPhaseCoupling_channel_',num2str(channel),'_iShank_',num2str(iShank),'.channelInfo.mat'],'thetaGammaPhaseCoupling');
    disp('Saving figure')
    saveas(fig,fullfile('C:\Users\peter\Dropbox\Buzsakilab Postdoc\GammaAnalysis',[recording.name, '.ThetaGammaPhaseCoupling.ch',num2str(channel),'.png']));
    
    saveStruct(thetaGammaPhaseCoupling,'session',session,'datatype','channelInfo');
    %     end
    
end
disp('Finished running the gamma analysis')

%% % Comodulogram
clear all, close all
MedialSeptum_Recordings
datasets_MS10 = [60,62,71,72];
datasets_MS12 = [77,78,79,80,81,82];
datasets_MS13 = [92,93,94];
datasets_MS21 = [147,149,140,151,126,152,155,156];
datasets_MS22 = [163,164,165,139,127,166,167,168,169,170];
datasets = [datasets_MS10,datasets_MS12,datasets_MS13,datasets_MS21,datasets_MS22];

state_labels = {'PreCooling','Cooling','PostCooling'};
specparms.frange = [4,120];
specparms.nfreqs = 100;
specparms.space = 'log';
specparms.ncyc = 5;

for iSession=30%:length(datasets)
    datasets(iSession)
    recording = recordings(datasets(iSession));
    disp(['Processing ', num2str(iSession),'/',num2str(length(datasets)),' recordingID = ' num2str(datasets(iSession)),', basename = ' recording.name])
    [session, basename, basepath, clusteringpath] = db_set_path('session',recording.name);
    
    sessionInfo = bz_getSessionInfo(pwd, 'noPrompts', true);
    %     xml = LoadXml(fullfile([recording.name, '.xml']));
    %     [session, basename, basepath, clusteringpath] = db_set_path('session',recording.name);
    load('trials.mat')
    load('animal.mat')
    thetaGammaPhaseCoupling = {};
    if exist([recording.name, '.ThetaGammaPhaseCoupling.channelInfo.mat'])
        load([recording.name, '.ThetaGammaPhaseCoupling.channelInfo.mat']);
    end
    
    channel = session.channelTags.GammaPhaseCoupling.channels;
    iShank = session.channelTags.GammaPhaseCoupling.spikeGroups;
    % %     for iShank = 1:size(sessionInfo.AnatGrps,2)
    %         disp(['Processing shank ' num2str(iShank),'/' num2str(size(sessionInfo.AnatGrps,2))])
    %         channel = sessionInfo.AnatGrps(iShank).Channels(end);
    %         channel = session.extracellular.spikeGroups.channels{iShank}(end);
    
    comod4 = {};
    climits = [];
    comod5 = {};
    climits2 = [];
    
    for iStates = 1:3
        trials_idx = setdiff(find(trials.cooling==iStates),trials.error);
        intervals = [animal.time(trials.start(trials_idx));animal.time(trials.end(trials_idx))]';
        lfp = bz_GetLFP(channel-1,'basepath',basepath,'intervals',intervals);
        lfp2 = lfp(1);
        lfp2.duration = sum(vertcat(lfp.duration));
        lfp2.data = vertcat(lfp.data);
        lfp2.timestamps = vertcat(lfp.timestamps);
        
        
        [comod4{iStates}] = bz_Comodulogram(lfp2,specparms);
%         [comod4{iStates}] = bz_ModIndex(lfp2,phaserange,amprange,0);
%         climits(iStates) = max(max(comod4{iStates}(8:end,:)));
        %                 [comod5{iStates}] = bz_ModIndex2(lfp2,phaserange,amprange,0);
        %                 climits2(iStates) = max(max(comod5{iStates}(8:end,:)));
    end
    fig = figure('Position', [50 50 1800 800]);
    for iStates = 1:3
        figure(fig)
        subplot(1,3,iStates)
        imagesc(comod4{iStates}.freqs,comod4{iStates}.freqs,comod4{iStates}.corrs)
        colormap jet, hold on, xlabel('Frequency'); ylabel('Frequency'), 
        title(state_labels{iStates})
%         clim([0,0.9*max(climits)])
        %                 subplot(2,3,iStates+3)
        %                 imagesc(phaserange(1:end-1),amprange(1:end-1),comod5{iStates})
        %                 colormap jet, hold on, xlabel('Frequency phase'); ylabel('Frequency amplitude'), axis xy
        %                 title(state_labels{iStates})
        %                 clim([0,0.9*max(climits2)])
        if iStates==3
            colorbar
        end
        drawnow
    end
    thetaGammaComodulogram.comod4{channel} = comod4;
    thetaGammaComodulogram.shank{channel} = iShank;

    save([recording.name, '.thetaGammaComodulogram_channel_',num2str(channel),'_iShank_',num2str(iShank),'.channelInfo.mat'],'thetaGammaComodulogram');
    disp('Saving figure')
    saveas(fig,fullfile('C:\Users\peter\Dropbox\Buzsakilab Postdoc\GammaAnalysis',[recording.name, '.thetaGammaComodulogram.ch',num2str(channel),'.png']));
    
    saveStruct(thetaGammaComodulogram,'channelInfo','session',session);
end
disp('Finished running thetaGamma comodulogram analysis')

%% % Batch analysis of theta and gamma power across recording channels
clear all, close all
MedialSeptum_Recordings
datasets_MS10 = [58,59,60,61,62,66,71,72]; % 8 sessions. More? 182
datasets_MS12 = [77,78,79,80,81,82,83,84,86,179]; % 10 sessions.
datasets_MS13 = [90,91,92,93,94,173,174,175]; % 8 sessions. More? 176,178
datasets_MS14 = [108];
datasets_MS21 = [122,126,140,146,147,149,151,152,153,155,156,157,158,160,161]; % 15 sessions
datasets_MS22 = [123,127,139,145,162,163,164,165,166,167,168,169]; % 12 sessions
datasets = [datasets_MS10,datasets_MS12,datasets_MS13,datasets_MS21,datasets_MS22];

% datasets = [126, 140, 139, 127, 78, 79, 80, 81, 92, 93]; % 62

for iSession=1:length(datasets)
    recording = recordings(datasets(iSession));
    disp(['Processing ', num2str(iSession),'/',num2str(length(datasets)),' recordingID = ' num2str(datasets(iSession)),', basename = ' recording.name])
    basepath = fullfile(datapath, recording.animal_id, recording.name);
    region = CA1subregionsFromPowerspectrums(basepath);
end

%% % Batch analysis of gamma power changes across trials
clear all, close all
MedialSeptum_Recordings
% datasets_MS10 = [58,59,60,61,62,66,70,71,72]; % 64, 63
% datasets_MS12 = [77,78,79,80,81,82,84]; % 83
% datasets_MS13 = [91,92,93,94,88]; % 90
% datasets_MS21 = [122,126,140,147,150,151,152,149,146,156,157,161,155,158,160]; % 148
% datasets_MS22 = [139,127,163,164,123,145,165,166,169,162,167,168]; %
datasets_MS10 = [58,59,60,61,62,66,71,72,70]; % 8 sessions. More? 181,182
datasets_MS12 = [77,78,79,80,81,82,83,84,86,179,180]; % 11 sessions.
datasets_MS13 = [90,91,92,93,94,173,174,175,88]; % 8 sessions. More? 176,178
datasets_MS14 = [108];
datasets_MS21 = [122,126,140,146,147,149,151,152,153,155,156,157,158,160,161]; % 15 sessions
datasets_MS22 = [123,127,139,145,162,163,164,165,166,167,168,169]; % 12 sessions


datasets = [datasets_MS10,datasets_MS12,datasets_MS13,datasets_MS21,datasets_MS22];
% datasets = [126, 140, 139, 127, 78, 79, 80, 81, 92, 93]; % 62
datasets = [83,86,179,90,173,174,175,149,153,161,139]
for iSession=3:length(datasets)
    % % % % % % % % % %
    % Loading session
    % % % % % % % % % %
    recording = recordings(datasets(iSession));
    disp(['Processing ', num2str(iSession),'/',num2str(length(datasets)),' recordingID = ' num2str(datasets(iSession)),', basename = ' recording.name])
    basepath = fullfile(datapath, recording.animal_id, recording.name);
    cd(basepath)
    
    % % % % % % % % % %
    % find best channels for the gamma analysis
    % % % % % % % % % %
    
    % Low gamma (35-80)
    disp('Determining best channel for low gamma (35Hz - 80Hz)')
    powerProfile_lowGamma = bz_PowerSpectrumProfile([35 80]);
    [~,peakChannel_lowGamma] = max(powerProfile_lowGamma.mean);
    
    % Mid gamma (60-140)
    disp('Determining best channel for mid gamma (60Hz - 140Hz)')
    powerProfile_midGamma = bz_PowerSpectrumProfile([60 140]);
    [~,peakChannel_midGamma] = max(powerProfile_midGamma.mean);
    % % % % % % % % % %
    % Load trial-limited LFP trace and calculate gamma power for these time periods
    % % % % % % % % % %
    load('trials.mat')
    load('animal.mat')
    lowGamma = [];
    midGamma = [];
    parfor(iTrial = 1:trials.total,18)
        intervals = [animal.time(trials.start(iTrial));animal.time(trials.end(iTrial))]';
        % Low gamma
        lfp = bz_GetLFP(peakChannel_lowGamma-1,'basepath',basepath,'intervals',intervals);
        lfp = bz_Filter(lfp,'passband',[35,80],'filter','butter');
        lowGamma(iTrial) = sum(lfp.data.^2)/length(lfp.data);
        
        % Mid gamma
        lfp = bz_GetLFP(peakChannel_midGamma-1,'basepath',basepath,'intervals',intervals);
        lfp = bz_Filter(lfp,'passband',[60,140],'filter','butter');
        midGamma(iTrial) = sum(lfp.data.^2)/length(lfp.data);
    end
    trials.power.lowGamma = lowGamma;
    trials.power.midGamma = midGamma;
    fig = figure('Position', [50 50 1800 800]);
    plot(trials.power.lowGamma), hold on
    plot(trials.power.midGamma)
    legend({['Low-gamma ch:' num2str(peakChannel_lowGamma)],['Mid-gamma ch:' num2str(peakChannel_midGamma)]})
    title(['Gamma band analysis for ' recording.name])
    xlabel('Trials'), ylabel('Power'), axis tight
    saveas(fig,'TrialwiseGammaPower.png');
    trialsGamma = trials;
    save('trialsGamma.mat','trialsGamma')
    drawnow
end


%% % Theta vs Speed
clear all, close all
MedialSeptum_Recordings

% Control sets
% MS10:
% MS12:
% MS13:
% MS14: 107
% MS18: 116,141,142
% MS21: 143,154,159
% MS22: 144
datasets_MS10 = [58,59,60,61,62,66,71,72]; % 8 sessions. More? 182
datasets_MS12 = [77,78,79,80,81,82,83,84,179]; % 9 sessions. %86
datasets_MS13 = [90,91,92,93,94,174,175]; % 7 sessions. More? 176,178, 173
datasets_MS14 = [108];
datasets_MS21 = [122,126,140,146,147,149,151,152,153,155,156,157,158,160,161]; % 15 sessions
datasets_MS22 = [123,127,139,145,162,163,164,165,166,167,168,169]; % 12 sessions

% datasets_MS10 = [58,59,60,61,62,66,71,72]; % 64, 63, 70 % PROBLEMS WITH lowpass theta: 60,61,62
% datasets_MS12 = [78,79,80,81,82,84]; % 83, 77
% datasets_MS13 = [91,92,93,94]; % 90 88
% datasets_MS21 = [122,126,140,147,151,152,149,146,156,157,161,155,158,160]; % 148, 150 (duplicated)
% datasets_MS22 = [139,127,163,164,145,165,166,169,162,167,168]; % 123
datasets = [datasets_MS10,datasets_MS12,datasets_MS13,datasets_MS21,datasets_MS22];
animalid2 = {datasets_MS10,datasets_MS12,datasets_MS13,datasets_MS21,datasets_MS22};
% datasets = [datasets_MS10];
% animalid2 = {datasets_MS10};
animal_names = {'MS10','MS12','MS13','MS21','MS22'};
animalid = [];
for i = 1:length(animalid2)
    animalid = [animalid,i*ones(1,length(animalid2{i}))];
end
animal_speed_max = [120,120,120,120,120];
thetaVsSpeed_all = {};
colors = {'r','b','g'};
colors1 = {'or','ob','og'};
conditions = {'Pre','Cooling','Post'};

for k = 1:length(datasets)
    recording = recordings(datasets(k));
    disp(['Processing ', num2str(k),'/',num2str(length(datasets)),' recordingID = ' num2str(datasets(k)),', basename = ' recording.name])
    cd(fullfile(datapath, recording.animal_id, recording.name))
    [session, basename, basepath, clusteringpath] = db_set_path('session',recording.name);
    
    % Animal
    disp('Loading animal structure')
    load([recording.name,'.animal.behavior.mat'])
    
    % Trials
    disp('Loading trials structure')
    load([recording.name,'.trials.behavior.mat'])
    
    % Loading cooling structure
    disp('Loading cooling structure')
    load([recording.name,'.cooling.manipulation.mat'])
    
    interval = [animal.time(1),animal.time(end)];
    session.extracellular.srLfp = 1250;
    thetaVsSpeed = plot_ThetaVsSpeed4(session,animal,cooling,interval,animal_speed_max(animalid(k)));
    saveas(gcf,['K:\Dropbox\Buzsakilab Postdoc\MatlabFigures\ThetaVsSpeed_Averages\',basename,'_plot_ThetaVsSpeed_3states.png'])
    thetaVsSpeed_all{k} = thetaVsSpeed;
    
%     figure(101)
% %     for j = 1:3
%         subplot(1,3,1)
%         plot(thetaVsSpeed.freqlist,thetaVsSpeed.powerSpec{j},colors{j}), hold on
%         title('Cooling'), xlabel('Frequency (Hz)'),ylabel('Power'), title('Average freq')
%     end
%         subplot(1,3,2)
%         plot(thetaVsSpeed.peakFreq{1},thetaVsSpeed.peakFreq{2},colors{1}), hold on
%         plot(thetaVsSpeed.peakFreq{3},thetaVsSpeed.peakFreq{2},colors{3}), hold on
%         xlabel('Speed (cm/s)'), ylabel('Frequency (Hz)'),title('Frequency Cooling')
%         
%         subplot(1,3,3)
%         plot(thetaVsSpeed.peakPower{1},thetaVsSpeed.peakPower{2},colors{1}), hold on,
%         plot(thetaVsSpeed.peakPower{3},thetaVsSpeed.peakPower{2},colors{3})
%         ylabel('Power'), xlabel('Speed (cm/s)')
    
end

save(['K:\Dropbox\Buzsakilab Postdoc\MatlabWorkspaces\','thetaVsSpeed_three_states.mat'],'thetaVsSpeed_all')

%%
colors0 = {'r','b','g','m','c'};
colors1 = {'-r','-b','-g','-m','-c'};
colors2 = {'--r','--b','--g','--m','--c'};

for j = 1%:5 % Animals
    idx = 1:length(animalid);
%     idx = find(animalid==j);
    % Power spectrum
    data_Cooling1 = zeros(length(thetaVsSpeed.freqlist),length(idx));
    data_Cooling2 = zeros(length(thetaVsSpeed.freqlist),length(idx));
    data_Cooling3 = zeros(length(thetaVsSpeed.freqlist),length(idx));
    for k = 1:length(idx)
        data_Cooling1(:,k) = thetaVsSpeed_all{idx(k)}.powerSpec{1};
        data_Cooling2(:,k) = thetaVsSpeed_all{idx(k)}.powerSpec{2};
        data_Cooling3(:,k) = thetaVsSpeed_all{idx(k)}.powerSpec{3};
    end
    figure(102)
    subplot(2,3,1)
    t_axis = thetaVsSpeed.freqlist;
    plot_mean = mean(data_Cooling1');
    plot_std = sem(data_Cooling1');
    patch([t_axis,flip(t_axis)], [plot_mean+plot_std,flip(plot_mean-plot_std)],colors0{1},'EdgeColor','none','FaceAlpha',.2), hold on
    plot(t_axis, plot_mean,colors1{1}), xlabel('Theta frequency (Hz)'), ylabel('Power'), axis tight, %xlim(xlimits)
    
    plot_mean = mean(data_Cooling2');
    plot_std = sem(data_Cooling2');
    patch([t_axis,flip(t_axis)], [plot_mean+plot_std,flip(plot_mean-plot_std)],colors0{2},'EdgeColor','none','FaceAlpha',.2), hold on
    plot(t_axis, plot_mean,colors1{2}), axis tight, %xlim(xlimits)
    
    plot_mean = mean(data_Cooling3');
    plot_std = sem(data_Cooling3');
    patch([t_axis,flip(t_axis)], [plot_mean+plot_std,flip(plot_mean-plot_std)],colors0{3},'EdgeColor','none','FaceAlpha',.2), hold on
    plot(t_axis, plot_mean,colors1{3}), axis tight, %xlim(xlimits)
    
    % Theta freq vs speed
    speed_intervals = [10:3:animal_speed_max];
    speed_intervals = mean([speed_intervals(2:end);speed_intervals(1:end-1)]);
    t_axis = speed_intervals;
    data_cooling1 = nan(length(t_axis),length(idx));
    data_cooling2 = nan(length(t_axis),length(idx));
    data_cooling3 = nan(length(t_axis),length(idx));

    for k = 1:length(idx)
        [~,idx2,~] = intersect(round(2*speed_intervals),round(2*thetaVsSpeed_all{idx(k)}.peakFreq1{1}));
        data_cooling1(idx2,k) = thetaVsSpeed_all{idx(k)}.peakFreq{1};
        [~,idx2,~] = intersect(round(2*speed_intervals),round(2*thetaVsSpeed_all{idx(k)}.peakFreq1{2}));
        data_cooling2(idx2,k) = thetaVsSpeed_all{idx(k)}.peakFreq{2};
        [~,idx2,~] = intersect(round(2*speed_intervals),round(2*thetaVsSpeed_all{idx(k)}.peakFreq1{3}));
        data_cooling3(idx2,k) = thetaVsSpeed_all{idx(k)}.peakFreq{3};
    end
    
    figure(102)
    subplot(2,3,2)
    plot_mean = nanmean(data_cooling1');
    plot_std = nansem(data_cooling1');
    patch([t_axis,flip(t_axis)], [plot_mean+plot_std,flip(plot_mean-plot_std)],colors0{1},'EdgeColor','none','FaceAlpha',.2), hold on
    plot(t_axis, plot_mean,colors1{1}),
    
    plot_mean = nanmean(data_cooling2');
    plot_std = nansem(data_cooling2');
    patch([t_axis,flip(t_axis)], [plot_mean+plot_std,flip(plot_mean-plot_std)],colors0{2},'EdgeColor','none','FaceAlpha',.2), hold on
    plot(t_axis, plot_mean,colors1{2}), ylabel('Theta frequency (Hz)'), axis tight, xlabel('Running speed (cm/s)')
    
    plot_mean = nanmean(data_cooling3');
    plot_std = nansem(data_cooling3');
    patch([t_axis,flip(t_axis)], [plot_mean+plot_std,flip(plot_mean-plot_std)],colors0{3},'EdgeColor','none','FaceAlpha',.2), hold on
    plot(t_axis, plot_mean,colors1{3}), ylabel('Theta frequency (Hz)'), axis tight, xlabel('Running speed (cm/s)')
    
    subplot(2,3,5)
    x = nanmean(data_cooling1');
    y1 = nanmean(data_cooling2');
    P1 = polyfit(x,y1,1); yfit = P1(1)*x+P1(2);
    plot(x, y1,['o',colors0{j}]), hold on
    plot(x,yfit,'-k');
    subplot(2,3,4), hold on
    plot([1,2],P1,colors0{j})
    
    % Theta power
    % Theta freq vs speed
    speed_intervals = [10:3:animal_speed_max];
    speed_intervals = mean([speed_intervals(2:end);speed_intervals(1:end-1)]);
    t_axis = speed_intervals;
    data_cooling1 = nan(length(t_axis),length(idx));
    data_cooling2 = nan(length(t_axis),length(idx));
    data_cooling3 = nan(length(t_axis),length(idx));
    for k = 1:length(idx)
        [~,idx2,~] = intersect(round(2*speed_intervals),round(2*thetaVsSpeed_all{idx(k)}.peakPower1{1}));
        data_cooling1(idx2,k) = thetaVsSpeed_all{idx(k)}.peakPower{1};
        [~,idx2,~] = intersect(round(2*speed_intervals),round(2*thetaVsSpeed_all{idx(k)}.peakPower1{2}));
        data_cooling2(idx2,k) = thetaVsSpeed_all{idx(k)}.peakPower{2};
        [~,idx2,~] = intersect(round(2*speed_intervals),round(2*thetaVsSpeed_all{idx(k)}.peakPower1{3}));
        data_cooling3(idx2,k) = thetaVsSpeed_all{idx(k)}.peakPower{3};
    end
    
    figure(102)
    subplot(2,3,3)
    plot_mean = nanmean(data_cooling1');
    plot_std = nansem(data_cooling1');
    patch([t_axis,flip(t_axis)], [plot_mean+plot_std,flip(plot_mean-plot_std)],colors0{1},'EdgeColor','none','FaceAlpha',.2), hold on
    plot(t_axis, plot_mean,colors1{1})
    
    plot_mean = nanmean(data_cooling2');
    plot_std = nansem(data_cooling2');
    patch([t_axis,flip(t_axis)], [plot_mean+plot_std,flip(plot_mean-plot_std)],colors0{2},'EdgeColor','none','FaceAlpha',.2), hold on
    plot(t_axis, plot_mean,colors1{2}), 
    
    plot_mean = nanmean(data_cooling3');
    plot_std = nansem(data_cooling3');
    patch([t_axis,flip(t_axis)], [plot_mean+plot_std,flip(plot_mean-plot_std)],colors0{3},'EdgeColor','none','FaceAlpha',.2), hold on
    plot(t_axis, plot_mean,colors1{3}), 
    ylabel('Theta power'), axis tight, ylim([0,6.3]), xlabel('Running speed (cm/s)')
    
    subplot(2,3,6)
    x = nanmean(data_cooling1');
    y1 = nanmean(data_cooling2');
    P1 = polyfit(x,y1,1); yfit = P1(1)*x+P1(2);
    plot(x, y1,['o',colors0{j}]), hold on
    plot(x,yfit,'-k');
    subplot(2,3,4)
    plot([3,4],P1,colors0{j})
end
subplot(2,3,4), xticks([1,2,3,4]), xticklabels({'Slope1','offset1','Slope2','offset2'})
subplot(2,3,5), axis tight, refline(1,0), title('Frequency'), xlabel('No cooling'), ylabel('Cooling')
subplot(2,3,6), axis tight, refline(1,0), title('Power'), xlabel('No cooling'), ylabel('Cooling')

%% Batch analysis running speed histograms

clear all, close all
MedialSeptum_Recordings

colors = {'r','b','g'};

datasets_MS10 = [58,59,60,61,62,66,71,72]; % 8 sessions. More? 182
datasets_MS12 = [77,78,79,80,81,82,83,84,86,179]; % 10 sessions.
datasets_MS13 = [90,91,92,93,94,173,174,175]; % 8 sessions. More? 176,178
datasets_MS14 = [108];
datasets_MS21 = [122,126,140,146,147,149,151,152,153,155,156,157,158,160,161]; % 15 sessions
datasets_MS22 = [123,127,139,145,162,163,164,165,166,167,168,169]; % 12 sessions

% New sessions (11-09-2019): 
% MS13: 173 OK,174 OK,175 OK,176 tracking missing,178,
% MS10: 181 no temperature?, 182 no temperature?
% MS14: ids 102 102 103 104 105 106 107 108
datasets = [datasets_MS10,datasets_MS12,datasets_MS13,datasets_MS21,datasets_MS22];
animalid2 = {datasets_MS10,datasets_MS12,datasets_MS13,datasets_MS21,datasets_MS22};
animal_names = {'MS10','MS12','MS13','MS21','MS22'};
animalid = [];
for i = 1:length(animalid2)
    animalid = [animalid,i*ones(1,length(animalid2{i}))];
end
bins_speed = [10:5:150];
bins_speed1 = bins_speed(1:end-1)+2.5;
sessions_speed = zeros(length(bins_speed1),length(datasets),3);

for k = 1:length(datasets)
    recording = recordings(datasets(k));
    disp(['Processing ', num2str(k),'/',num2str(length(datasets)),' recordingID = ' num2str(datasets(k)),', basename = ' recording.name])
    cd(fullfile(datapath, recording.animal_id, recording.name))
    
    % Behavior
    load([recording.name, '.animal.behavior.mat'])
    
    % Trials
    load([recording.name, '.trials.behavior.mat'])
    temp3 = find(trials.cooling==3,1);
    for j = 1:3
        if j == 1
            trials_state1 = find(trials.cooling==1);
            tmep1 = find(ismember(trials.trials2,trials_state1));
        elseif j == 2
            trials_state1 = find(trials.cooling==2 & trials.cooling<temp3(end));
            tmep1 = find(ismember(trials.trials2,trials_state1));
        else
            trials_state1 = find(trials.cooling==3);
            tmep1 = find(ismember(trials.trials2,trials_state1));
        end
        temp = histcounts(animal.speed(tmep1),bins_speed,'normalization','probability');
        sessions_speed(:,k,j) = temp;
    end
end

figure
subplot(2,1,1)
for k = 1:length(datasets)
    for j = 1:3
        plot(bins_speed1,sessions_speed(:,k,j),colors{j}), hold on
    end
end

subplot(2,1,2)
for j = 1:3
    color_in = colors{j};
    x_data = bins_speed1;
    y_data = sessions_speed(:,:,j)';
    plot_mean = mean(y_data)
    plot_std = sem(y_data)
    patch([x_data,flip(x_data)], [plot_mean+plot_std,flip(plot_mean-plot_std)],color_in,'EdgeColor','none','FaceAlpha',.3), hold on
    plot(bins_speed1,mean(sessions_speed(:,:,j),2),colors{j},'linewidth',2)
end

%% % Gamma band analysis
clear all, close all
MedialSeptum_Recordings
datasets_MS10 = [58,59,60,61,62,66,71,72]; % 8 sessions. More? 182
datasets_MS12 = [77,78,79,80,81,82,83,84,86,179]; % 10 sessions.
datasets_MS13 = [90,91,92,93,94,173,174,175]; % 8 sessions. More? 176,178
datasets_MS14 = [108];
datasets_MS21 = [122,126,140,146,147,149,151,152,153,155,156,157,158,160,161]; % 15 sessions
datasets_MS22 = [123,127,139,145,162,163,164,165,166,167,168,169]; % 12 sessions
datasets = [datasets_MS12,datasets_MS13,datasets_MS21,datasets_MS22];
t_offset = 200;
t_duration = 900;
t_duration2 = t_duration-1;
k = 1;
theta_all = {};

% Spectrogram settings
srLfp = 1250;
Fc2 = [50];
[b1,a1]=butter(3,Fc2*2/srLfp,'low');
downsample_n1 = 10;
sr_lfp2 = srLfp/downsample_n1;
freqlist = 5.5:0.1:10.5; % freqlist = 10.^(0.6021:0.01:1.0792);
wt_hippocampus = [];
k2 = 1;
% datasets = [126, 140, 139, 127, 78, 79, 80, 81, 92, 93]; % 62
figure(1)
for iSession=1:length(datasets)
    datasets(iSession)
    recording = recordings(datasets(iSession));
    disp(['Processing ', num2str(iSession),'/',num2str(length(datasets)),' recordingID = ' num2str(datasets(iSession)),', basename = ' recording.name])
    basepath = fullfile(datapath, recording.animal_id, recording.name);
    cd(basepath)
    theta.sr_freq = 10;
    InstantaneousTheta = calcInstantaneousTheta(recording);
    
    theta.time = InstantaneousTheta.signal_time;
    theta.freq = InstantaneousTheta.signal_freq{recording.ch_theta};
    
    % Temperature
    load([recording.name, '.temperature.timeseries.mat'])

%     % Cooling
%     load([recording.name, '.cooling.manipulation.mat'])
    
    if isempty(recording.cooling_onsets)
        figure,plot(temperature.time,temperature.temp), title(num2str(datasets(iSession)))
%         warndlg('give input')
%         return
    else
        figure(1)
        onsets = recording.cooling_onsets-t_offset;
        for i = 1:length(onsets)
            t_interval = round(onsets(i)*temperature.sr):round((onsets(i)+t_duration)*temperature.sr);
            idx1 = find(theta.time>onsets(i),1);
            idx2 = find(theta.time>onsets(i)+t_duration,1);
            theta_all.temperature(:,k) = interp1(temperature.time(t_interval)-onsets(i),temperature.temp(t_interval),[0:t_duration2]);
            theta_all.theta(:,k) = interp1(theta.time(idx1:idx2)-onsets(i),theta.freq(idx1:idx2),[0:t_duration2]);
            
            subplot(2,2,1), hold on
            plot(temperature.time(t_interval)-onsets(i),temperature.temp(t_interval))
            axis tight
            subplot(2,2,2), hold on
            plot(theta.time(idx1:idx2)-onsets(i),theta.freq(idx1:idx2))
            axis tight
            k = k + 1;
        end
    end
    subplot(2,2,3), cla
    plot_error_patch([0:t_duration2]-t_offset,theta_all.temperature','k'), grid on
    subplot(2,2,4), cla
    plot_error_patch([0:t_duration2]-t_offset,theta_all.theta','b'), grid on
        
    % Behavior
    load([recording.name, '.animal.behavior.mat'])
    
    %     % Trials
    %     load([recording.name, '.trials.behavior.mat'])
    
    %     [session, basename, basepath, clusteringpath] = db_set_path('session',recording.name);
    %     classification_DeepSuperficial(session)
    
    % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % 
    % Getting theta frequency from spectrogram (Theta frequency as a function of temperature)
    figure
    for i = 1:length(onsets)
        lfp_cooling = 0.195 * double(LoadBinary([recording.name,'.lfp'],'nChannels',recording.nChannels,'channels',recording.ch_theta,'precision','int16','frequency',srLfp,'start',onsets(i),'duration',t_duration));
        lfp_filt = filtfilt(b1,a1,lfp_cooling);
        
        % Spectrogram
        wt = spectrogram(downsample(lfp_filt,downsample_n1),sr_lfp2,sr_lfp2-1,freqlist,sr_lfp2);
        wt = [zeros(length(freqlist),floor(sr_lfp2/2)),abs(wt), zeros(length(freqlist),ceil(sr_lfp2/2))];
        wavelets_lfp = abs(wt);
        t_axis = [1:size(wavelets_lfp,2)]/sr_lfp2;
        
        speeds = interp1(animal.time, animal.speed, t_axis+onsets(i));
        temp = interp1(animal.time, animal.temperature, t_axis+onsets(i),'linear');
        
        idx = find(speeds < 20);
        wavelets_lfp2 = wavelets_lfp;
        wavelets_lfp2(:,idx) = nan;
        wt_hippocampus(:,:,k) = wavelets_lfp2;
        k2 = k2 + 1;
        wavelets_lfp(:,idx) = [];
        temp(idx) = [];
        speeds(idx) = [];
        subplot(length(onsets),1,i)
        imagesc([1:size(wavelets_lfp,2)]/sr_lfp2,freqlist,wavelets_lfp)
%         imagesc(time-t_offset,freqlist,wavelets_lfp), 
        title([num2str(datasets(iSession)),', cooling trial ' num2str(i)]), set(gca,'YDir','normal'), hold on
        plot([0,0],[freqlist(1),freqlist(end)],'w')
        t_interval = onsets(i)*temperature.sr:(onsets(i)+t_duration)*temperature.sr;
        plot([1:size(wavelets_lfp,2)]/sr_lfp2,temp/4+1,'w')
        plot([1:size(wavelets_lfp,2)]/sr_lfp2,(speeds-20)/60+5.5,'w')
%         plot(temperature.time(t_interval)-onsets(i)-t_offset,temperature.temp(t_interval)/4,'w')
        colorbar, clim([0,5000]),
    end
    drawnow
end

figure,
imagesc(t_axis-t_offset,freqlist,nanmean(wt_hippocampus,3)), title('Average'), set(gca,'YDir','normal'), hold on
plot([0,0],[freqlist(1),freqlist(end)],'w')
plot([0:t_duration2]-t_offset,nanmean(theta_all.temperature'/4),'w')
colorbar, clim([0,2000])


%% Spatial information for place cells across sessions
clear all, close all
datasets = [61,64,78,79,80,81,83,91,92,93,126,140,149,153,151,139,127,166]; % 62 63, 94,168

MedialSeptum_Recordings

bin_size = 5;
speed_th = 5;
Information_1_all = [];
Information_2_all = [];
Sparsity_all = [];
colors = {'r','b','g'};
kk = 1;
% id = 93;
for j = 1:length(datasets)
    id = datasets(j);
    recording = recordings(id);
    disp(['Processing ', num2str(j),'/',num2str(length(datasets)),' recordingID = ' num2str(datasets(j)),', basename = ' recording.name])
    cd(fullfile(datapath, recording.animal_id, recording.name))
    % [session, basename, basepath, clusteringpath] = db_set_path('session',recording.name);
    
    temperature  = loadStruct('temperature','timeseries','recording',recording);
    cooling      = loadStruct('cooling','manipulation','recording',recording);
    animal       = loadStruct('animal','behavior','recording',recording);
    trials       = loadStruct('trials','behavior','recording',recording);
    spatial_bins = [animal.pos_linearized_limits(1):bin_size:animal.pos_linearized_limits(2)];
    
    spikes       = loadStruct('spikes','cellinfo','recording',recording);
    cell_metrics = loadStruct('cell_metrics','cellinfo','recording',recording);
    pyramical_cells = find(contains(cell_metrics.putativeCellType,'Pyramidal') & cell_metrics.placeCell);
    
    for i = pyramical_cells
        for k = 1:3
            indexes1 = find(animal.state == k &animal.speed > speed_th);
            TimeSpent  = histcounts(animal.pos_linearized(indexes1),spatial_bins)/animal.sr; % 1- TimeSpent = Time Spent (s) per bin
            
            indexes2 = find(spikes.state{i} == k & spikes.speed{i} > speed_th);
            nSpikes = histcounts(spikes.pos_linearized{i}(indexes2),spatial_bins); % 2- nSpikes = Spikes (count) per bin
            
            Rate_Map = nSpikes ./ TimeSpent; % 3- Rate_Map = nSpikes / TimeSpent
            
            [Information_1,Information_2,Sparsity,Coefficient,Selectivity] = PlaceCellInfo(Rate_Map, nSpikes, TimeSpent);
            Information_1_all(kk,k) = Information_1;
            Information_2_all(kk,k) = Information_2;
            Sparsity_all(kk,k) = Sparsity;
            kk = kk + 1;
        end
    end
    disp(' ')
end
disp(['Processed ',num2str(kk-1),' place cells'])

figure
bins{1} = [0.00001:0.2:6];
bins{2} = [0.00001:0.5:20];
bins{3} = [0.00001:0.05:1];
for k = 1:3
    subplot(1,3,1),
    temp1 = histcounts(Information_1_all(:,k),bins{1},'normalization','probability');
    plot(bins{1}(1:end-1),temp1,colors{k}), hold on, title('Information: bits per spike'), axis tight, ylim([0,0.1])
    subplot(1,3,2),
    temp2 = histcounts(Information_2_all(:,k),bins{2},'normalization','probability');
    plot(bins{2}(1:end-1),temp2,colors{k}), hold on, title('Information: bits per second'), axis tight, ylim([0,0.1])
    subplot(1,3,3),
    temp3 = histcounts(Sparsity_all(:,k),bins{3},'normalization','probability');
    plot(bins{3}(1:end-1),temp3,colors{k}), hold on, title('Sparsity'), axis tight, ylim([0,0.1])
end
subplot(1,3,1),
idx = find(sum(isnan(Information_1_all)')==0);
[p1,h1] = signrank(Information_1_all(idx,1),Information_1_all(idx,2));
[p2,h2] = signrank(Information_1_all(idx,1),Information_1_all(idx,3));
[p3,h3] = signrank(Information_1_all(idx,2),Information_1_all(idx,3));
text(0.1,0.09,[' (1,2) ',num2str(p1),',  ',num2str(h1)]);
text(0.1,0.08,[' (1,3) ',num2str(p2),'  ,',num2str(h2)]);
text(0.1,0.07,[' (2,3) ',num2str(p3),',  ',num2str(h3)]);
subplot(1,3,2),
idx = find(sum(isnan(Information_2_all)')==0);
[p1,h1] = signrank(Information_2_all(idx,1),Information_2_all(idx,2));
[p2,h2] = signrank(Information_2_all(idx,1),Information_2_all(idx,3));
[p3,h3] = signrank(Information_2_all(idx,2),Information_2_all(idx,3));
text(0.1,0.09,[' (1,2) ',num2str(p1),',  ',num2str(h1)]);
text(0.1,0.08,[' (1,3) ',num2str(p2),'  ,',num2str(h2)]);
text(0.1,0.07,[' (2,3) ',num2str(p3),',  ',num2str(h3)]);
subplot(1,3,3),
idx = find(sum(isnan(Sparsity_all)')==0);
[p1,h1] = signrank(Sparsity_all(idx,1),Sparsity_all(idx,2));
[p2,h2] = signrank(Sparsity_all(idx,1),Sparsity_all(idx,3));
[p3,h3] = signrank(Sparsity_all(idx,2),Sparsity_all(idx,3));
text(0.1,0.09,[' (1,2) ',num2str(p1),',  ',num2str(h1)]);
text(0.1,0.08,[' (1,3) ',num2str(p2),'  ,',num2str(h2)]);
text(0.1,0.07,[' (2,3) ',num2str(p3),',  ',num2str(h3)]);

%% % % % % % % % % % % % % % % % % % % % % % % % % % % %
% Effect on behavior 

clear all, close all
MedialSeptum_Recordings

colors = {'r','b','g'};
datasets_MS10 = [58,59,60,61,62,66,71,72]; % 8 sessions. More? 182
datasets_MS12 = [77,78,79,80,81,82,83,84,86,179]; % 10 sessions.
datasets_MS13 = [90,91,92,93,94,173,174,175]; % 8 sessions. More? 176,178
datasets_MS14 = [108];
datasets_MS21 = [122,126,140,146,147,149,151,152,153,155,156,157,158,160,161]; % 15 sessions
datasets_MS22 = [123,127,139,145,162,163,164,165,166,167,168,169]; % 12 sessions

% New sessions (11-09-2019): 
% MS13: 173 OK,174 OK,175 OK,176 tracking missing,178,
% MS10: 181 no temperature?, 182 no temperature?
% MS14: ids 102 102 103 104 105 106 107 108
datasets = [datasets_MS10,datasets_MS12,datasets_MS13,datasets_MS21,datasets_MS22];
animalid2 = {datasets_MS10,datasets_MS12,datasets_MS13,datasets_MS21,datasets_MS22};
% datasets = [datasets_MS10];
% animalid2 = {datasets_MS10};
animal_names = {'MS10','MS12','MS13','MS21','MS22'};
animalid = [];
for i = 1:length(animalid2)
    animalid = [animalid,i*ones(1,length(animalid2{i}))];
end

% MS14: ids 102 102 103 104 105 106 107 108
trials_speed(1,5).cooling = [];
trials_speed(1,5).nocooling = [];
trials_batch = {};
intertrial_intervals{1} = [];
intertrial_intervals{2} = [];
intertrial_intervals{3} = [];
trials_duration{1} = [];
trials_duration{2} = [];
trials_duration{3} = [];
interval_bins = 10.^([-0.5:0.1:1.2]);
interval_bins_center = 10.^([-0.55:0.1:1.1]);
for k = 1:length(datasets)
    recording = recordings(datasets(k));
    disp(['Processing ', num2str(k),'/',num2str(length(datasets)),' recordingID = ' num2str(datasets(k)),', basename = ' recording.name])
    cd(fullfile(datapath, recording.animal_id, recording.name))
%     temperature  = loadStruct('temperature','timeseries','recording',recording);
%     cooling      = loadStruct('cooling','manipulation','recording',recording);
    animal       = loadStruct('animal','behavior','recording',recording);
    trials = loadStruct('trials','behavior','recording',recording);
    
    figure
    for i = 1:3
        idx = find(trials.cooling==i);
        start = trials.start(idx);
        stop = trials.end(idx);
        start = start(2:end)/animal.sr;
        stop = stop(1:end-1)/120;
        intertrial_intervals{i} = [intertrial_intervals{i},start-stop];
        trials_duration{i} =  (trials.end(idx)- trials.start(idx))/animal.sr;
        subplot(2,1,1)
        plot(idx(1:end-1),start-stop,['.-',colors{i}]), hold on, ylim([0,20]), title(recording.name)
        plot(idx,(trials.end(idx)- trials.start(idx))/animal.sr,'k')
        subplot(2,1,2)
        temp = histcounts(start-stop,interval_bins,'normalization','probability');
        plot(interval_bins_center,temp,['.-',colors{i}]), hold on, 
        xlabel('Time (sec)')
    end
end

figure
for i = 1:3
    subplot(1,2,1)
    temp = histcounts(intertrial_intervals{i},interval_bins,'normalization','probability');
    plot(interval_bins_center,temp,['.-',colors{i}]), hold on
    xlabel('Time (sec)'),title('Intertrial intervals')
    subplot(1,2,2)
    temp = histcounts(trials_duration{i},interval_bins,'normalization','probability');
    plot(interval_bins_center,temp,['.-',colors{i}]), hold on
    xlabel('Time (sec)'),title('Trial duration')
end
legend({'Pre','Cooling','Post'})
