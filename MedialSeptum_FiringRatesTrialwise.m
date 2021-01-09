%% % Rates of pyramidal cells and interneuros

close all, clear all
sessionNames = {'Peter_MS12_170714_122034_concat','Peter_MS12_170715_111545_concat','Peter_MS12_170716_172307_concat','Peter_MS12_170717_111614_concat','Peter_MS12_170719_095305_concat'...
    'Peter_MS12_170717_111614_concat','Peter_MS13_171129_105507_concat','Peter_MS13_171130_121758_concat','Peter_MS13_171128_113924_concat','Peter_MS13_171201_130527_concat',...
    'Peter_MS21_180629_110332_concat','Peter_MS21_180627_143449_concat','Peter_MS21_180625_153927_concat','Peter_MS21_180712_103200_concat','Peter_MS21_180628_155921_concat',...
    'Peter_MS22_180628_120341_concat','Peter_MS22_180629_110319_concat','Peter_MS22_180720_110055_concat','Peter_MS22_180711_112912_concat'};

batchName = 'All';
colors = {'r','b','g'};
colors1 = {'r','b','g','m','c','k'};
celltypes = {'Interneuron','Pyramidal'};
states = {'Pre','Cooling','Post'};
speed_threshold = 10;
boundaries = [85, 130+85];
xlimits = [30,137];
kk1 = 0;
kk2 = 0;
spikesTrialswise1 = [];
spikesTrialswise2 = [];
spikesTrialswise1_theta = [];
spikesTrialswise2_theta = [];
temperatureTrialswise = [];
thetaTrialswise = [];
for k = 1:length(sessionNames)
    disp(['*** Processing sessions: ', num2str(k),'/', num2str(length(sessionNames)),' sessions: ' sessionNames{k}])
    [session, basename, basepath, clusteringpath] = db_set_session('sessionName',sessionNames{k});
    
    trials = loadStruct('trials','behavior','session',session);
    animal = loadStruct('animal','behavior','session',session);
    
    % Theta
    theta.channel =  session.channelTags.Theta.channels;
    InstantaneousTheta = calcInstantaneousTheta2(session);
    if length(InstantaneousTheta.timestamps)~= length(InstantaneousTheta.ThetaInstantFreq{theta.channel})
        InstantaneousTheta = calcInstantaneousTheta2(session,'forceReload',true);
    end
    
    % spikes
    spikes = loadSpikes('session',session);
    cell_metrics = loadStruct('cell_metrics','cellinfo','session',session);
    cell_metrics_idx1 =  get_CellMetrics('cell_metrics',cell_metrics,'putativeCellType',{celltypes{1}});
    cell_metrics_idx2 =  get_CellMetrics('cell_metrics',cell_metrics,'putativeCellType',{celltypes{2}});
    
    trial_offset = 65-find(trials.cooling==2,1);
    if ~isempty(intersect(find(trials.cooling==3),find(trials.cooling==2)-1))
        trial_end = intersect(find(trials.cooling==3),find(trials.cooling==2)-1);
    else
        trial_end = length(trials.cooling);
    end
    
    for iTrial = 1:trial_end
        animal_trials = sum(animal.speed > speed_threshold & trials.trials2 == iTrial)/animal.sr;
        temperatureTrialswise(k,iTrial+trial_offset) = nanmean(animal.temperature(animal.speed > speed_threshold & trials.trials2 == iTrial));
        thetaTrialswise(k,iTrial+trial_offset) = nanmean(interp1(InstantaneousTheta.timestamps,InstantaneousTheta.ThetaInstantFreq{theta.channel},animal.time(animal.speed > speed_threshold & trials.trials2 == iTrial)));
%         trials_batch.theta_power{k}(iTrial) = nanmean(interp1(theta.time,theta.power,animal.time(idx)));

        for iCell = 1:length(cell_metrics_idx1)
            spikesTrialswise1(iCell+kk1,iTrial+trial_offset) = sum(spikes.trials{cell_metrics_idx1(iCell)} == iTrial & spikes.speed{cell_metrics_idx1(iCell)} > speed_threshold)/animal_trials;
            spikesTrialswise1_theta(iCell+kk1,iTrial+trial_offset) = sum(spikes.trials{cell_metrics_idx1(iCell)} == iTrial & spikes.speed{cell_metrics_idx1(iCell)} > speed_threshold)/animal_trials/thetaTrialswise(k,iTrial+trial_offset);
        end
        for iCell = 1:length(cell_metrics_idx2)
            spikesTrialswise2(iCell+kk2,iTrial+trial_offset) = sum(spikes.trials{cell_metrics_idx2(iCell)} == iTrial & spikes.speed{cell_metrics_idx2(iCell)} > speed_threshold)/animal_trials;
            spikesTrialswise2_theta(iCell+kk2,iTrial+trial_offset) = sum(spikes.trials{cell_metrics_idx2(iCell)} == iTrial & spikes.speed{cell_metrics_idx2(iCell)} > speed_threshold)/animal_trials/thetaTrialswise(k,iTrial+trial_offset);
        end
    end
    kk1 = length(cell_metrics_idx1) + kk1;
    kk2 = length(cell_metrics_idx2) + kk2;
end
spikesTrialswise1(spikesTrialswise1==0) = nan;
spikesTrialswise2(spikesTrialswise2==0) = nan;
spikesTrialswise1_theta(spikesTrialswise1==0) = nan;
spikesTrialswise2_theta(spikesTrialswise2==0) = nan;
temperatureTrialswise(temperatureTrialswise==0) = nan;

%%
figure, 
x_trials = [1:size(temperatureTrialswise,2)]-64;
xlimits1 = x_trials(xlimits);

subplot(3,2,1)
plot(x_trials,spikesTrialswise1','.'), title(celltypes{1}),xlim(xlimits1)
subplot(3,2,2)
plot(x_trials,spikesTrialswise2','.'), title(celltypes{2}),xlim(xlimits1)
subplot(3,2,3)
errorbarPatch(x_trials,nanmean(spikesTrialswise1),nansem(spikesTrialswise1),'b'), title(celltypes{1}),xlim(xlimits1)
subplot(3,2,4)
errorbarPatch(x_trials,nanmean(spikesTrialswise2),nansem(spikesTrialswise2),'r'), title(celltypes{2}),xlim(xlimits1)
subplot(3,2,5)
errorbarPatch(x_trials,nanmean(temperatureTrialswise),nansem(temperatureTrialswise),'k'),xlim(xlimits1)
subplot(3,2,6)
errorbarPatch(x_trials,nanmean(temperatureTrialswise),nansem(temperatureTrialswise),'k')
xlim(xlimits1)

figure, 
subplot(3,2,1)
plot(x_trials,spikesTrialswise1_theta','.'), title(celltypes{1}),xlim(xlimits1)
subplot(3,2,2)
plot(x_trials,spikesTrialswise2_theta','.'), title(celltypes{2}),xlim(xlimits1)
subplot(3,2,3)
errorbarPatch(x_trials,nanmean(spikesTrialswise1_theta),nansem(spikesTrialswise1_theta),'b'), title(celltypes{1}),xlim(xlimits1)
subplot(3,2,4)
errorbarPatch(x_trials,nanmean(spikesTrialswise2_theta),nansem(spikesTrialswise2_theta),'r'), title(celltypes{2}),xlim(xlimits1)
subplot(3,2,5)
errorbarPatch(x_trials,nanmean(temperatureTrialswise),nansem(temperatureTrialswise),'k'),xlim(xlimits1)
subplot(3,2,6)
errorbarPatch(x_trials,nanmean(temperatureTrialswise),nansem(temperatureTrialswise),'k')
xlim(xlimits1)

figure
plot(x_trials,nanzscore(nanmean(temperatureTrialswise)),'k'), hold on
plot(x_trials,nanzscore(nanmean(spikesTrialswise1)),'b')
plot(x_trials,nanzscore(nanmean(spikesTrialswise2)),'r')
xlim(xlimits1), legend(['Temperature',celltypes])

%%
% [126,127,140,93,78,81,168,166,151,149]
% MS12: 78 (Peter_MS12_170714_122034_concat) OK  
%       81 (Peter_MS12_170717_111614_concat) OK 
%       93 (Peter_MS13_171130_121758_concat) OK 
% MS21: 126 (Peter_MS21_180629_110332_concat) OK 
%       140 (Peter_MS21_180627_143449_concat) OK 
%       149 (Peter_MS21_180625_153927_concat) OK 
%       151 (Peter_MS21_180628_155921_concat) OK 
%       127 (Peter_MS22_180629_110319_concat) OK 
%       168 (Peter_MS22_180720_110055_concat) OK 
%       166 (Peter_MS22_180711_112912_concat) OK 

sessionNames = {'Peter_MS12_170714_122034_concat','Peter_MS12_170717_111614_concat','Peter_MS13_171130_121758_concat','Peter_MS21_180629_110332_concat','Peter_MS21_180627_143449_concat','Peter_MS21_180625_153927_concat','Peter_MS21_180628_155921_concat','Peter_MS22_180629_110319_concat','Peter_MS22_180720_110055_concat','Peter_MS22_180711_112912_concat'}
batchName = 'All';
colors = {'r','b','g'};
colors1 = {'r','b','g','m','c','k'};
states = {'Pre','Cooling','Post'};
speed_threshold = 30;
boundaries = [85, 130+85];
xlimits = [30,137];
kk1 = 0;
kk2 = 0;
speed_noCooling_sessions = [];
speed_Cooling_sessions = [];
for k = 1:length(sessionNames)
    disp(['*** Processing sessions: ', num2str(k),'/', num2str(length(sessionNames)),' sessions: ' sessionNames{k}])
    [session, basename, basepath, clusteringpath] = db_set_session('sessionName',sessionNames{k});
    
    trials = loadStruct('trials','behavior','session',session);
    animal = loadStruct('animal','behavior','session',session);
    
    if ~isempty(intersect(find(trials.cooling==3),find(trials.cooling==2)-1))
        trial_end = intersect(find(trials.cooling==3),find(trials.cooling==2)-1);
    else
        trial_end = length(trials.cooling);
    end
    idx_noCooling = find(trials.cooling(1:trial_end)==1 | trials.cooling(1:trial_end)==3);
    idx_Cooling = find(trials.cooling(1:trial_end)==2);
    speed_noCooling = animal.speed(ismember(trials.trials2,idx_noCooling));
    speed_Cooling = animal.speed(ismember(trials.trials2,idx_Cooling));
    speed_noCooling(speed_noCooling<speed_threshold) = nan;
    speed_Cooling(speed_Cooling<speed_threshold) = nan;
    speed_noCooling_sessions(k) = nanmean(speed_noCooling);
    speed_Cooling_sessions(k) = nanmean(speed_Cooling);
end
figure,
plot(speed_noCooling_sessions,'r'), hold on
plot(speed_Cooling_sessions,'b')
mean(speed_noCooling_sessions)
std(speed_noCooling_sessions)
mean(speed_Cooling_sessions)
std(speed_Cooling_sessions)
