MedialSeptum_Recordings;
id = 140
% MS10: 61, 62, 63, 64 % gamma id: 62
% MS12: 78 (Peter_MS12_170714_122034_concat), 79 (Peter_MS12_170715_111545_concat), 80 (Peter_MS12_170716_172307_concat), 81 (Peter_MS12_170717_111614_concat),
% MS13: 92 (Peter_MS13_171129_105507_concat), 93 (Peter_MS13_171130_121758_concat)
% MS14: 
% MS18: 
% MS21: 126 (Peter_MS21_180629_110332_concat), 140 (Peter_MS21_180627_143449_concat), 143 (control: Peter_MS21_180719_155941_concat)
% MS22: 139 (Peter_MS22_180628_120341_concat), 127 (Peter_MS22_180629_110319_concat), 144 (control: Peter_MS22_180719_122813_concat)
% id = 126 % 62

% rate metrics completed: 126,140,143,139,127,

recording = recordings(id);
% if ~isempty(recording.dataroot)
%     datapath = recording.dataroot;
% end

[session, basename, basepath, clusteringpath] = db_set_path('session',recording.name);
if isempty(session.epochs.duration) | session.epochs.duration == 0
    session = db_update_session(session);
end

cd(fullfile(datapath, recording.animal_id, recording.name))

% % loading theta, spikes and trials
InstantaneousTheta = calcInstantaneousTheta(recording);
spikes = loadSpikes('clusteringpath',recording.SpikeSorting.path,'clusteringformat',recording.SpikeSorting.method,'basename',recording.name);

animal = loadStruct('animal','behavior','recording',recording);
trials = loadStruct('trials','behavior','recording',recording);

% Cell assembly analysis
% Determine the distance being encoded in each theta sequence
PyramidalIndexes3 = get_CellMetrics('session',recording.name,'putativeCellType',{'Pyramidal'});

[~,~,PyramidalIndexes1] = intersect(find(cellfun(@isempty,recording.SpikeSorting.center_arm_placecells)==0),spikes.cluID);
PyramidalIndexes = PyramidalIndexes1';
[~,~,PyramidalIndexes2] = intersect(find(cellfun(@isempty,recording.SpikeSorting.polar_theta_placecells)==0),spikes.cluID);
PyramidalIndexes = PyramidalIndexes2';
%%
n = length(PyramidalIndexes);
% colormap2 = [[1:n]/n;[n:-1:1]/n;[1:n]/n]';
colormap2 = jet(n);
bins_position = [0:5:350]; 
pos_interval2 = [0,85];
pos_interval2 = [85,215];
pos_interval2 = [215,345];
trialsStat = 2; % 1 for left and 2 for right
figure
bins_position1 = [0:1:350];
for jj = 1:3
    trials_state = intersect(find(trials.cooling==jj),setdiff([1:length(trials.cooling)],trials.error));
    idx = find(ismember(trials.trials2,trials_state) & animal.pos_linearized>0);
    N_animal = histcounts(animal.pos_linearized(idx),bins_position1);
    N_animal = N_animal/animal.sr;
    subplot(3,1,jj)
    for ii = 1:length(PyramidalIndexes)
%         i = pos_sorted(ii);
        i = PyramidalIndexes(pos_sorted(ii));
        idx2 = find(ismember(spikes.trials{i},trials_state) & spikes.pos_linearized{i}>0);
        N = histcounts(spikes.pos_linearized{i}(idx2),bins_position1);
        plot(bins_position1(1:end-1),nanconv(N./N_animal,gausswin(20)'/sum(gausswin(20))),'-','color',colormap2(ii,:)), hold on
    end
    plot([pos_interval2;pos_interval2],[0,80;0,80]','k'), axis tight
end

trials_state = intersect(find(trials.cooling<10),setdiff([1:length(trials.cooling)],trials.error));
idx = find(ismember(trials.trials2,trials_state) & animal.pos_linearized>0);
N_animal = histcounts(animal.pos_linearized(idx),bins_position);
N_animal = N_animal/animal.sr;
%     subplot(3,1,jj)
figure
k = 1;
placefields = nan(length(bins_position)-1,length(PyramidalIndexes));

for ii = PyramidalIndexes
    idx2 = find(ismember(spikes.trials{ii},trials_state) & spikes.pos_linearized{ii}>0);
    N = histcounts(spikes.pos_linearized{ii}(idx2),bins_position);
    plot(bins_position(1:end-1),N./N_animal), hold on
    placefields(:,k) = N./N_animal;
    k = k + 1;
end


pos_interval = [find( bins_position>=pos_interval2(1),1),find( bins_position>pos_interval2(2),1)-1];

test = cumsum(placefields(pos_interval(1):pos_interval(2),:)); test = test./max(test);
idx3 = [];
for ii = 1:length(PyramidalIndexes)
    idx = find(test(:,ii) > 0.5,1);
    if ~isempty(idx)
        idx3(ii) = idx;
    else
        idx3(ii) = 0;
    end
end
[~,pos_sorted] = sort(idx3);

figure, 
subplot(2,1,1)
for ii = 1:length(PyramidalIndexes)
    i = pos_sorted(ii);
    plot(bins_position(pos_interval(1):pos_interval(2)),test(:,i),'color',colormap2(ii,:)), hold on
end
title('Cumsum place fields'), xlabel('Position (cm)')
subplot(2,1,2)
for ii = 1:length(PyramidalIndexes)
    i = pos_sorted(ii);
    plot(bins_position(1:end-1),placefields(:,i),'color',colormap2(ii,:)), hold on
    plot([pos_interval2;pos_interval2],[0,80;0,80]','k')
end
title('Place fields'), xlabel('Position (cm)')

trials_state2 = intersect(find(trials.stat<10 & trials.stat==trialsStat),setdiff([1:length(trials.stat)],trials.error));
temp = [];
temp2 = [];
for j = 1:length(trials_state2)
    temp(j) = animal.time(find(trials.trials2 == trials_state2(j) & animal.pos_linearized > mean(pos_interval2),1));
    temp2(j) = InstantaneousTheta.signal_phase2{1}(round(temp(j)*(recording.sr/16)));
end
figure
for ii = 1:length(PyramidalIndexes)
    i = pos_sorted(ii);
    i = PyramidalIndexes(i);
    idx2 = find(ismember(spikes.trials{i},trials_state2) & spikes.pos_linearized{i}>pos_interval2(1) & spikes.pos_linearized{i}<pos_interval2(2));
    if ~isempty(idx2)
        subplot(2,1,1)
        plot(spikes.pos_linearized{i}(idx2),spikes.trials{i}(idx2),'.','color',colormap2(ii,:)), hold on
        subplot(2,1,2)
        for j = 1:length(trials_state2)
            idx3 = find(spikes.trials{i}==trials_state2(j) & spikes.pos_linearized{i}>pos_interval2(1) & spikes.pos_linearized{i}<pos_interval2(2));
            find(spikes.pos_linearized{i}(idx3)> mean(pos_interval2),1)
            if ~isempty(idx3)
                offset = ceil(temp2(j)/(2*pi));
%                 offset = floor(spikes.theta_phase2{i}(idx3(1))/(2*pi));
                plot(spikes.theta_phase2{i}(idx3)/(2*pi)-offset,j*ones(1,length(idx3)),'.','color',colormap2(ii,:)), hold on
            end
        end
    end
end
subplot(2,1,1)
xlabel('Position (cm)'), ylabel('Trials')
subplot(2,1,2)
plot([-5:5;-5:5],[0,j]'*ones(1,11),'-','color',[0.6,0.6,0.6]), axis tight
trial_cooling_onset = find(trials_state2>find(trials.cooling==2,1),1);
trial_cooling_offset = find(trials_state2>find(trials.cooling==0,1),1);
plot([-10,10],[trial_cooling_onset,trial_cooling_onset],'b')
plot([-10,10],[trial_cooling_offset,trial_cooling_offset],'b')
xlabel('Theta cycles'), ylabel('Trials')
xlim([-5,5])

