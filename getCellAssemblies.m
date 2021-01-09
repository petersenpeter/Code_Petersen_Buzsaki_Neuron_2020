function cell_assemblies = getCellAssemblies(id)
% clear all, close all
MedialSeptum_Recordings
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
load('trials.mat')

% Cell assembly analysis
% Determine the number of cells activated in each theta sequence

% cell_metrics = LoadCellMetricBatch('sessions',{recording.name});
% PyramidalIndexes = find(contains(cell_metrics.putativeCellType,'Pyramidal'));

PyramidalIndexes = get_CellMetrics('session',recording.name,'putativeCellType',{'Pyramidal'});
colors = {'r','b','g'};
% bins = [0.01:0.01:1];
bins = [0:1:length(PyramidalIndexes)];
spike_length3 = {};
cell_assemblies = {};
figure, subplot(2,1,1)
for jj = 1:3
    kkk = 1;
    CellAssemblies = [];
    trials_state = intersect(find(trials.cooling==jj),setdiff([1:length(trials.cooling)],trials.error));
    spike_length2 = [];
    for j = 1:length(trials_state)
        spikes_temp = [];
        spikes_temp2 = [];
        spike_length = [];
        for i = 1:length(PyramidalIndexes)
            ii = PyramidalIndexes(i);
            spikes_temp{i} = ceil(spikes.theta_phase2{ii}(find(spikes.trials{ii}==trials_state(j) & spikes.speed{ii}>20))/(2*pi));
            if length(spikes_temp{i})==1
                spikes_temp{i} = [];
            end
            spikes_temp2{i} = i*ones(1,length(spikes_temp{i}));
            spike_length(i) = length(find(spikes.trials{ii}==trials_state(j) & spikes.speed{ii}>20));
        end
        spike_length2(j) = sum(spike_length>0);
        all_spikes = horzcat(spikes_temp{:});
        all_spikes2 = horzcat(spikes_temp2{:});
        for k = min(horzcat(spikes_temp{:})):max(horzcat(spikes_temp{:}))
            CellAssemblies(kkk) = length(unique(all_spikes2(find(all_spikes==k)))); % /length(PyramidalIndexes)
            kkk = kkk + 1;
        end
    end
    spike_length3{jj} = spike_length2;
    [N,edges] = histcounts(CellAssemblies,bins,'Normalization','cdf');
    plot((bins(1:end-1)+0.01)/mean(spike_length3{jj}), N, colors{jj}), hold on
    cell_assemblies{jj}.xbins = (bins(1:end-1)+0.01)/mean(spike_length3{jj});
    cell_assemblies{jj}.hist = N;
    cell_assemblies{jj}.session = recording.name;
    cell_assemblies{jj}.spike_length = spike_length3{jj};
    cell_assemblies{jj}.CellAssemblies = CellAssemblies;
end

xlim([0,0.8]), legend({'Pre','Cooling','Post'}), % set(gca, 'XScale', 'log')
xlabel('Fraction of cells being active'), ylabel('Fraction of theta cycles'), title(recording.name,'interpreter','none')
subplot(2,1,2)
test332423 = [0,cumsum(cellfun(@length, spike_length3))];
for j = 1:3
    plot([1:length(spike_length3{j})]+test332423(j), spike_length3{j},'color',colors{j}), hold on 
end

[p,h] = ranksum(cell_assemblies{1}.CellAssemblies,cell_assemblies{2}.CellAssemblies);
cell_assemblies{1}.p = p;
cell_assemblies{1}.h = h;
[p,h] = ranksum(cell_assemblies{2}.CellAssemblies,cell_assemblies{3}.CellAssemblies);
cell_assemblies{2}.p = p;
cell_assemblies{2}.h = h;
[p,h] = ranksum(cell_assemblies{1}.CellAssemblies,cell_assemblies{3}.CellAssemblies);
cell_assemblies{3}.p = p;
cell_assemblies{3}.h = h;
print(gcf, ['CellAssemblies_' recording.name],'-dpdf')
save('cell_assemblies.mat','cell_assemblies')
