function cell_assembliesSpatialCoverage = getCellAssembliesSpatailCoverage2(id)
% clear all, close all
MedialSeptum_Recordings;
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
PyramidalIndexes = get_CellMetrics('session',recording.name,'putativeCellType',{'Pyramidal'});

spikes.numcells = length(spikes.UID);
for cc = 1:spikes.numcells
    groups{cc}=spikes.UID(cc).*ones(size(spikes.times{cc}));
end

colors = {'r','b','g'};
states = {'Pre','Cooling','Post'};
% bins = [0.01:0.01:1];
bins = [0:1:length(PyramidalIndexes)];
spike_length3 = {};
cell_assembliesSpatialCoverage = {};
bins_position = [0:20:350];
bins_speed = [0:5:150];
bins_temporal = [0:4:200];
bins_phase = [0:0.1:2*pi];
bins_assembly_size = [0:0.005:0.3];
step_size = 0.003;
bins_speed2 = [0.05:step_size:0.2];

figure('position',[50,50,1200,1000])
for jj = 1:3
    p1 = [];
    p2 = [];
    position = [];
    speed = [];
    spike_count = [];
    assembly_size = [];
    
    kkk = 1;
    CellAssemblies = [];
    x_data = [];
    y_data = [];
    
    trials_state = intersect(find(trials.cooling==jj),setdiff([1:length(trials.cooling)],trials.error));
    allspikes_times = cat(1,spikes.times{PyramidalIndexes});
    allspikes_groups = cat(1,groups{PyramidalIndexes});
    allspikes_trials = cat(1,spikes.trials{PyramidalIndexes});
    allspikes_theta2 = cat(2,spikes.theta_phase2{PyramidalIndexes})';
    allspikes_theta = cat(2,spikes.theta_phase{PyramidalIndexes})';
    allspikes_pos_linearized = cat(1,spikes.pos_linearized{PyramidalIndexes});
    allspikes_speed = cat(1,spikes.speed{PyramidalIndexes});
    
    [allspikes_times,sortidx] = sort(allspikes_times);
    allspikes_groups = allspikes_groups(sortidx);
    allspikes_trials = allspikes_trials(sortidx);
    allspikes_theta2 = allspikes_theta2(sortidx);
    allspikes_theta = allspikes_theta(sortidx);
    allspikes_pos_linearized = allspikes_pos_linearized(sortidx);
    allspikes_speed = allspikes_speed(sortidx);
    
    ia = find(ismember(allspikes_trials,trials_state) & allspikes_pos_linearized>0);
    allspikes_times = allspikes_times(ia);
    allspikes_groups = allspikes_groups(ia);
    allspikes_trials = allspikes_trials(ia);
    allspikes_theta2 = allspikes_theta2(ia);
    allspikes_theta = allspikes_theta(ia);
    allspikes_pos_linearized = allspikes_pos_linearized(ia);
    allspikes_speed = allspikes_speed(ia);

    slope_temporal_phase = [];
    slope_position_phase = [];
    slope_position_temporal = [];
    slope_position_temporal = [];
    temporal_offset_all = [];
    phase_offset_all = [];

    hold on
    for j = 1:ceil(max(allspikes_theta2)/(2*pi))
        idx = find(allspikes_theta2 > (j-1)*2*pi-pi & allspikes_theta2<j*2*pi-pi);
        if idx>5 & max(allspikes_pos_linearized(idx))-min(allspikes_pos_linearized(idx)) < 50
            temp = unique(allspikes_groups(idx));
            unit_time = [];
            unit_position = [];
            unit_speed = [];
            unit_phase2 = [];
            if length(temp)>1
                for j_unit = 1:length(temp)
                    idx2 = find(spikes.theta_phase2{temp(j_unit)}>(j-1)*2*pi-pi  &  spikes.theta_phase2{temp(j_unit)}<j*2*pi-pi);
                    unit_time(j_unit) = mean(spikes.times{temp(j_unit)}(idx2));
                    unit_position(j_unit) = mean(spikes.pos_linearized{temp(j_unit)}(idx2));
                    unit_speed(j_unit) = mean(spikes.speed{temp(j_unit)}(idx2));
                    unit_phase2(j_unit) = mean(spikes.theta_phase{temp(j_unit)}(idx2));
                end
                
                pairs = nchoosek([1:length(temp)], 2);
                temporal_offset = [];
                position_offset = [];
                speed_offset = [];
                phase_offset = [];
                position_offset3 = [];
                for j_pairs = 1:size(pairs,1)
                    temporal_offset(j_pairs) = abs(unit_time(pairs(j_pairs,2)) - unit_time(pairs(j_pairs,1)));
                    position_offset(j_pairs) = abs(unit_position(pairs(j_pairs,2)) - unit_position(pairs(j_pairs,1)));
                    speed_offset(j_pairs) = mean([unit_speed(pairs(j_pairs,2)), unit_speed(pairs(j_pairs,1))]);
                    position_offset3(j_pairs) = mean([unit_position(pairs(j_pairs,2)), unit_position(pairs(j_pairs,1))]);
                    phase_offset(j_pairs) = abs(unit_phase2(pairs(j_pairs,2)) - unit_phase2(pairs(j_pairs,1)));
                end
                slope_temporal_phase = [slope_temporal_phase, mean(temporal_offset./phase_offset)*2*pi];
                slope_position_phase = [slope_position_phase, mean(position_offset./phase_offset)*2*pi];
                slope_position_temporal = [slope_position_temporal, mean(position_offset./temporal_offset)];
                temporal_offset_all = [temporal_offset_all,temporal_offset];
                phase_offset_all = [phase_offset_all,phase_offset];
                position = [position,mean(position_offset3)];
                speed = [speed,mean(speed_offset)];
                spike_count = [spike_count,length(idx)];
                assembly_size = [assembly_size,length(temp)./length(PyramidalIndexes)];
            end
        end
    end
    
    cell_assembliesSpatialCoverage.slope_temporal_phase{jj} = slope_temporal_phase;
    cell_assembliesSpatialCoverage.slope_position_phase{jj} = slope_position_phase;
    cell_assembliesSpatialCoverage.slope_position_temporal{jj} = slope_position_temporal;
    cell_assembliesSpatialCoverage.temporal_offset{jj} = temporal_offset_all;
    cell_assembliesSpatialCoverage.phase_offset{jj} = phase_offset_all;
    cell_assembliesSpatialCoverage.position{jj} = position;
    cell_assembliesSpatialCoverage.speed{jj} = speed;
    cell_assembliesSpatialCoverage.spike_count{jj} = spike_count;
    cell_assembliesSpatialCoverage.assembly_size{jj} = assembly_size;
    
    subplot(4,3,1)
    step_size2 = 0.3;
    bins = [0:step_size2:20];
    N = histcounts(slope_position_phase,bins,'Normalization','probability');
    plot(bins(1:end-1)+step_size2/2,N,'color',colors{jj}), hold on, 
    cell_assembliesSpatialCoverage.hist{jj} = N;
    
    subplot(4,3,2)
    N = histcounts(slope_temporal_phase,bins_speed2,'Normalization','probability');
    plot(bins_speed2(1:end-1)+step_size/2,N,'color',colors{jj}), hold on,
    cell_assembliesSpatialCoverage.hist_speed_normalized{jj} = N;
    cell_assembliesSpatialCoverage.speed_normalized_mean{jj} = nanmean(slope_temporal_phase(find(slope_temporal_phase<Inf)));
    
    subplot(4,3,4)
    plot(position,slope_position_phase,'.','color',colors{jj}), hold on,

    subplot(4,3,5)
    plot(position,slope_temporal_phase,'.','color',colors{jj}), hold on,

    subplot(4,3,7)
    plot(speed,slope_position_phase,'.','color',colors{jj}), hold on,

    subplot(4,3,8)
    plot(speed,slope_temporal_phase,'.','color',colors{jj}), hold on,

    subplot(4,3,10)
    N = histcounts(speed,bins_speed,'Normalization','probability');
    plot(bins_speed(1:end-1)+5,N,'color',colors{jj}), hold on

    subplot(4,3,3)
    N = histcounts(temporal_offset_all*1000,bins_temporal,'Normalization','probability');
    plot(bins_temporal(1:end-1)+1,N,'color',colors{jj}), hold on
    cell_assembliesSpatialCoverage.cumsum_temporal_offset{jj} = N;
    cell_assembliesSpatialCoverage.temporal_offset_mean{jj} = nanmean(temporal_offset_all*1000);
    subplot(4,3,6) % Spike count
    bins_spike_count = [1:max(spike_count)];
    N = histcounts(spike_count,bins_spike_count,'Normalization','probability');
    plot(bins_spike_count(1:end-1)+0.5,N,'color',colors{jj}), hold on
    cell_assembliesSpatialCoverage.cumsum_spike_count{jj} = N;
    cell_assembliesSpatialCoverage.spike_count_mean{jj} = nanmean(bins_spike_count);
    subplot(4,3,9) % Assembly size
    N = histcounts(assembly_size,bins_assembly_size,'Normalization','probability');
    plot(bins_assembly_size(1:end-1)+0.0025,N,'color',colors{jj}), hold on
    cell_assembliesSpatialCoverage.cumsum_assembly_size{jj} = N;
    cell_assembliesSpatialCoverage.assembly_size_mean{jj} = nanmean(assembly_size);

    subplot(4,3,12)
    N = histcounts(phase_offset_all,bins_phase,'Normalization','probability');
    plot(bins_phase(1:end-1)+0.05,N,'color',colors{jj}), hold on
    cell_assembliesSpatialCoverage.cumsum_phase_offset{jj} = N;
    cell_assembliesSpatialCoverage.phase_offset_mean{jj} = nanmean(phase_offset_all);
end


subplot(4,3,1), legend(states), xlabel('Distance (cm/cycle)')
subplot(4,3,2), xlabel('Time (sec/cycle)'), title([recording.name])
subplot(4,3,4), xlabel('Position (cm)'), xlim([0,350]), ylim([0,20]), ylabel('Distance (cm/cycle)')
subplot(4,3,5), xlabel('Position (cm)'), xlim([0,350]), ylim([0,0.20]), ylabel('Time (sec/cycle)')
subplot(4,3,7), xlabel('Speed (cm/s)'), ylabel('Distance'), xlim([0,150]), ylim([0,20])
subplot(4,3,8), xlabel('Speed (cm/s)'), ylabel('Time'), xlim([0,150]), ylim([0,0.20])
subplot(4,3,10), xlabel('Speed (cm/s)'), ylabel('Distribution'), xlim([0,150])
% subplot(4,2,8), plot(animal.time,animal.temperature), xlabel('Time (s)'), ylabel('Temperature')
subplot(4,3,3), plot(animal.time,animal.temperature), xlabel('Temporal offsets (ms)'), ylabel('Distribution'), xlim([0,150])
subplot(4,3,6), xlabel('Spikes (count/cycle)'), ylabel('Distribution'), axis tight %, xlim([0,100]), 
subplot(4,3,9), xlabel('Assembly fraction active'), ylabel('Distribution'), xlim([0,0.3])
subplot(4,3,12), xlabel('Phase offset'), ylabel('Distribution'), xlim([0,2*pi])

path = 'K:\Dropbox\Buzsakilab Postdoc\MatlabFigures\PhasePositionSlopes_fromSpikePairs';
print(gcf, [path,'\CellAssemblies_SpatialCoverage2_' recording.name],'-dpng')
save('cell_assembliesSpatialCoverage2.mat','cell_assembliesSpatialCoverage')
