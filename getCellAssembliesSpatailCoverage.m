function cell_assembliesSpatialCoverage = getCellAssembliesSpatailCoverage(id)
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
bins_speed = [0:10:150];

figure('position',[50,50,850,1000])
for jj = 1:3
    p1 = [];
    p2 = [];
    position = [];
    speed = [];
    kkk = 1;
    CellAssemblies = [];
    x_data = [];
    y_data = [];
    if any(diff(trials.cooling)==-1)
        doubleState = find(diff(trials.cooling)==-1);
        doubleState = 500;
    else
        doubleState = 500;
    end
    trials_state = intersect(find(trials.cooling==jj & trials.cooling < doubleState),setdiff([1:length(trials.cooling)],trials.error));
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

%     figure(1)
    hold on
    for j = 1:ceil(max(allspikes_theta2)/(2*pi))
        idx = find(allspikes_theta2 > (j-1)*2*pi-pi & allspikes_theta2<j*2*pi-pi);
        if idx>5 & max(allspikes_pos_linearized(idx))-min(allspikes_pos_linearized(idx)) < 50
            x_data = [x_data;allspikes_theta(idx)];
            y_data = [y_data;allspikes_pos_linearized(idx)-min(allspikes_pos_linearized(idx))];
            
%             subplot(3,2,jj*2-1), hold on, xlim([0,20])
%             plot(allspikes_pos_linearized(idx)-min(allspikes_pos_linearized(idx)),allspikes_theta(idx),'.-')
%             subplot(3,2,jj*2), hold on, xlim([0,0.3])
%             plot((allspikes_pos_linearized(idx)-min(allspikes_pos_linearized(idx)))/mean(allspikes_speed(idx)),allspikes_theta(idx),'.-')
            p = polyfit(allspikes_theta(idx),allspikes_pos_linearized(idx)-min(allspikes_pos_linearized(idx)),1);
            p1 = [p1,p(1)*2*pi];
            p = polyfit(allspikes_theta(idx),(allspikes_pos_linearized(idx)-min(allspikes_pos_linearized(idx)))/mean(allspikes_speed(idx)),1);
            p2 = [p2,p(1)*2*pi];
            position = [position,mean(allspikes_pos_linearized(idx))];
            speed = [speed,mean(allspikes_speed(idx))];
        end
    end
    
    keyboard
    InstantaneousTheta
    
    cell_assembliesSpatialCoverage.x_data{jj} = x_data;
    cell_assembliesSpatialCoverage.y_data{jj} = y_data;
    cell_assembliesSpatialCoverage.p1{jj} = p1;
    cell_assembliesSpatialCoverage.p2{jj} = p2;
    cell_assembliesSpatialCoverage.position{jj} = position;
    cell_assembliesSpatialCoverage.speed{jj} = speed;
    
%     figure(2)
    subplot(4,2,1)
    step_size = 0.3;
    bins = [5:step_size:20];
    N = histcounts(p1,bins,'Normalization','probability');
    plot(bins(1:end-1)+step_size/2,N,'color',colors{jj}), hold on, 
    cell_assembliesSpatialCoverage.hist{jj} = N;
    
    
    
    subplot(4,2,2)
    step_size = 0.003;
    bins = [0.05:step_size:0.20];
    N = histcounts(p2,bins,'Normalization','probability');
    plot(bins(1:end-1)+step_size/2,N,'color',colors{jj}), hold on, 
    cell_assembliesSpatialCoverage.hist_speed_normalized{jj} = N;
    
    subplot(4,2,3)
    plot(position,p1,'.','color',colors{jj}), hold on,
%     mean1 = [];
%     std1 = [];
%     for i = 1:length(bins_position)-1
%         idx1 = find(position > bins(i) & position < bins(i+1) & 1./p1>0 & 1./p1 < 3);
%         mean1(i) = mean(1./p1(idx1));
%         std1(i) = std(1./p1(idx1));
%     end
%     errorbar(bins_position(1:end-1)+10,mean1,std1,'color',colors{jj});
    
    subplot(4,2,5)
    plot(speed,p1,'.','color',colors{jj}), hold on,
    subplot(4,2,4)
    plot(position,p2,'.','color',colors{jj}), hold on,
%     mean1 = [];
%     std1 = [];
%     for i = 1:length(bins_position)-1
%         idx2 = find(position > bins(i) & position < bins(i+1) & p2>20 & p2 < 80);
%         mean1(i) = mean(p2(idx2));
%         std1(i) = std(p2(idx2));
%     end
%     errorbar(bins_position(1:end-1)+10,mean1,std1,'color',colors{jj});
    
    subplot(4,2,6)
    plot(speed,p2,'.','color',colors{jj}), hold on,

    subplot(4,2,7)
    N = histcounts(speed,bins_speed,'Normalization','probability');
    plot(bins_speed(1:end-1)-5,N,'color',colors{jj}), hold on
end

% figure(1)
% subplot(3,2,1), title(['Phase va position']), ylabel(states{1}), 
% subplot(3,2,2), title(['Phase va time'])
% subplot(3,2,3), ylabel(states{2}), 
% subplot(3,2,5), ylabel(states{3}), 

% figure(2)
subplot(4,2,1), title([recording.name]), legend(states), xlabel('Distance (cm/cycle)')
subplot(4,2,2), xlabel('Time (sec/cycle)')
subplot(4,2,3), xlabel('Position (cm)'), xlim([0,350]), ylim([0,20]), ylabel('Distance (cm/cycle)')
subplot(4,2,5), xlabel('Speed (cm/s)'), ylabel('Distance'), xlim([0,150]), ylim([0,20])
subplot(4,2,4), xlabel('Position (cm)'), xlim([0,350]), ylim([0,0.20]), ylabel('Time (sec/cycle)')
subplot(4,2,6), xlabel('Speed (cm/s)'), ylabel('Time'), xlim([0,150]), ylim([0,0.20])
subplot(4,2,7), xlabel('Speed (cm/s)'), ylabel('Distribution'), xlim([0,150])
subplot(4,2,8), plot(animal.time,animal.temperature), xlabel('Time (s)'), ylabel('Temperature')

path = 'C:\Users\peter\Dropbox\Buzsakilab Postdoc\Medial Septum Cooling Project\PhasePositionSlopes';
% print(gcf, [path,'\CellAssemblies_SpatialCoverage_' recording.name],'-dpdf')
save('cell_assembliesSpatialCoverage.mat','cell_assembliesSpatialCoverage')
