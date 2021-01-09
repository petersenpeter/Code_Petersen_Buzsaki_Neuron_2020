function generateFigures(id)
% DOCID = '1WBqEo0OM5qdqmAD_7cGsJe0iyGf6hjgPXy-V2VlVY9c'
% result = GetGoogleSpreadsheet(DOCID);
% Medial Septum Circular Track
MedialSeptum_Recordings
% MS10: 61, 62, 63, 64 % gamma id: 62
% MS12: 78, 79, 80, 81,
% MS13: 92 (Peter_MS13_171129_105507_concat), 93 (Peter_MS13_171130_121758_concat)
% MS14:
% MS18:
% MS21: 126 (Peter_MS21_180629_110332_concat), 140 (Peter_MS21_180627_143449_concat), 143 (control: Peter_MS21_180719_155941_concat)
% MS22: 139 (Peter_MS22_180628_120341_concat), 127 (Peter_MS22_180629_110319_concat), 144 (control: Peter_MS22_180719_122813_concat)
% id = 126 % 62

% rate metrics completed: 78,79,126,140,143,139,127,80,81,92,93
% k = id
% recording = recordings(id);
% if ~isempty(recording.dataroot)
%     datapath = recording.dataroot;
% end
% for k = 9:length(datasets)

    recording = recordings(id);
    [session, basename, basepath, clusteringpath] = db_set_session('sessionName',recording.name);
    load([recording.name,'.animal.behavior.mat'])
    load([recording.name,'.trials.behavior.mat'])
%     load([recording.name,'.cooling.manipulation.mat'])
    
    spikes = loadSpikes('clusteringpath',recording.SpikeSorting.path,'clusteringformat',recording.SpikeSorting.method,'basename',recording.name);

    prebehaviortime = 0;
    
if recording.concat_behavior_nb > 0
    prebehaviortime = 0;
    nChannels = session.extracellular.nChannels;
    sr = session.extracellular.sr;
    if all(recording.concat_behavior_nb > 1)
        for i = 1:recording.concat_behavior_nb-1
            fullpath = fullfile([datapath,recording.animal_id], recording.concat_recordings{i}, [recording.concat_recordings{i}, '.dat']);
            temp2_ = dir(fullpath);
            prebehaviortime = prebehaviortime + temp2_.bytes/nChannels/2/sr;
        end
    end
    behaviortime = 0;
    for i = 1:length(recording.concat_behavior_nb)
        i1 = recording.concat_behavior_nb(i);
        fullpath = fullfile([datapath, recording.animal_id], recording.concat_recordings{i1}, [recording.concat_recordings{i1}, '.dat']);
        temp2_ = dir(fullpath);
        behaviortime = behaviortime+temp2_.bytes/nChannels/2/sr;
    end
    
else
    temp_ = dir(fname);
    behaviortime = temp_.bytes/nChannels/2/sr;
end
    
%     keyboard
    % Firing rate linearized
    behavior = animal;
    trials.state(find(trials.state==3))=1;
    behavior.state = trials.state;
    behavior.state = trials.state;
    behavior.pos = animal.pos_linearized;
    behavior.rim = zeros(size(animal.arm));
    behavior.rim(find(animal.arm | animal.rim)) = 1;
    behavior.pos_limits = animal.pos_linearized_limits;
    behavior.maze.boundaries = [diff(animal.pos_y_limits),diff(animal.pos_y_limits)+abs(animal.maze.polar_theta_limits(1))-5];
    behavior.speed_th = 10;
    
    spikes2 = [];
    
    for i = 1:size(spikes.ts,2)
        spikes2.ts{i} = spikes.ts{i};
        spikes2.times{i} = spikes.times{i};
        spikes2.cluID(i) = spikes.cluID(i);
        spikes2.total(i) = spikes.total(i);
        spikes2.pos{i} = interp1(behavior.time,behavior.pos,spikes2.times{i});
        %     spikes2.pos_linearized{i} = interp1(behavior.time,behavior.pos_linearized,spikes2.times{i});
        spikes2.speed{i} = interp1(behavior.time,behavior.speed,spikes2.times{i});
        spikes2.rim{i} = interp1(behavior.time,behavior.rim,spikes2.times{i},'nearest');
        spikes2.trials{i} = spikes.trials{i};
        spikes2.theta_freq{i} = spikes.theta_freq{i};
        spikes2.speed{i} = spikes.speed{i};
    end
    
    
%     firingRateMap = plot_FiringRateMapAverage('animal',behavior,'spikes',spikes2);
%     for i= 1:size(firingRateMap.map,2)
%         firingRateMap.map{i}(isnan(firingRateMap.map{i}))= 0;
%     end
%     firingRateMap.total = spikes2.total./behaviortime;
%     firingRateMap.boundaries = behavior.maze.boundaries;
    
%     sucess = saveStruct(firingRateMap,'firingRateMap','session',session);
    % save([recording.name, '.firingRateMap.firingRateMap.mat'],'firingRateMap')
    
    CoolingStates2 = plot_FiringRateMap2('animal',behavior,'spikes',spikes2,'trials',trials);
    CoolingStates2.boundaries = behavior.maze.boundaries;
    CoolingStates2.labels = {'NoCooling','Cooling'};
    for i=1:length(behavior.state_labels)
        CoolingStates2.trial_count(i) = length(unique(trials.trials2(find(trials.state==i))));
    end
    for i = 1:size(CoolingStates2.map,2)
        CoolingStates2.map{i}(isnan(CoolingStates2.map{i}))= 0;
    end    
    sucess = saveStruct(CoolingStates2,'firingRateMap','session',session);
    
    % save([recording.name, '.CoolingStates.firingRateMap.mat'],'CoolingStates')
    
%     
%     % Linearized
%     % PhasePrecessionSlope_linearized = plot_FiringRateMap(behavior,spikes2,trials,theta,sr,'linearized_cooling_states');
%     
%     % Linearized and colored and grouped by left/right trials
%     trials2 = trials;
%     trials2.state(find(~isnan(trials2.state))) = trials.stat(trials.trials2(find(~isnan(trials2.state))));
%     behavior2 = behavior;
%     behavior2.state(find(~isnan(trials2.state))) = trials.stat(trials.trials2(find(~isnan(trials2.state))));
%     behavior2.state_labels = trials.labels;
%     % PhasePrecessionSlope_linearized_left_right = plot_FiringRateMap(behavior2,spikes2,trials2,theta,sr,'linearized_left_right');
%     
%     LeftRight = plot_FiringRateMap2('animal',behavior2,'spikes',spikes2,'trials',trials);
%     LeftRight.boundaries = behavior.maze.boundaries;
%     LeftRight.labels = {'Left','Right'};
%     
    % % % 
    
%     sucess = saveStruct(LeftRight,'firingRateMap','session',session);
    % save([recording.name, '.LeftRight.firingRateMap.mat'],'LeftRight')
% end
% % % % % %

% %%
% % Firing rate linearized
% behavior = animal;
% behavior.state = trials.state;
% behavior.pos = animal.pos_linearized;
% behavior.rim = zeros(size(animal.arm));
% behavior.rim(find(animal.arm | animal.rim)) = 1;
% behavior.pos_limits = animal.pos_linearized_limits;
% behavior.maze.boundaries = [diff(animal.pos_y_limits),diff(animal.pos_y_limits)+abs(animal.maze.polar_theta_limits(1))-5];
% behavior.speed_th = 10;
% 
% spikes2 = [];
% 
% for i = 1:size(spikes.ts,2)
%     spikes2.ts{i} = spikes.ts{i};
%     spikes2.times{i} = spikes.times{i};
%     spikes2.cluID(i) = spikes.cluID(i);
%     spikes2.total(i) = spikes.total(i);
%     spikes2.pos{i} = interp1(behavior.time,behavior.pos,spikes2.times{i});
% %     spikes2.pos_linearized{i} = interp1(behavior.time,behavior.pos_linearized,spikes2.times{i});
%     spikes2.pos{i} = interp1(behavior.time,behavior.pos,spikes2.times{i});
%     spikes2.speed{i} = interp1(behavior.time,behavior.speed,spikes2.times{i});
%     spikes2.rim{i} = interp1(behavior.time,behavior.rim,spikes2.times{i},'nearest');
%     spikes2.trials{i} = spikes.trials{i};
%     spikes2.theta_freq{i} = spikes.theta_freq{i};
%     spikes2.speed{i} = spikes.speed{i};
% end
% 
% firingRateMap = plot_FiringRateMapAverage('animal',behavior,'spikes',spikes2);
% firingRateMap.total = spikes2.total./behaviortime;
% firingRateMap.unit(isnan(firingRateMap.unit)) = 0;
% firingRateMap.boundaries = behavior.maze.boundaries;
% save('firingRateMap.mat','firingRateMap')
% 
% firingRateMap_CoolingStates = plot_FiringRateMap2('animal',behavior,'spikes',spikes2,'trials',trials);
% firingRateMap_CoolingStates.boundaries = behavior.maze.boundaries;
% firingRateMap_CoolingStates.labels = behavior.state_labels;
% save('firingRateMap_CoolingStates.mat','firingRateMap_CoolingStates')
% 
% % Linearized
% % PhasePrecessionSlope_linearized = plot_FiringRateMap(behavior,spikes2,trials,theta,sr,'linearized_cooling_states');
% 
% % Linearized and colored and grouped by left/right trials
% trials2 = trials;
% trials2.state(find(~isnan(trials2.state))) = trials.stat(trials.trials2(find(~isnan(trials2.state))));
% behavior2 = behavior;
% behavior2.state(find(~isnan(trials2.state))) = trials.stat(trials.trials2(find(~isnan(trials2.state))));
% behavior2.state_labels = trials.labels;
% % PhasePrecessionSlope_linearized_left_right = plot_FiringRateMap(behavior2,spikes2,trials2,theta,sr,'linearized_left_right');
% 
% firingRateMap_LeftRight = plot_FiringRateMap2('animal',behavior2,'spikes',spikes2,'trials',trials);
% firingRateMap_LeftRight.boundaries = behavior.maze.boundaries;
% firingRateMap_LeftRight.labels = {'Left','Right'};
% save('firingRateMap_LeftRight.mat','firingRateMap_LeftRight')

%% Geisler analysis
% load('FiringRateMap_CoolingStates.mat')
% sucess = saveStruct(LeftRight,'firingRateMap','session',session);
load(fullfile(session.general.clusteringPath,[session.general.name,'.CoolingStates2.firingRateMap.mat']));
[PyramidalIndexes, cell_metrics] = loadCellMetrics('session',session.general.name,'putativeCellType',{'Pyr'});

conditions = {'NoCooling','Cooling'};
% conditions = {'Pre','Cooling','Post'};
colors = {'g','b','r'};

placefield_difference_all ={};
ccg_delay_all = {};
placefield_speed_all = {};
placefield_time_offset_all = {};
precession_slope_all = {};
ccg_delay_out = [];

plot_placefields = false;
plot_ccg_peaks = true;
plot_ccg_phase = false
;
close all

SpatialCoherence = [];
condition = [];
placefield_count = [];
placefield_interval = [];
placefield_state = [];
firingRateMap_CoolingStates = CoolingStates2;

x_bins = firingRateMap_CoolingStates.x_bins;
boundaries = firingRateMap_CoolingStates.boundaries;

for iii = 1:length(conditions)
    kk = 1;
    kk2 = 0;
%     for i = 1:size(firingRateMap_CoolingStates.map,2)
    for i = 1:size(PyramidalIndexes,2)
        
        temp = firingRateMap_CoolingStates.map{PyramidalIndexes(i)}(:,iii);
        temp2 = place_cell_condition(temp);
        
        SpatialCoherence(i,iii) = temp2.SpatialCoherence;
        condition(i,iii) = temp2.condition;
        placefield_count(i,iii) = temp2.placefield_count;
        placefield_interval{i,iii} = temp2.placefield_interval;
        placefield_state(:,i,iii) = temp2.placefield_state;
        
        %         figure(60+iii+(kk2-1)*3)
        %         subplot(4,4,kk)
        %         plot(x_bins,temp(:,i),'.-k'), hold on, plot(x_bins(find(placefield_state(:,i))),temp(find(placefield_state(:,i)),i),'or'),
        %         title([num2str(i), ' (UID ', num2str(PyramidalIndexes(i)),') ', num2str(iii), ])
        
        kk = kk + 1;
        if kk > 16
            kk = 1;
            kk2 = kk2+1;
        end
    end
end

place_cells_arm = find(sum(placefield_state(find(x_bins<boundaries(2)+150),:,1)));
placefield_state2 = sum(placefield_state,3);
placefield_interval2 = placefield_interval;
for kk1 = 1:size(placefield_interval,1)
    for kk2 = 1:length(conditions)
        for kk5 = 1:size(placefield_interval{kk1,kk2},1)
            temp3 = placefield_interval{kk1,kk2}(kk5,:);
            temp3 = temp3(1):temp3(2);
            for kk3 = setdiff([1:length(conditions)],kk2)
                conversed = 0;
                for kk4 = 1:size(placefield_interval{kk1,kk3},1)
                    temp4 = placefield_interval{kk1,kk3}(kk4,1):placefield_interval{kk1,kk3}(kk4,2);
                    if length(intersect(temp3,temp4))>5
                        conversed = 1;
                    end
                end
                if ~conversed
                    if ~isempty(placefield_interval2{kk1,kk2}) & size(placefield_interval2{kk1,kk2},1) >= kk5
                        placefield_interval2{kk1,kk2}(kk5,:) = nan;
                    end
                end
            end
        end
    end

    for kk2 = 1:length(conditions)
        while any(any(isnan(placefield_interval2{kk1,kk2})))
        j = 1;
        while j < size(placefield_interval2{kk1,kk2},1)+1
            if isnan(placefield_interval2{kk1,kk2}(j,:))
                placefield_interval2{kk1,kk2}(j,:) = [];
            end
            j = j +1;
        end
        end
    end
    if length(unique(cell2mat(cellfun(@(x) size(x,1), {placefield_interval2{kk1,:}},'UniformOutput',false))))>1
        for i9 = 1:length(conditions)
            if size(placefield_interval2{kk1,i9},1) > min(unique(cell2mat(cellfun(@(x) size(x,1), {placefield_interval2{kk1,:}},'UniformOutput',false))))
                [B,I] = sort(placefield_interval2{kk1,1}(:,2)-placefield_interval2{kk1,1}(:,1),'descend')
                temp3 = min(unique(cell2mat(cellfun(@(x) size(x,1), {placefield_interval2{kk1,:}},'UniformOutput',false))))
                placefield_interval2{kk1,i9} = placefield_interval2{kk1,i9}(I(1:temp3),:);
            end
        end
    end
end

for iii = 1:length(conditions)
    placefield_peak = [];
    placefield_speed = [];
    placefield_difference = [];
    placefield_speed_av = [];
    placefield_time_offset = [];
    
    spikes3 = [];
    times = [];
    times2 = [];
    groups = [];
    slope1 = [];
    placefield_interval = {placefield_interval2{:,iii}};
    k = 1;
%     place_cells_arm = find(sum(placefield_state(find(x_bins<boundaries(2)+150),:,1)));
    
    if plot_placefields
        figure(50)
        subplot(3,3,1+iii-1), hold on
    end
    for i = 1:length(place_cells_arm)
        fields = find(sum(placefield_interval{place_cells_arm(i)} < (boundaries(2)+150)/3,2));
        colors2 = rand(1,3);
        for j = 1:length(fields)
            pf_interval = x_bins(placefield_interval{place_cells_arm(i)}(fields(j),:));
            cell_id = PyramidalIndexes(place_cells_arm(i));
            spikes_infield = find(spikes.speed{cell_id} > 20 & spikes.pos_linearized{cell_id} > pf_interval(1) - 1 & spikes.pos_linearized{PyramidalIndexes(place_cells_arm(i))} < pf_interval(2) + 1 & spikes.state{PyramidalIndexes(place_cells_arm(i))} == iii);
            spikes3.times{k} = spikes.times{cell_id}(spikes_infield);
            spikes3.theta_phase2{k} = spikes.theta_phase2{cell_id}(spikes_infield);
            spikes3.theta_phase{k} = spikes.theta_phase{cell_id}(spikes_infield);
            spikes3.pos_linearized{k} = spikes.pos_linearized{cell_id}(spikes_infield);
            spikes3.pos_linearized{k}(find(spikes3.pos_linearized{k}>boundaries(2))) = spikes3.pos_linearized{k}(find(spikes3.pos_linearized{k}>boundaries(2))) + 200;
            spikes3.pos_linearized{k}(find(spikes3.pos_linearized{k}>boundaries(1))) = spikes3.pos_linearized{k}(find(spikes3.pos_linearized{k}>boundaries(1))) + 200;
            spikes3.UID(k) = k;
            spikes3.UID_old(k) = cell_id;
            times = [times; spikes3.times{k}];
            times2 = [times2; spikes3.theta_phase2{k}'];
            groups = [groups; k * ones(length(spikes3.times{k}),1)];
            if ~isempty(spikes3.theta_phase{k}) && length(spikes_infield)>5
                [slope1(k),~,~] = CircularLinearRegression(spikes3.theta_phase{k},spikes3.pos_linearized{k},-1);
            else
                slope1(k) = 0;
            end
            
            if plot_placefields
                plot(spikes.pos_linearized{cell_id}(spikes_infield),i * ones(1,length(spikes_infield)),'.','color',colors2), hold on,
            end
            
            placefield_peak(k) = mean(spikes.pos_linearized{cell_id}(spikes_infield));
            placefield_speed(k) = nanmean(spikes.speed{cell_id}(spikes_infield));
            k = k + 1;
        end
    end
    
    [times,I] = sort(times);
    times2 = times2(I);
    groups = groups(I);

    if plot_placefields
        xlabel('Position (cm)'), ylabel('Placefield'), title(['Placefields - ', conditions{iii} ]), xlim([0,345]), axis tight, gridxy(boundaries)
    end

    % Fast theta-cycle difference between place fields
    [ccg,t] = CCG(times,groups,'binSize',0.001,'duration',1.2);
    t = t * 1000; % from sec to msec
    ACGs = [];
    
    % ACGs in time space
    for i = 1:size(ccg,2)
        ACGs(i,:) = nanconv(ccg(:,i,i)',gausswin(60)','edge');
    end
    
    % Peak-peak estimation between place fields
    [ccg2,t2] = CCG(times,groups,'binSize',0.005,'duration',1.3);
    
    % Phase space 
    [ccg3,t3] = CCG(times2,groups,'binSize',0.0628,'duration',20*pi);
    ccg_count = [];
    ccg_delay = [];
    locs = [];
    locs_all = {};
    locs3_all = {};
    pks = [];
    kk = 1;
    before = [];
    locs3 = [];
    pks3 = [];
    before3 = [];
    
    placefield_ccgs_time = [];
    placefield_ccgs_phase = [];
    pairIDs_CCGs_phase = [];
    pairIDs_CCGs_time = [];
    pairIDs_time = [];
    pairIDs_phase = [];
    pairIDs_PyrInt = [];
            
    t0 = find(t == 0);
    t03 = find(t3 == 0);
    precession_slope = [];
    for i = 1:size(ccg,3)-1
        if plot_ccg_peaks | plot_ccg_phase
            figure
            kkk = 1;
        end
        for j = i+1:size(ccg,3)
            
            ccg_trace2 = nanconv(ccg2(:,i,j)',gausswin(160)','edge');
            ccg_trace3 = nanconv(ccg3(:,i,j)',gausswin(60)','edge');
            [~,indx] = max(ccg_trace2);
            placefield_time_offset(kk) = t2(indx);
            
            placefield_difference(kk) = placefield_peak(j) - placefield_peak(i); 
            placefield_speed_av(kk) = nanmean([placefield_speed(j),placefield_speed(i)]);
            ccg_trace = nanconv(ccg(:,i,j)',gausswin(60)','edge');
            [ccg_count(kk),ccg_delay(kk)] = max(ccg_trace);
            locs(kk) = 0;
            pks(kk) = 0;
            before(kk) = 0;
            
            locs3(kk) = 0;
            pks3(kk) = 0;
            before3(kk) = 0;
            precession_slope(kk,:) = [slope1(i), slope1(j)];
            pairIDs_Pyr(kk,:) = [i,j];
            locs_all{kk} = nan;
            locs3_all{kk} = nan;
            
            if ccg_count(kk)>0.2
                % Time
                [pks_temp,locs_temp] =  findpeaks(ccg_trace);
                locs_temp(ccg_trace(locs_temp)<0.08) = [];
                locs_temp(locs_temp<16) = []; locs_temp(locs_temp>length(ccg_trace)-16) = []; 
                locs_temp = locs_temp(find(ccg_trace(locs_temp+15)<ccg_trace(locs_temp) & ccg_trace(locs_temp-15)<ccg_trace(locs_temp)));
                pks_temp = ccg_trace(locs_temp);
                
                % Phase space
                [pks_temp3,locs_temp3] =  findpeaks(ccg_trace3);
                locs_temp3(ccg_trace3(locs_temp3)<0.8) = [];
                locs_temp3(locs_temp3<16) = []; locs_temp3(locs_temp3>length(ccg_trace3)-16) = []; 
                locs_temp3 = locs_temp3(find(ccg_trace3(locs_temp3+15)<ccg_trace3(locs_temp3) & ccg_trace3(locs_temp3-15)<ccg_trace3(locs_temp3)));
                pks_temp3 = ccg_trace3(locs_temp3);
                
                if plot_ccg_peaks && kkk<50
                    subplot(7,7,kkk)
                    plot(t,ccg(:,i,j)','color',[0.5,0.5,0.5]), hold on, plot(t,ccg_trace,'linewidth',2), plot(t(locs_temp),ccg_trace(locs_temp),'or'), xlim([t(1),t(end)])
                    kkk = kkk+1;
                end
                
                if plot_ccg_phase && kkk<50
                    subplot(7,7,kkk)
                    plot(t3,ccg3(:,i,j)','color',[0.5,0.5,0.5]), hold on, plot(t3,ccg_trace3,'linewidth',2), plot(t3(locs_temp3),ccg_trace3(locs_temp3),'or'), xlim([t3(1),t3(end)])
                    kkk = kkk+1;
                end

                if ~isempty(locs_temp)
                    if any(locs_temp<=t0) & any(locs_temp>t0)
                        
                        [~,idx] = sort(abs(locs_temp-t0));
                        temp333 = find(locs_temp<=t0); 
                        temp333 = locs_temp(temp333(end));
                        temp334 = locs_temp(find(locs_temp>t0,1));
                        if ccg_trace(temp333) > ccg_trace(temp334)
                            locs(kk) = temp333;
                            pks(kk) = ccg_trace(temp333);
                            before(kk) = 1;
                        else
                            locs(kk) = temp334;
                            pks(kk) = ccg_trace(temp334);
                            before(kk) = 2;
                        end
                        locs_all{kk} = t(locs_temp);
                        placefield_ccgs_time(kk,:) = ccg_trace;
                        pairIDs_CCGs_time(kk,:) = [i,j];
                        if plot_ccg_peaks
                            plot(t(locs(kk)),pks(kk),'xk','linewidth',4);
                        end
                        
                    end
                    
                    % Phase
                    if any(locs_temp3<=t03) & any(locs_temp3>t03)
                        [~,idx] = sort(abs(locs_temp3-t03));
                        temp333 = find(locs_temp3<=t03); 
                        temp333 = locs_temp3(temp333(end));
                        temp334 = locs_temp3(find(locs_temp3>t03,1));
                        if ccg_trace3(temp333) > ccg_trace3(temp334)
                            locs3(kk) = temp333;
                            pks3(kk) = ccg_trace3(temp333);
                            before3(kk) = 1;
                        else
                            locs3(kk) = temp334;
                            pks3(kk) = ccg_trace3(temp334);
                            before3(kk) = 2;
                        end
                        locs3_all{kk} = t3(locs_temp3);
                        placefield_ccgs_phase(kk,:) = ccg_trace3;
                        pairIDs_CCGs_phase(kk,:) = [i,j];
                        if plot_ccg_phase
                            plot(t3(locs3(kk)),pks3(kk),'xk','linewidth',4);
                        end
                    end
                    
%                     if ccg_delay(kk)<t0
%                         temp333 = find(locs_temp<t0);
%                         if ~isempty(temp333)
%                             locs(kk) = locs_temp(temp333(end));
%                             pks(kk) = pks_temp(temp333(end));
%                             before(kk) = 1;
%                         end
%                     else
%                         temp333 = find(locs_temp>t0,1);
%                         if ~isempty(temp333)
%                             locs(kk) = locs_temp(temp333);
%                             pks(kk) = pks_temp(temp333);
%                             before(kk) = 2;
%                         end
%                     end
                end
            end
            
            kk = kk + 1;
            
        end
    end
    
    indx = find(ccg_count==0 | pks == 0 ); % | abs(placefield_difference)>80
    placefield_difference(indx) = [];
    placefield_speed_av(indx) = [];
    placefield_time_offset(indx) = [];
    precession_slope(indx,:) = [];
    pairIDs_Pyr(indx,:) = [];
    
    ccg_delay(indx) = [];
    ccg_count(indx) = [];
    locs(indx) = [];
    
    pks(indx) = [];
    locs3(indx) = [];
    locs3(locs3==0) = 1;
    pks3(indx) = [];
    
    locs_all(indx) = [];
    locs3_all(indx) = [];
    
    ccg_delay = t(locs)';
    ccg_delay_phase = t3(locs3)';
    
    ccg_delay_time_all = locs_all;
    ccg_delay_phase_all2 = locs3_all;
    
    
    x = placefield_difference; y1 = ccg_delay;
    subset = find(y1 < 95 & y1 > -95);
    x = x(subset); y1 = y1(subset);
    
    if plot_placefields
        figure(50)
        subplot(3,3,4+iii-1)
        plot(placefield_difference,ccg_delay,'o'), hold on
        P = polyfit(x,y1,1); yfit = P(1)*x+P(2); plot(x,yfit,[colors{iii},'-']);
        text(-75,-150,['Slope: ' num2str(P(1),3)],'Color','k')
        [R,P] = corrcoef(x,y1);
        text(-75,100,[conditions{iii},': R = ' num2str(R(2,1),3),', P = ', num2str(P(2,1),3)],'Color','k')
        xlabel('Distance (cm)'), ylabel('Time lag (s)'), title(['Compression - state ' num2str(iii)]), grid on, hold on, xlim([-80,80]), ylim([-110,110])
        %     [param]=sigm_fit(x,y1);
        subplot(3,3,7+iii-1)
        plot(placefield_difference./placefield_speed_av,ccg_delay,'o'), xlabel('L - [Distance] (s)'), ylabel('Time lag (s)'), xlim([-0.8,0.8]), grid on
    end
    
    % Interneurons
    placefield_interneurons_ccgs_time = [];
    kkk5 = 1;
    if exist([recording.SpikeSorting.path,'/cell_metrics.mat'])
        load([recording.SpikeSorting.path,'/cell_metrics.mat']);
        InterneuronsIndexes = find(contains(cell_metrics.putativeCellType,'Narrow Interneuron'));
        spikes4 = spikes3;
        if length(InterneuronsIndexes)> 0
            for i = 1:length(InterneuronsIndexes)
                cell_id = InterneuronsIndexes(i);
                spikes4.times{i+length(spikes3.times)} = spikes.times{cell_id};
                spikes4.UID(i+length(spikes3.times)) = i+length(spikes3.times);
            end
            spikes4 = generateSpinDices(spikes4);
            [ccg4,t4] = CCG(spikes4.spindices(:,1),spikes4.spindices(:,2),'binSize',0.001,'duration',1.2);
            for i = 1:length(InterneuronsIndexes)
                for j = 1:length(spikes3.times)
                    ccg_trace4 = nanconv(ccg4(:,i+length(spikes3.times),j)',gausswin(60)','edge');
                    if sum(ccg_trace4) > 0
                        placefield_interneurons_ccgs_time(kkk5,:) = ccg_trace4;
                        pairIDs_PyrInt(kkk5,:) = [spikes3.UID_old(j),InterneuronsIndexes(i)];
                        kkk5 = kkk5 + 1;
                    end
                end
            end
        end
    end
    
    placefield_difference_all{iii} = placefield_difference;
    ccg_delay_all{iii} = ccg_delay;
    ccg_delay_phase_all{iii} = ccg_delay_phase;
    placefield_speed_all{iii} = placefield_speed_av;
    placefield_time_offset_all{iii} = placefield_time_offset;
    precession_slope_all{iii} = precession_slope;
    pairIDs_Pyr_all{iii} = pairIDs_Pyr;
    
    % CCG traces for each pair
    placefield_ccgs_time_all{iii} = placefield_ccgs_time;
    pairIDs_CCGs_time_all{iii} = pairIDs_CCGs_time;
    placefield_ccgs_phase_all{iii} = placefield_ccgs_phase;
    pairIDs_CCGs_phase_all{iii} = pairIDs_CCGs_phase;
    
    % All peaks
    ccg_delay_out{iii}.time_all = ccg_delay_time_all;
    ccg_delay_out{iii}.phase_all = ccg_delay_phase_all2;
    
    % ACGs
    ACGs_all{iii} = ACGs;
    
    % CCG between pyramidal cells and interneurons
    placefield_interneurons_ccgs_time_all{iii} = placefield_interneurons_ccgs_time;
    pairIDs_PyrInt_all{iii} = pairIDs_PyrInt;
end

% save('PlaceFields2.mat','placefield_difference_all','ccg_delay_all','placefield_speed_all','placefield_time_offset_all','precession_slope_all','ccg_delay_phase_all','ccg_delay_out','placefield_ccgs_time_all','placefield_ccgs_phase_all','ACGs_all','placefield_interneurons_ccgs_time_all','pairIDs_PyrInt_all','pairIDs_Pyr_all','pairIDs_CCGs_time_all','pairIDs_CCGs_phase_all')
disp('generateFigures complete!')
