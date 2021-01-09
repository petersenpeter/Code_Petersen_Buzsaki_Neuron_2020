% Mark Brandon plot
clear all
MedialSeptum_Recordings
% MS10: 61, 62, 63, 64 % gamma id: 62
% MS12: 78, 79, 80, 81,
% MS13: 92 (Peter_MS13_171129_105507_concat), 93 (Peter_MS13_171130_121758_concat)
% MS14:
% MS18:
% MS21: 126 (Peter_MS21_180629_110332_concat), 140 (Peter_MS21_180627_143449_concat), 143 (control: Peter_MS21_180719_155941_concat)
% MS22: 139 (Peter_MS22_180628_120341_concat), 127 (Peter_MS22_180629_110319_concat), 144 (control: Peter_MS22_180719_122813_concat)

close all,
datasets_stim = {'Peter_MS21_180629_110332_concat','Peter_MS21_180627_143449_concat','Peter_MS22_180628_120341_concat','Peter_MS22_180629_110319_concat','Peter_MS13_171129_105507_concat','Peter_MS13_171130_121758_concat'};
datasets_nostim = {'Peter_MS21_180719_155941_concat','Peter_MS22_180719_122813_concat'};

datasets_collection{1} = datasets_stim;
datasets_collection{2} = datasets_nostim;
[C1,ia1,ib] = intersect({recordings.name},datasets_collection{1},'stable');
[C2,ia2,ib] = intersect({recordings.name},datasets_collection{2},'stable');
animals_unique = unique({recordings(ia1).animal_id,recordings(ia2).animal_id});
corrmatrix_in{1} = [1,2;1,3;2,3];
corrmatrix_in{2} = [1,2;1,3;2,3];
corrmatrix_out = [];
xlimits = [0:0.05:1];
h2 = [];
mmm = 0;
corrmatrix2 = [];
corrmatrix3 = [];
labels = {'Pre','Cooling','Post'};
placefield_changed = [];
placefield_changed2 = [];
cell_metrics_effects = [];
cell_metrics_effects2 = [];
temp333 = nan(1,500);

for m = 1:2
    recording_names = {};
    corrmatrix = [];
    clear placecell
    clear normalized_map
    datasets = datasets_collection{m};
    [datasets,ia,ib] = intersect({recordings.name},datasets);
    if m == 1
        cell_metrics_batch = LoadCellMetricBatch('sessions',datasets)
%         cell_metrics_batch = LoadCellMetricBatch('sessions',datasets,'saveAs','cell_metrics_excludeOpto');
    else
        cell_metrics_batch = LoadCellMetricBatch('sessions',datasets);
    end
    placecell_stability = [];
    firing_rate_map_all.unit = [];
    firing_rate_map_average_all.unit =[];
    firing_rate_map_all.trial_count = [];
    firing_rate_map_all.animal_id = [];
    firing_rate_map_all.dataset_id = [];
    firing_rate_map_all.unit_count = [];
    placefield_minus_poststim = 0;
    placefield_minus_stim = 0;
    placefield_plus_stim = 0;
    placefield_plus_poststim = 0;

    placefield_minus_poststim2 = zeros(1,length(datasets));
    placefield_minus_stim2 = zeros(1,length(datasets));
    placefield_plus_stim2 = zeros(1,length(datasets));
    placefield_plus_poststim2 = zeros(1,length(datasets));
    placefields_binary_hist = [];
%     stim_pos_linearized_all = [];
%     reward_points_linearized  = [];
%     stim_pos_linearized_waveform_all = [];
    
    for k = 1:length(datasets)
        id = ia(k);
        cd(fullfile(datapath,recordings(id).animal_id,recordings(id).name))
%         cd(recordings(id).folder)
        
        animal_id(k) = find(strcmp(recordings(id).animal_id,animals_unique));
        recording = recordings(id);
        
        temp = load('firingRateMap_CoolingStates.mat');
        temp2 = load('firingRateMap.mat');
        firing_rate_map = temp.firingRateMap_CoolingStates;
        firing_rate_map_average = temp2.firingRateMap;
%         stim_pos_linearized = temp.stim_pos_linearized;
        test23 = unique(firing_rate_map.state_labels,'stable');
        
        test231(1) = find(strcmp(test23,labels{1}));
        test231(2) = find(strcmp(test23,labels{2}));
        test231(3) = find(strcmp(test23,labels{3}));

        firing_rate_map_all.unit = [firing_rate_map_all.unit,firing_rate_map.unit(:,:,test231)];
        firing_rate_map_all.trial_count = [firing_rate_map_all.trial_count,firing_rate_map.trial_count(test231)'*ones(1,size(firing_rate_map.unit,2))];
        firing_rate_map_all.animal_id = [firing_rate_map_all.animal_id,animal_id(k)*ones(1,size(firing_rate_map.unit,2))];
        firing_rate_map_all.dataset_id = [firing_rate_map_all.dataset_id,k*ones(1,size(firing_rate_map.unit,2))];
        firing_rate_map_average_all.unit = [firing_rate_map_average_all.unit,firing_rate_map_average.unit];
%         stim_pos_linearized = hist(stim_pos_linearized, firing_rate_map.x_bins);
%         stim_pos_linearized_all = [stim_pos_linearized_all; stim_pos_linearized];
%         reward_points_linearized = [reward_points_linearized; recording.maze.reward_points_linearized];
%         recording_names = [recording_names,recording.name];
        
%         if m == 1
%             stim_pos_linearized_waveform = temp.stim_pos_linearized_waveform;
%             stim_pos_linearized_waveform_all = [stim_pos_linearized_waveform_all;stim_pos_linearized_waveform];
%         end
    end
    
    firing_rate_map.unit = firing_rate_map_all.unit;
    firing_rate_map.animal_id = firing_rate_map_all.animal_id;
    firing_rate_map.trial_count = firing_rate_map_all.trial_count;
    firing_rate_map.dataset_id = firing_rate_map_all.dataset_id;
    indexOriginal = [1:size(firing_rate_map.dataset_id,2)];
%     index2delete = find(firing_rate_map.total>10);
%     indexOriginal(index2delete) = [];
%     firing_rate_map_average_all.unit(:,index2delete,:)=[];
%     firing_rate_map.unit(:,index2delete,:)=[];
%     firing_rate_map.animal_id(index2delete)= [];
%     firing_rate_map.trial_count(:,index2delete) = [];
%     firing_rate_map.dataset_id(index2delete)= [];

    firing_rate_map_average_all.unit(isnan(firing_rate_map_average_all.unit)) = 0;
    firing_rate_map.unit(isnan(firing_rate_map.unit)) = 0;

    for i = 1:size(temp.firingRateMap_CoolingStates.unit,2)
        placecell(i) = place_cell_condition(firing_rate_map_average_all.unit(:,i)','firing_rate_peak_min',4);
    end
    
    index2delete = find([placecell(:).condition]==0);
    indexOriginal(index2delete) = [];
    firing_rate_map.unit(:,index2delete,:) = [];
    firing_rate_map.animal_id(index2delete)= [];
    firing_rate_map.trial_count(:,index2delete) = [];
    firing_rate_map.dataset_id(index2delete)= [];

    unit_count_dataset = unique(firing_rate_map.dataset_id);
    unique_datasets = [];
    for k = 1:length(unit_count_dataset)
        unique_datasets(k) = length(find(firing_rate_map.dataset_id==unit_count_dataset(k)));
    end
    clear placecell_states;
    clear placecell_stability;
    for i = 1:size(firing_rate_map.unit,2)
        X2 = zeros();
        for j = 1:length(test231)
            placecell_states(i,j) = place_cell_condition(firing_rate_map.unit(:,i,j)','firing_rate_peak_min',4);
        end
        
        % placefield_minus_poststim
        test10 = diff([0,placecell_states(i,1).placefield_state,0] .* [0,placecell_states(i,2).placefield_state,0]);
        index1 = find(test10==1)-1; if index1==0, index1=1; end
        index2 = find(test10==-1)-1; if index2> length(placecell_states(i,1).placefield_state), index2=index2-1; end
        for k = 1:length(index1)
            if index1(k)==0, index1(k)=1; end
            if index2(k)> length(placecell_states(i,1).placefield_state), index2(k)=index2(k)-1; end
            if any(placecell_states(i,3).placefield_state(index1(k):index2(k)))==0
                placefield_minus_poststim = placefield_minus_poststim+1./firing_rate_map.trial_count(3,i);
                temp333(i) = firing_rate_map.trial_count(3,i);
                placefield_minus_poststim2(firing_rate_map.dataset_id(i)) = placefield_minus_poststim2(firing_rate_map.dataset_id(i))+1./firing_rate_map_all.trial_count(3,i);
            end
        end

        % placefield_minus_stim
        test10 = diff([0,placecell_states(i,1).placefield_state,0]);
        index1 = find(test10==1);
        index2 = find(test10==-1)-1;
        for k = 1:length(index1)
            if index1(k)==0, index1(k)=1; end
            if index2(k)> length(placecell_states(i,1).placefield_state), index2(k)=index2(k)-1; end
            if any(placecell_states(i,2).placefield_state(index1(k):index2(k)) + placecell_states(i,3).placefield_state(index1(k):index2(k)))==0
                placefield_minus_stim = placefield_minus_stim+1./firing_rate_map.trial_count(2,i);
                placefield_minus_stim2(firing_rate_map.dataset_id(i)) = placefield_minus_stim2(firing_rate_map.dataset_id(i))+1./firing_rate_map.trial_count(2,i);
            end
        end

        % placefield_creation_stim
        test10 = diff([0,placecell_states(i,2).placefield_state,0] .* [0,placecell_states(i,3).placefield_state,0]);
        index1 = find(test10==1)-1; 
        index2 = find(test10==-1)-1;
        for k = 1:length(index1)
            if index1(k)==0, index1(k)=1; end
            if index2(k)> length(placecell_states(i,1).placefield_state), index2(k)=index2(k)-1; end
            if any(placecell_states(i,1).placefield_state(index1(k):index2(k)))==0
                placefield_plus_stim = placefield_plus_stim+1./firing_rate_map.trial_count(2,i);
                placefield_plus_stim2(firing_rate_map.dataset_id(i)) = placefield_plus_stim2(firing_rate_map.dataset_id(i))+1./firing_rate_map.trial_count(2,i);
            end
        end

        % placefield_creation_poststim
        test10 = diff([0,placecell_states(i,3).placefield_state,0]);
        index1 = find(test10==1);
        index2 = find(test10==-1)-1; 
        for k = 1:length(index1)
            if index1(k)==0, index1(k)=1; end
            if index2(k)> length(placecell_states(i,1).placefield_state), index2(k)=index2(k)-1; end
            if any(placecell_states(i,2).placefield_state(index1(k):index2(k)) + placecell_states(i,1).placefield_state(index1(k):index2(k)))==0
                placefield_plus_poststim = placefield_plus_poststim+1./firing_rate_map_all.trial_count(3,i);
                placefield_plus_poststim2(firing_rate_map.dataset_id(i)) = placefield_plus_poststim2(firing_rate_map.dataset_id(i))+1./firing_rate_map_all.trial_count(3,i);
            end
        end
        test = placecell_states(i,3).placefield_state+placecell_states(i,2).placefield_state+placecell_states(i,1).placefield_state;
        pks = findpeaks(test);
        if any(pks == 2)
            placecell_stability(i)=0;
        else
            placecell_stability(i)=1;
        end
    end
    
    placefield_count = sum([placecell.placefield_count]);
    placefield_changed(:,m) = [placefield_plus_stim/placefield_count, placefield_plus_poststim/placefield_count, placefield_minus_stim/placefield_count, placefield_minus_poststim/placefield_count];
    placefield_changed2{m} = [placefield_plus_stim2', placefield_plus_poststim2', placefield_minus_stim2', placefield_minus_poststim2']./unique_datasets';

    % CellMetrics
    cell_metrics_batch.placeCellStability = repmat({''},1,length(cell_metrics_batch.spikeCount));
    cell_metrics_batch.placeCellStability(indexOriginal(find(placecell_stability))) = repmat({'Stable'},1,sum(placecell_stability));
    cell_metrics_batch.placeCellStability(indexOriginal(find(~placecell_stability))) = repmat({'Instable'},1,sum(~placecell_stability));
    CellExplorer('metrics',cell_metrics_batch)
    
    [~,ipp] = max(firing_rate_map.unit(:,:,end));
    [~,i2] = sort(ipp);
    [C,ia,ib] = intersect(i2, find(placecell_stability==0),'stable');
    [C2,ia2,ib2] = intersect(i2, find(placecell_stability==1),'stable');
    i2 = [C,C2];
    
    firing_rate_map.unit = firing_rate_map.unit(:,i2,:);
    
    for j = 1:size(firing_rate_map.unit,3)
        figure(200+m),
        normalized_map{j} = (firing_rate_map.unit(:,:,j)./max(firing_rate_map.unit(:,:,j)))'; 
        subplot(2,size(firing_rate_map.unit,3),j), imagesc(firing_rate_map.x_bins,1:size(firing_rate_map.unit,2),normalized_map{j}), 
        title(labels{j}), xlabel('Position (cm)'), hold on, plot([0;192],(length(C)+1)*[1;1],'w')
        subplot(2,size(firing_rate_map.unit,3),j+size(firing_rate_map.unit,3))
        stairs(firing_rate_map.x_bins,nansum(normalized_map{j}(find(placecell_stability==1),:))./sum(placecell_stability)), hold on
        stairs(firing_rate_map.x_bins,nansum(normalized_map{j}(find(placecell_stability==0),:))./sum(~placecell_stability)), xlabel('Position (cm)'), if j == 1; ylabel('Position of place fields'), legend({'Stabile fields','Changing fields'}); end
        figure(210+m)
        plot(firing_rate_map.x_bins,nansum(normalized_map{j})), hold on, xlabel('Position (cm)'), ylabel('Position of place fields');
        if j==size(firing_rate_map.unit,3)
            xlabel('Position (cm)'), title('Position of place fields'), legend({'Pre','Stim','PostStim'})
        end
    end
    
    for mm = 1:size(corrmatrix_in{m},1)
        for i = 1:size(normalized_map{1},1)
            corrmatrix(mm,i) = corr(normalized_map{corrmatrix_in{m}(mm,1)}(i,:)',normalized_map{corrmatrix_in{m}(mm,2)}(i,:)'); % ,'type','Spearman'
        end
        h1 = histcounts(corrmatrix(mm,:),xlimits,'Normalization','probability');
        mmm = mmm + 1;
        h2(mmm,:) = [h1,h1(end)];
    end
    corrmatrix2{m} = sort(corrmatrix');
    corrmatrix3{m} = corrmatrix;
    
    if m ==1
        figure(100),
        subplot(2,3,4)
        scatter(corrmatrix(1,:),corrmatrix(3,:),'MarkerFaceColor',[0.0,0.0,0.9],'MarkerEdgeColor','none','MarkerFaceAlpha',.3), hold on, plot([0,1],[0,1]), xlabel('Stim vs Control'), ylabel('Stim vs PostStim'),xlim([0,1]),ylim([0,1])
        diagol_below = 100*sum(corrmatrix(1,:)>corrmatrix(3,:))/size(corrmatrix,2);
        diagol_above = 100*sum(corrmatrix(1,:)<corrmatrix(3,:))/size(corrmatrix,2);
        text(0.01,1.04,[num2str(round(diagol_above)),'%'],'Units','normalized')
        text(1.04,0.01,[num2str(round(diagol_below)),'%'],'Units','normalized','Rotation',90)
        %scatterDiagHist(corrmatrix(1,:),corrmatrix(3,:),20)
        subplot(2,3,5)
        scatter(corrmatrix(2,:),corrmatrix(3,:),'MarkerFaceColor',[0.0,0.0,0.9],'MarkerEdgeColor','none','MarkerFaceAlpha',.3), hold on, plot([0,1],[0,1]), xlabel('Control vs PostStim'), ylabel('Stim vs PostStim'),xlim([0,1]),ylim([0,1])
        diagol_below = 100*sum(corrmatrix(2,:)>corrmatrix(3,:))/size(corrmatrix,2);
        diagol_above = 100*sum(corrmatrix(2,:)<corrmatrix(3,:))/size(corrmatrix,2);
        text(0.01,1.04,[num2str(round(diagol_above)),'%'],'Units','normalized')
        text(1.04,0.01,[num2str(round(diagol_below)),'%'],'Units','normalized','Rotation',90)
        %scatterDiagHist(corrmatrix(2,:),corrmatrix(3,:),20)
        subplot(2,3,6)
        boxplot([(corrmatrix(1,:))',(corrmatrix(2,:))',(corrmatrix(3,:))']), hold on
        xticks([1,2,3]), xticklabels({'Control vs stim','Control vs PostStim','Stim vs PostStim'}), ylabel('Correlation'), ylim([-0.2,1]), grid on
        xtickangle(45)
        figure(101), hold on
        mean_velocity = nanmedian([(corrmatrix(1,:))',(corrmatrix(2,:))',(corrmatrix(3,:))']);
        std_velocity = nanstd([(corrmatrix(1,:))',(corrmatrix(2,:))',(corrmatrix(3,:))']);
        H = bar([2,4,6],mean_velocity,0.2)
        H(1).FaceColor = 'b'; % set the colour of one bar
        errorbar([2,4,6],mean_velocity,std_velocity,'.')
        xticks([2,4,6]), xticklabels({'Control vs stim','Control vs PostStim','Stim vs PostStim'}), ylabel('Correlation'), ylim([-0.2,1]), grid on
        xtickangle(45)
        
    elseif m == 2
        figure(100),
        subplot(2,3,1)
        scatter(corrmatrix(1,:),corrmatrix(3,:),'MarkerFaceColor',[0.1,0.1,0.1],'MarkerEdgeColor','none','MarkerFaceAlpha',.3), hold on, plot([0,1],[0,1]), xlabel('NoStim vs Control'), ylabel('NoStim vs PostPostNoStim'),xlim([0,1]),ylim([0,1])
        diagol_below = 100*sum(corrmatrix(1,:)>corrmatrix(3,:))/size(corrmatrix,2);
        diagol_above = 100*sum(corrmatrix(1,:)<corrmatrix(3,:))/size(corrmatrix,2);
        text(0.01,1.04,[num2str(round(diagol_above),2),'%'],'Units','normalized')
        text(1.04,0.01,[num2str(round(diagol_below),2),'%'],'Units','normalized','Rotation',90)
        %scatterDiagHist(corrmatrix(1,:),corrmatrix(3,:),20)
        subplot(2,3,2)
        scatter(corrmatrix(2,:),corrmatrix(3,:),'MarkerFaceColor',[0.1,0.1,0.1],'MarkerEdgeColor','none','MarkerFaceAlpha',.3), hold on, plot([0,1],[0,1]), xlabel('Control vs PostNoStim'), ylabel('NoStim vs PostNoStim'),xlim([0,1]),ylim([0,1])
        diagol_below = 100*sum(corrmatrix(2,:)>corrmatrix(3,:))/size(corrmatrix,2);
        diagol_above = 100*sum(corrmatrix(2,:)<corrmatrix(3,:))/size(corrmatrix,2);
        text(0.01,1.04,[num2str(round(diagol_above),2),'%'],'Units','normalized')
        text(1.04,0.01,[num2str(round(diagol_below),2),'%'],'Units','normalized','Rotation',90)
        %scatterDiagHist(corrmatrix(2,:),corrmatrix(3,:),20)
        subplot(2,3,3)
        boxplot([(corrmatrix(1,:))',(corrmatrix(2,:))',(corrmatrix(3,:))']), hold on
        xticks([1,2,3]), xticklabels({'Control vs NoStim','Control vs PostNoStim','Stim vs PostNoStim'}), ylabel('Correlation'), ylim([-0.2,1]), grid on
        xtickangle(45)
        figure(101), hold on
        mean_velocity = nanmedian([(corrmatrix(1,:))',(corrmatrix(2,:))',(corrmatrix(3,:))']);
        std_velocity = nanstd([(corrmatrix(1,:))',(corrmatrix(2,:))',(corrmatrix(3,:))']);
        H = bar([1,3,5],mean_velocity,0.2)
        H(1).FaceColor = [0.3 0.3 0.3]; % set the colour of one bar
        errorbar([1,3,5],mean_velocity,std_velocity,'.')
        xticks([1,3,5]), xticklabels({}), ylabel('Correlation'), ylim([-0.2,1]), grid on
        xticks([1:6]), xticklabels({'Control vs NoStim','Control vs stim','Control vs PostNoStim', 'Control vs PostStim','Stim vs PostNoStim','Stim vs PostStim'}), ylabel('Correlation'), ylim([-0.2,1]), grid on
        xtickangle(45)
        
    end

    figure(220+m),
    yyaxis left
    for j = 1:size(firing_rate_map.unit,3)
        placefields_binary_hist(:,j) = sum([placecell_states(find([placecell_states(:,j).condition]),j).placefield_state]');
    plot(firing_rate_map.x_bins,placefields_binary_hist(:,j)), hold on
    end
    xlabel('Position (cm)'), title('Position of binary place fields'), legend({'Pre','Stim','PostStim'})
    figure
    plot(firing_rate_map.x_bins,placefields_binary_hist(:,2)-placefields_binary_hist(:,1)), hold on
    plot(firing_rate_map.x_bins,placefields_binary_hist(:,3)-placefields_binary_hist(:,2))
    plot(firing_rate_map.x_bins,placefields_binary_hist(:,3)-placefields_binary_hist(:,1))
    xlabel('Position (cm)'), title('Differente in position of binary place fields')
    figure(220+m), yyaxis right
%     plot(firing_rate_map.x_bins,nansum(stim_pos_linearized_all)/sum(nansum(stim_pos_linearized_all)),'k'), hold on
%     [N,edges] = histcounts(reward_points_linearized(:), firing_rate_map.x_bins,'Normalization','probability')
%     stairs(edges(1:end-1)+mean(diff(edges))./2,N,'--k'),
%     legend({'Pre','Cooling','PostStim','StimLocation','RewardLocation'})
%     if m == 1
%         plot(firing_rate_map.x_bins,sum(stim_pos_linearized_waveform_all),'r'), legend({'Pre','Stim','PostStim','StimLocation','RewardLocation','StimLocation2'})
%     end
    
    % Cell metrics differences
    temp = fieldnames(cell_metrics_batch);
    temp3 = struct2cell(structfun(@class,cell_metrics_batch,'UniformOutput',false));
%     subindex = intersect(find(~contains(temp3',{'cell','struct'})), find(~contains(temp,{'TruePositive','FalsePositive','PutativeConnections','ACG','ACG2'})));
    subindex = intersect(find(~contains(temp3',{'cell','struct'})), find(~contains(temp,{'truePositive','falsePositive','putativeConnections','acg','acg2','placefield_peak_rate','placecell_stability','optoPSTH','place_cell','firing_rate_map','isolationDistance','LRatio','maxChannel','peakVoltage','batchIDs','cv2','cellID','firingRateISI','refractoryPeriodViolation','rippleCorrelogram','sessionID','spikeCount','spikeGroup','spikeSortingID','spikeWaveforms','spikeWaveforms_std','derivative_TroughtoPeak','firing_rate_map','ripplePeakDelay','synapticConnectionsIn','synapticConnectionsOut'})));
    temp1 = intersect(find(strcmp(cell_metrics_batch.placeCellStability,'Stable')),find(contains(cell_metrics_batch.putativeCellType,'Pyramidal')));
    temp2 = intersect(find(strcmp(cell_metrics_batch.placeCellStability,'Instable')),find(contains(cell_metrics_batch.putativeCellType,'Pyramidal')));
    
    [labels2,I]= sort(temp(subindex));
    for j = 1:length(labels2)
        jj = labels2{j};
        if sum(isnan(cell_metrics_batch.(jj)(temp1))) < length(temp1) && sum(isnan(cell_metrics_batch.(jj)(temp2))) < length(temp2)
            [h,p] = kstest2(cell_metrics_batch.(jj)(temp1),cell_metrics_batch.(jj)(temp2));
            cell_metrics_effects(j,m)= p;
            cell_metrics_effects2(j,m)= h;
        else
            cell_metrics_effects(j,m)= 0;
            cell_metrics_effects2(j,m)= 0;
        end
    end
    
end

image2 = log10(cell_metrics_effects);
cell_metrics_effects3 = cell_metrics_effects2;
image2( intersect(find(~cell_metrics_effects3(:,1)), find(image2(:,1)<log10(0.05))) ,1) = -image2( intersect(find(~cell_metrics_effects3(:,1)), find(image2(:,1)<log10(0.05))) ,1);
image2( intersect(find(~cell_metrics_effects3(:,2)), find(image2(:,2)<log10(0.05))) ,2) = -image2( intersect(find(~cell_metrics_effects3(:,2)), find(image2(:,2)<log10(0.05))) ,2);

figure('pos',[10 10 500 800])
imagesc(image2),colormap(jet),colorbar, hold on
if ~isempty(find(cell_metrics_effects(:,1)<0.05))
plot(1,find(cell_metrics_effects(:,1)<0.05),'xw','linewidth',2)
end
if ~isempty(find(cell_metrics_effects(:,2)<0.05))
plot(2,find(cell_metrics_effects(:,2)<0.05),'xw','linewidth',2)
end
if sum(cell_metrics_effects(:,1)<0.003)
plot(1.1,find(cell_metrics_effects(:,1)<0.003),'xw','linewidth',2)
end
if sum(cell_metrics_effects(:,2)<0.003)
plot(2.1,find(cell_metrics_effects(:,2)<0.003),'xw','linewidth',2)
end
xticks([1,2]), 
xticklabels({'Stim', 'Control'})
yticks(1:length(labels2))
yticklabels(labels2)
set(gca,'TickLabelInterpreter','none')
caxis([-3,3]);

% subplot(2,3,2)
% scatterDiagHist(corrmatrix3{1}(2,:),corrmatrix3{1}(3,:),20)
% subplot(2,3,5)
% scatterDiagHist(corrmatrix3{2}(2,:),corrmatrix3{2}(3,:),20)

figure(200), bar([1,2,3,4],placefield_changed),
xticks([1,2,3,4]), xticklabels({'emerge during stim', 'emerge during poststim', 'disappear during stim', 'disappear during poststim'})
ylim([0,0.2]), legend({['Stim (', num2str(length(datasets_collection{1})),' sessions)'],['NoStim (', num2str(length(datasets_collection{2})),' sessions)']}), title('Placefield emergence & disappearence')

kstest01 = corrmatrix3{1}(1,:)'-corrmatrix3{1}(3,:)';
kstest02 = corrmatrix3{2}(1,:)'-corrmatrix3{2}(3,:)';
kstest03 = corrmatrix3{1}(2,:)'-corrmatrix3{1}(3,:)';
kstest04 = corrmatrix3{2}(2,:)'-corrmatrix3{2}(3,:)';
[h11,p1] = kstest(kstest01(~isnan(kstest01))')
[h12,p2] = kstest(kstest02(~isnan(kstest02))')
[h13,p3] = kstest2(kstest01(~isnan(kstest01))',kstest02(~isnan(kstest02))')

figure,
subplot(2,1,1)
plot(xlimits,(h2)), hold on
legend({'Control vs Stim','Control vs PostStim','Stim vs PostStim','Control vs NoStim','Control vs PostNoStim','NoStim vs PostNoStim'}),title('Correlation coeff across population')
subplot(2,1,2),
for m = 1:size(corrmatrix2,2)
    for i = 1:size(corrmatrix2{m},2)
        cdfplot(corrmatrix2{m}(:,i)), hold on
    end
end
figure,
xbins = [-1:0.05:1];
xbins1 = xbins(1:end-1);
hist_1 = histogram(kstest01(~isnan(kstest01)),xbins,'Normalization','pdf'); hold on
hist_2 = histogram(kstest02(~isnan(kstest02)),xbins,'Normalization','pdf');
hist_3 = histogram(kstest03(~isnan(kstest03)),xbins,'Normalization','pdf');
hist_4 = histogram(kstest04(~isnan(kstest04)),xbins,'Normalization','pdf');

figure,
plot(xbins1,hist_1.Values-hist_2.Values), hold on
%plot(xbins1,hist_2.Values)
plot(xbins1,hist_3.Values-hist_4.Values)
%plot(xbins1,hist_4.Values)
legend({'Stim vs (Control and Post)','NoStim vs (Control and Post)','PostStim vs (Control and Stim)','PostNoStim vs (Control and NoStim)'})

figure
subplot(2,1,1),hold on
plot([1,2,3,4]-0.1,placefield_changed2{2}','.','color',[0.3 0.3 0.3],'markers',15), xlim([0,5])
plot([1,2,3,4]+0.1,placefield_changed2{1}','.b','markers',15)
xticks([1,2,3,4]), xticklabels({'emerge during stim', 'emerge during poststim', 'disappear during stim', 'disappear during poststim'})
title('Placefield emergence & disappearence')
subplot(2,1,2)
H = bar([1,2,3,4]-0.1,[mean(placefield_changed2{2});mean(placefield_changed2{1})]'), hold on, xlim([0,5])
H(2).FaceColor = 'b'; % set the colour of one bar
H(1).FaceColor = [0.3 0.3 0.3]; % set the colour of one bar
xticks([1,2,3,4]), xticklabels({'emerge during stim', 'emerge during poststim', 'disappear during stim', 'disappear during poststim'})
errorbar([1,2,3,4]-0.25,[mean(placefield_changed2{2})]',std(placefield_changed2{2}),'o','color',[0.1 0.1 0.1])
errorbar([1,2,3,4]+0.05,[mean(placefield_changed2{1})]',std(placefield_changed2{1}),'o','color',[0.1 0.1 0.1])
[h21,p21] = kstest2(placefield_changed2{1}(:,1),placefield_changed2{2}(:,1));
[h22,p22] = kstest2(placefield_changed2{1}(:,2),placefield_changed2{2}(:,2));
[h23,p23] = kstest2(placefield_changed2{1}(:,3),placefield_changed2{2}(:,3));
[h24,p24] = kstest2(placefield_changed2{1}(:,4),placefield_changed2{2}(:,4));
text(0.8,-0.001,['p=',num2str(p21)]),text(1.8,-0.001,['p=',num2str(p22)]),text(2.8,-0.001,['p=',num2str(p23)]),text(3.8,-0.001,['p=',num2str(p24)])

%% % Behavioral effects across sessions
disp('second part')
clear all
ham11_datasets = [8,10,11,12,13,14,15];
ham12_datasets = [4,5,6,7,16,17,19:21,23,25:27]; % 22,24

ham5

ham5_datasets = [1]; % ham5_769_batch
ham8_datasets = [2]; % ham8_191-192_amp
ham21_datasets = [32,39,42,43,44,45,46]; % ham21_33-35_amp,  ham21_106-108_amp, ham21_93, ham21_96, ham21_103, ham21_118, ham21_124
ham21_datasets_shams = [41,48]; % 31:ham21_27-29_amp, ham21_54_sham, ham21_28_sham
ham21_datasets_control = [37,38,40]; % ham21_98-100_amp, ham21_18-22_amp, ham21_109-111_amp
% Mixed session = 33:ham21_66-68_amp

datasets = [ham11_datasets,ham12_datasets,ham21_datasets];
Viktors_Datasets

trials_errors = [];
for k = 1:length(datasets)
    id = datasets(k);
    cd(recordings(id).folder)
    recording = recordings(id);
    
    maze = recordings(id).maze;
    maze.polar_rho_limits = [22,40];
    maze.polar_theta_limits = [-2.7,2.7]*maze.radius_in;
    maze.pos_x_limits = [-8,8];
    maze.pos_y_limits = [-22,24];
    
    maze.boundary{1} = [0,20];
    maze.boundary{2} = [0,10];
    maze.boundary{3} = [maze.pos_x_limits(1),25];
    maze.boundary{4} = [maze.pos_x_limits(2),25];
    maze.boundary{5} = [maze.radius_in,maze.polar_theta_limits(2)];
    
    Intan_rec_info = read_Intan_RHD2000_file_Peter(recording.folder);
    sr = Intan_rec_info.frequency_parameters.amplifier_sample_rate;
    recording.sr = sr;
    recording.sr_lfp = sr/16;
    recording.nbChan = size(Intan_rec_info.amplifier_channels,2);
    nbChan = size(Intan_rec_info.amplifier_channels,2);
    ch_theta = recording.ch_theta;
    load('digitalchannels.mat');
    load('optogenetics.mat');
    load('optitrack.mat');
    
    temp_ = dir([recording.name,'.dat']);
    % behaviortime = temp_.bytes/nbChan/sr/2;
    
    offset = recording.optiTrack_offset;
    optitrack.position3D = optitrack.position3D - offset';
    pos_outside = find(optitrack.position3D(1,:)< -45 | optitrack.position3D(1,:) > 45 | optitrack.position3D(2,:) < -60 | optitrack.position3D(2,:) > 45);
    optitrack.position3D(:,pos_outside) = nan;
    pos_acce = find(optitrack.FrameRate*sqrt(sum((diff(optitrack.position3D').^2),2))' > 100);
    optitrack.position3D(:,pos_acce+1) = nan;
    optitrack.position3D(:,pos_acce) = nan;
    gausswin_size = optitrack.FrameRate/4;
    for i = 1:3
        optitrack.position3D(i,:) = medfilt1(optitrack.position3D(i,:),11,'omitnan');
        optitrack.position3D(i,:) = nanconv(optitrack.position3D(i,:),gausswin(gausswin_size)','edge');
    end
    
    inputs.ch_opto = recording.ch_opto;
    
    
    disp('Aligning tracking with electrophysiology')
    if recording.prebehaviortime~=0
        temp_ = dir(recording.prebehaviortime_file);
        prebehaviortime2 = temp_.bytes/sr/2/2;
        prebehaviortime = 0;
    else
        prebehaviortime2 = 0;
    end
    
    animal = [];
    for fn = fieldnames(maze)'
        animal.(fn{1}) = maze.(fn{1});
    end
    ch_optiTrack_sync = recording.ch_optiTrack_sync;
    animal.pos = optitrack.position3D;
    animal.speed  = [optitrack.FrameRate*sqrt(sum((diff(optitrack.position3D').^2),2))',0]; %Optitrack.FrameRate*diff(Optitrack.position3D);
    animal.speed_th = 5;
    animal.sr = optitrack.FrameRate;
    animal.time = digital_on{ch_optiTrack_sync}'/sr;
    animal.time = animal.time(find(animal.time>prebehaviortime2));
    if ~isempty(recording.useOptitrackTime)
        animal.time = optitrack.Time'+animal.time(1);
    elseif ~isempty(recording.useOptitrackTimeReversed)
        animal.time = animal.time(length(animal.time)-length(optitrack.Time)+1:end);
    else
        animal.time = animal.time(1:length(optitrack.Time));
    end
    
    [animal.polar_theta,animal.polar_rho] = cart2pol(animal.pos(2,:),animal.pos(1,:));
    animal.polar_theta = animal.polar_theta*maze.radius_in;
    
    animal.circularpart = find(animal.polar_theta > animal.polar_theta_limits(1) & animal.polar_theta < animal.polar_theta_limits(2) & animal.polar_rho > animal.polar_rho_limits(1) & animal.polar_rho < animal.polar_rho_limits(2));
    animal.centralarmpart = find(animal.pos(1,:) > animal.pos_x_limits(1) & animal.pos(1,:) < animal.pos_x_limits(2) & animal.pos(2,:) > animal.pos_y_limits(1) & animal.pos(2,:) < animal.pos_y_limits(2));
    animal.arm = double(animal.pos(1,:) > animal.pos_x_limits(1) & animal.pos(1,:) < animal.pos_x_limits(2) & animal.pos(2,:) > animal.pos_y_limits(1) & animal.pos(2,:) < animal.pos_y_limits(2));
    animal.rim = double((animal.polar_rho > animal.polar_rho_limits(1) & animal.polar_rho < animal.polar_rho_limits(2) & animal.polar_theta > animal.polar_theta_limits(1) & animal.polar_theta < animal.polar_theta_limits(2)));
    animal.error_trials = zeros(1,size(animal.pos,2));
    animal.maze = maze;
    animal.state_labels = {'No stim','Stim'};
    animal.pos_linearized = linearize_pos(animal,recording.arena);
    animal.pos_linearized_limits = [0,diff(animal.pos_y_limits) + diff(animal.polar_theta_limits)];
    
    if isempty(optogenetics)
        left = animal.time(find((diff(animal.pos(2,:) > 12 & animal.pos(1,:) < -20)) == 1 ));
        right = animal.time(find((diff(animal.pos(2,:) > 12 & animal.pos(1,:) > 20)) == 1 ));
        center = animal.time(find((diff(animal.pos(2,:) > 12 & animal.pos(1,:) < 10 & animal.pos(1,:) > -10)) == 1 ));
        optogenetics.peak = sort([left(round(end/2):end),right(round(end/2):end),center(round(end/2):end)]);
        clear left right center
    end
    % Plots
    figure,
    plot_ThetaMaze(maze)
    axis equal
    plot3(optitrack.position3D(1,:),optitrack.position3D(2,:),optitrack.position3D(3,:),'.-')
    xlim([-45,45]),ylim([-60,45])
    
    optogenetics.pos = interp1(animal.time,animal.pos',optogenetics.peak)';
    optogenetics.polar_rho = interp1(animal.time,animal.polar_rho',optogenetics.peak);
    optogenetics.polar_theta = interp1(animal.time,animal.polar_theta',optogenetics.peak);
    optogenetics.pos_linearized = interp1(animal.time,animal.pos_linearized',optogenetics.peak);
    
    optogenetics.rim = (optogenetics.polar_rho > animal.polar_rho_limits(1) & optogenetics.polar_rho < animal.polar_rho_limits(2) & optogenetics.polar_theta > animal.polar_theta_limits(1) & optogenetics.polar_theta < animal.polar_theta_limits(2));
    % optogenetics.arm = (optogenetics.pos(1,:) > animal.pos_x_limits(1) & optogenetics.pos(1,:) < animal.pos_x_limits(2) & optogenetics.pos(2,:) > animal.pos_y_limits(1) & optogenetics.pos(2,:) < animal.pos_y_limits(2));
    % optogenetics.left = (optogenetics.polar_rho > animal.polar_rho_limits(1) & optogenetics.polar_rho < animal.polar_rho_limits(2) & optogenetics.polar_theta > animal.polar_theta_limits(1) & optogenetics.polar_theta < 0);
    % optogenetics.right = (optogenetics.polar_rho > animal.polar_rho_limits(1) & optogenetics.polar_rho < animal.polar_rho_limits(2) & optogenetics.polar_theta > 0 & optogenetics.polar_theta < animal.polar_theta_limits(2));
    
    optogenetics.arm = (optogenetics.pos(1,:) > animal.pos_x_limits(1)   & optogenetics.pos(2,:) > animal.pos_y_limits(1)/2 & optogenetics.pos(1,:) < animal.pos_x_limits(2) );
    optogenetics.left = (optogenetics.pos(1,:) < animal.pos_x_limits(1)  & optogenetics.pos(2,:) > animal.pos_y_limits(1));
    optogenetics.right = (optogenetics.pos(1,:) > animal.pos_x_limits(2) & optogenetics.pos(2,:) > animal.pos_y_limits(1));
    
    figure,
    plot(optogenetics.pos(1,:),optogenetics.pos(2,:),'ok'), hold on, plot_ThetaMaze(maze)
    plot(optogenetics.pos(1,optogenetics.arm),optogenetics.pos(2,optogenetics.arm),'or')
    plot(optogenetics.pos(1,optogenetics.left),optogenetics.pos(2,optogenetics.left),'ob')
    plot(optogenetics.pos(1,optogenetics.right),optogenetics.pos(2,optogenetics.right),'om')
    
    
    % Separating left and right trials
    [trials,animal,optogenetics] = trials_thetamaze(animal, maze, optogenetics,[],0);
    
    optogenetics.trials(optogenetics.trials==0) = nan;
    trials.labels = {'Left','Right'};
    trials.total = length(trials.start);
    
    figure,
    subplot(2,1,1)
    stem(find(trials.optogenetics),trials.stat(find(trials.optogenetics)),'r'), hold on
    stem(find(~trials.optogenetics),trials.stat(find(~trials.optogenetics)),'b')
    yticks([1, 2]), yticklabels(trials.labels),xlabel('Trials'),title('Trials '),axis tight
    subplot(2,1,2)
    error1 = find([0,diff(trials.stat)==0]  & trials.optogenetics);
    bar(1,100*length(error1)/sum(trials.optogenetics),'r'), hold on
    error2 = find([0,diff(trials.stat)==0]  & ~trials.optogenetics);
    bar(2, 100*length(error2)/sum(~trials.optogenetics),'b')
    xticks([1, 2]), xticklabels({'Stim trials','Sham trials'}),ylabel('Percentage of errors'),title('Error trials (%)'),ylim([0,100]),xlim([0.5,2.5])
    
    trials_errors(:,k) = [100*length(error1)/sum(trials.optogenetics),100*length(error2)/sum(~trials.optogenetics)];
end

figure,
subplot(2,1,1)
plot([1,2], trials_errors,'.-','markersize',8), hold on
xticks([1, 2]), xticklabels({'Stim trials','Sham trials'}),ylabel('Percentage of errors'),title('Error trials (%)'),axis tight,
xlim([0.5,2.5]),ylim([0,100])

subplot(2,1,2)
boxplot(trials_errors')
xticks([1, 2]), xticklabels({'Stim trials','Sham trials'}),ylabel('Percentage of errors')
[h,p] = ttest(trials_errors(:,1),trials_errors(:,2))
if h == 1
    title(['Paired t-test: Different with  p = ', num2str(p)])
else
    title(['Paired t-test: Indifferent with  p = ', num2str(p)])
end

%% % Theta and gamme effect by stimulation across sessions
clear all
ham_datasets = [1,2,3,4,5,7,8];
ham11_datasets = [3,8,10,11,12,13,14,15];
ham12_datasets = [4,5,6,7,16,17,19:21,23,25:27]; % 22,24 
datasets_stim = [1,4,5,6,7,32,33,35,39,49,63,64];
datasets = datasets_stim;
Viktors_Datasets
colorschema = [0,0,1;0,1,0;1,0,0;1,1,0;0,1,1;1,0,1;0.5,0.5,0.5];

freqband = 'gamma';
stats = [];
for k = 1:length(datasets)
   
    id = datasets(k);
    cd(recordings(id).folder)
    recording = recordings(id);
    maze = recordings(id).maze;
    Intan_rec_info = read_Intan_RHD2000_file_Peter(recording.folder);
    sr = Intan_rec_info.frequency_parameters.amplifier_sample_rate;
    recording.sr = sr;
    recording.sr_lfp = sr/16;
    recording.nbChan = size(Intan_rec_info.amplifier_channels,2);
    nbChan = size(Intan_rec_info.amplifier_channels,2);
    ch_theta = recording.ch_theta;
    
    load('digitalchannels.mat');
    load('optogenetics.mat');
    load('optitrack.mat');
    offset = recording.optiTrack_offset;
    optitrack.position3D = optitrack.position3D - offset';
    pos_outside = find(optitrack.position3D(1,:)< -45 | optitrack.position3D(1,:) > 45 | optitrack.position3D(2,:) < -60 | optitrack.position3D(2,:) > 45);
    optitrack.position3D(:,pos_outside) = nan;
    pos_acce = find(optitrack.FrameRate*sqrt(sum((diff(optitrack.position3D').^2),2))' > 100);
    optitrack.position3D(:,pos_acce+1) = nan;
    optitrack.position3D(:,pos_acce) = nan;
    gausswin_size = optitrack.FrameRate/4;
    ch_optiTrack_sync = recording.ch_optiTrack_sync;
    optiTrackFile = recording.optiTrackFile;
    
    prebehaviortime = recording.prebehaviortime;
    if prebehaviortime~=0
        temp_ = dir(recording.prebehaviortime_file);
        prebehaviortime2 = temp_.bytes/sr/2/2;
        prebehaviortime = 0;
    else
        prebehaviortime2 = 0;
    end
    animal = [];
    animal.pos = optitrack.position3D;
    animal.speed  = [optitrack.FrameRate*sqrt(sum((diff(optitrack.position3D').^2),2))',0]; %Optitrack.FrameRate*diff(Optitrack.position3D);
    animal.speed_th = 5;
    animal.sr = optitrack.FrameRate;
    animal.time = digital_on{ch_optiTrack_sync}'/sr;
    animal.time = animal.time(find(animal.time>prebehaviortime2));
    animal.time = animal.time(1:size(optitrack.position3D,2));
    [animal.polar_theta,animal.polar_rho] = cart2pol(animal.pos(2,:),animal.pos(1,:));
    animal.polar_theta = animal.polar_theta*maze.radius_in;
    animal.polar_rho_limits = [20,40];
    animal.polar_theta_limits = [-2.8,2.8]*maze.radius_in;
    animal.pos_x_limits = [-7,7];
    animal.pos_y_limits = [-23,24];
    animal.circularpart = find(animal.polar_theta > animal.polar_theta_limits(1) & animal.polar_theta < animal.polar_theta_limits(2) & animal.polar_rho > animal.polar_rho_limits(1) & animal.polar_rho < animal.polar_rho_limits(2));
    animal.centralarmpart = find(animal.pos(1,:) > animal.pos_x_limits(1) & animal.pos(1,:) < animal.pos_x_limits(2) & animal.pos(2,:) > animal.pos_y_limits(1) & animal.pos(2,:) < animal.pos_y_limits(2));
    animal.arm = double(animal.pos(1,:) > animal.pos_x_limits(1) & animal.pos(1,:) < animal.pos_x_limits(2) & animal.pos(2,:) > animal.pos_y_limits(1) & animal.pos(2,:) < animal.pos_y_limits(2));
    animal.rim = double((animal.polar_rho > animal.polar_rho_limits(1) & animal.polar_rho < animal.polar_rho_limits(2) & animal.polar_theta > animal.polar_theta_limits(1) & animal.polar_theta < animal.polar_theta_limits(2)));
    animal.error_trials = zeros(1,size(animal.pos,2));
    animal.maze = maze;
    animal.state_labels = {'No stim','Stim'};
    
    optogenetics.pos = interp1(animal.time,animal.pos',optogenetics.peak)';
    optogenetics.polar_rho = interp1(animal.time,animal.polar_rho',optogenetics.peak);
    optogenetics.polar_theta = interp1(animal.time,animal.polar_theta',optogenetics.peak);
    optogenetics.rim = (optogenetics.polar_rho > animal.polar_rho_limits(1) & optogenetics.polar_rho < animal.polar_rho_limits(2) & optogenetics.polar_theta > animal.polar_theta_limits(1) & optogenetics.polar_theta < animal.polar_theta_limits(2));
    % optogenetics.arm = (optogenetics.pos(1,:) > animal.pos_x_limits(1) & optogenetics.pos(1,:) < animal.pos_x_limits(2) & optogenetics.pos(2,:) > animal.pos_y_limits(1) & optogenetics.pos(2,:) < animal.pos_y_limits(2));
    % optogenetics.left = (optogenetics.polar_rho > animal.polar_rho_limits(1) & optogenetics.polar_rho < animal.polar_rho_limits(2) & optogenetics.polar_theta > animal.polar_theta_limits(1) & optogenetics.polar_theta < 0);
    % optogenetics.right = (optogenetics.polar_rho > animal.polar_rho_limits(1) & optogenetics.polar_rho < animal.polar_rho_limits(2) & optogenetics.polar_theta > 0 & optogenetics.polar_theta < animal.polar_theta_limits(2));
    
    optogenetics.arm = (optogenetics.pos(1,:) > animal.pos_x_limits(1)   & optogenetics.pos(2,:) > animal.pos_y_limits(1)/2 & optogenetics.pos(1,:) < animal.pos_x_limits(2) );
    optogenetics.left = (optogenetics.pos(1,:) < animal.pos_x_limits(1)  & optogenetics.pos(2,:) > animal.pos_y_limits(1));
    optogenetics.right = (optogenetics.pos(1,:) > animal.pos_x_limits(2) & optogenetics.pos(2,:) > animal.pos_y_limits(1));
    
    % Separating left and right trials
    maze.boundary{1} = [0,20];
    maze.boundary{2} = [0,10];
    maze.boundary{3} = [-15,25];
    maze.boundary{4} = [15,25];
    maze.boundary{5} = [maze.radius_in,150/180*pi*maze.radius_in];
    
    plots = 0;
    [trials,animal,optogenetics] = trials_thetamaze(animal, maze, optogenetics,[],plots);
    optogenetics.trials(optogenetics.trials==0) = nan;
    trials.labels = {'Left','Right'};
    trials.total = length(trials.start);
    
    optogenetics.artificial = [];
    optogenetics.artificial.threshold.arm = mean(optogenetics.pos(2,optogenetics.arm));
    optogenetics.artificial.threshold.left = mean(optogenetics.polar_theta(optogenetics.left));
    optogenetics.artificial.threshold.right = mean(optogenetics.polar_theta(optogenetics.right));
    optogenetics.artificial.peak = [];
    optogenetics.artificial.loc = [];
    for i = 1:length(trials.optogenetics)
        if trials.optogenetics(i) == 0
            interval = trials.start(i):trials.end(i);
            test1 = find(diff(animal.pos(2,interval) > optogenetics.artificial.threshold.arm)==1);
            test2 = find(diff(animal.polar_theta(interval) < optogenetics.artificial.threshold.left)==1);
            test3 = find(diff(animal.polar_theta(interval) > optogenetics.artificial.threshold.right)==1);
            optogenetics.artificial.loc = [optogenetics.artificial.loc,ones(1,length(test1)), ones(1,length(test2))*2, ones(1,length(test3))*3];
            optogenetics.artificial.peak = [optogenetics.artificial.peak, animal.time(interval([test1,test2,test3]))];
        end
    end
    
    optogenetics.artificial.pos = interp1(animal.time+prebehaviortime,animal.pos',optogenetics.artificial.peak)';
    optogenetics.artificial.polar_rho = interp1(animal.time+prebehaviortime,animal.polar_rho',optogenetics.artificial.peak);
    optogenetics.artificial.polar_theta = interp1(animal.time+prebehaviortime,animal.polar_theta',optogenetics.artificial.peak);
    optogenetics.artificial.rim = (optogenetics.artificial.polar_rho > animal.polar_rho_limits(1) & optogenetics.artificial.polar_rho < animal.polar_rho_limits(2) & optogenetics.artificial.polar_theta > animal.polar_theta_limits(1) & optogenetics.artificial.polar_theta < animal.polar_theta_limits(2));
    optogenetics.artificial.arm = (optogenetics.artificial.pos(1,:) > animal.pos_x_limits(1) & optogenetics.artificial.pos(1,:) < animal.pos_x_limits(2) & optogenetics.artificial.pos(2,:) > animal.pos_y_limits(1) & optogenetics.artificial.pos(2,:) < animal.pos_y_limits(2));
    optogenetics.artificial.left = (optogenetics.artificial.polar_rho > animal.polar_rho_limits(1) & optogenetics.artificial.polar_rho < animal.polar_rho_limits(2) & optogenetics.artificial.polar_theta > animal.polar_theta_limits(1) & optogenetics.artificial.polar_theta < 0);
    optogenetics.artificial.right = (optogenetics.artificial.polar_rho > animal.polar_rho_limits(1) & optogenetics.artificial.polar_rho < animal.polar_rho_limits(2) & optogenetics.artificial.polar_theta > 0 & optogenetics.artificial.polar_theta < animal.polar_theta_limits(2));
    
    switch freqband
        case 'theta'
            freqlist = [4:0.025:10];
            Fpass = [4,10];
            sr_theta = animal.sr;
            caxis_range = [0,1.];
        case 'gamma'
            freqlist = [30:5:100];
            Fpass = [30,100];
            sr_theta = 400;
            caxis_range = [0,0.45];
    end
    signal = 0.000050354 * double(LoadBinary([recording.name '.lfp'],'nChannels',recording.nbChan,'channels',recording.ch_theta,'precision','int16','frequency',recording.sr_lfp)); % ,'start',start,'duration',duration
    
    running_window = sr_theta*3/4;
    signal2 = resample(signal,sr_theta,recording.sr_lfp);
    clear signal
    theta_samples_pre = 2*sr_theta;
    theta_samples_post = 2*sr_theta;
    window_time = [-theta_samples_pre:theta_samples_post]/sr_theta;
    window_stim = [theta_samples_pre-sr_theta/2:theta_samples_pre+sr_theta/2];
    window_prestim = [1:theta_samples_pre-sr_theta/2-1];
    Wn_theta = [Fpass(1)/(sr_theta/2) Fpass(2)/(sr_theta/2)]; % normalized by the nyquist frequency
    [btheta,atheta] = butter(3,Wn_theta);
    signal_filtered = filtfilt(btheta,atheta,signal2);
    %[wt,~,~] = awt_freqlist(signal_filtered,sr_temperature,freqlist);
    %wt2 = abs(wt)'; clear wt
    wt = spectrogram(signal_filtered,running_window,running_window-1,freqlist,sr_theta);
    wt2 = [zeros(length(freqlist),running_window/2-1),abs(wt), zeros(length(freqlist),running_window/2)]; clear wt
    
    % Opto stim
    opto_peaks1 = round(sr_theta*optogenetics.peak(find(optogenetics.peak > animal.time(1) & optogenetics.peak < animal.time(end))));
    theta_triggered1 = [];
    for i = 1:length(opto_peaks1)
        theta_triggered1(:,:,i) = wt2(:,opto_peaks1(i)-theta_samples_pre:opto_peaks1(i)+theta_samples_post);
    end
    
    % Artificial stim
    opto_peaks2 = round(sr_theta*optogenetics.artificial.peak(find(optogenetics.artificial.peak > animal.time(1) & optogenetics.artificial.peak < animal.time(end))));
    theta_triggered2 = [];
    for i = 1:length(opto_peaks2)
        theta_triggered2(:,:,i) = wt2(:,opto_peaks2(i)-theta_samples_pre:opto_peaks2(i)+theta_samples_post);
    end
    
    % Figure for all sessions
    figure,
    subplot(3,2,1)  % Opto stimulated trials
    imagesc(window_time,freqlist,mean(theta_triggered1,3)), set(gca,'YDir','normal'), title(['Stim trials ' recording.name]), xlabel('Time (s)'), ylabel('Frequency (Hz)')
    hold on, plot([-100:100]/200,-cos([-100:100]/100*pi+pi)+freqlist(1)+1,'w'), plot([-0.5,-0.5;0.5,0.5]',[freqlist(1),freqlist(end)],'--w'), caxis(caxis_range)
    
    subplot(3,2,2) % Sham trials
    imagesc(window_time,freqlist,mean(theta_triggered2,3)), set(gca,'YDir','normal'), title('Sham trials'), xlabel('Time (s)'), ylabel('Frequency (Hz)')
    hold on, plot([-0.5,-0.5;0.5,0.5]',[freqlist(1),freqlist(end)],'--w'), caxis(caxis_range)
    
    subplot(3,2,3) % Frequency peak of the spectrogram
    %[~,index] = max(theta_triggered1);
    %plot(time_window,permute(freqlist(index),[2,3,1]),'color',[1,0,0,0.1]), hold on
    [~,index] = max(mean(theta_triggered1,3)); hold on
    plot(window_time,freqlist(index),'r','linewidth',2)
    %[~,index] = max(theta_triggered2);
    %plot(time_window,permute(freqlist(index),[2,3,1]),'color',[0,0,1,0.1]), hold on
    [~,index] = max(mean(theta_triggered2,3));
    plot(window_time,freqlist(index),'b','linewidth',2), title('Frequency'), ylabel('Frequency (Hz)')
    plot([-100:100]/200,-cos([-100:100]/100*pi+pi)+6,'k','linewidth',2), gridxy([-0.5,-0.5;0.5,0.5]','linestyle','--','color','k')
    
    subplot(3,2,4) % Power of the spectrogram
    %plot(time_window,permute(mean(theta_triggered1),[2,3,1]),'color',[1,0,0,0.1]), hold on
    plot(window_time,mean(mean(theta_triggered1,3)),'r','linewidth',2), hold on
    
    %plot(time_window,permute(mean(theta_triggered2),[2,3,1]),'color',[0,0,1,0.1]), hold on
    plot(window_time,mean(mean(theta_triggered2,3)),'b','linewidth',2), title('Power'), xlabel('Time (s)'), ylabel('Power'), axis tight
    gridxy([-0.5,-0.5;0.5,0.5]','linestyle','--','color','k')
    
    [~,index1] = max(mean(theta_triggered2,3));
    index1 = index1(window_stim);
    [~,index2] = max(mean(theta_triggered1,3));
    index2 = index2(window_stim);
    
    subplot(3,2,5) % Effect of stimulation on frequency
    plot([1,2],[mean(freqlist(index1)),mean(freqlist(index2))],'o-'), hold on % Mean
    plot([1,2],[min(freqlist(index1)),min(freqlist(index2))],'v-') % Min
    plot([1,2],[max(freqlist(index1)),max(freqlist(index2))],'x-') % Min
    xticks([1, 2]), xticklabels({'Sham trials','Stim trials'}),ylabel('Frequency'),title('Effect on frequency')
    
    subplot(3,2,6) % Effect of stimulation on power
    plot([1,2],[mean(mean(mean(theta_triggered2(:,window_stim,:),3))),mean(mean(mean(theta_triggered1(:,window_stim,:),3)))],'o-'), hold on % Mean
    xticks([1, 2]), xticklabels({'Sham trials','Stim trials'}),ylabel('Power'),title('Effect on power')
    
    stats.theta(k,:) = [mean(freqlist(index1)),mean(freqlist(index2))];
    stats.range(k,:) = [range(freqlist(index1)),range(freqlist(index2))];
    stats.power(k,:) = [mean(mean(mean(theta_triggered2(:,window_stim,:),3)))/mean(mean(mean(theta_triggered2(:,window_prestim,:),3))),mean(mean(mean(theta_triggered1(:,window_stim,:),3)))/mean(mean(mean(theta_triggered1(:,window_prestim,:),3)))];
    stats.session{k} = recording.name;
end

figure(100)
subplot(2,2,1)
plot([1,2],stats.theta,'o-'), hold on % Mean
xticks([1, 2]), xticklabels({'Sham trials','Stim trials'}),ylabel('Frequency'),title('Mean frequency')

subplot(2,2,2)
plot([1,2],stats.range,'o-'), hold on
xticks([1, 2]), xticklabels({'Sham trials','Stim trials'}),ylabel('Frequency'),title('Range of frequency')

subplot(2,2,3)
plot([1,2],stats.power,'o-'), hold on % Mean
xticks([1, 2]), xticklabels({'Sham trials','Stim trials'}),ylabel('Relative power'),title('Relative power')

subplot(2,2,4)
boxplot(stats.power), hold on % Mean
xticks([1, 2]), xticklabels({'Sham trials','Stim trials'}),ylabel('Relative power'),title('Relative power')

%% Correlations within sessions for first vs last trials of behavior
clear all
Viktors_Datasets
datasets = [];
datasets.stim = [1,3,4,5,6,7,8,32,33,35,39,49]; % 33 consists of both sham and real stimulation trials
datasets.nostim = [30,37,38,40];
datasets.all = [datasets.stim, datasets.nostim];
datasets.type = [2*ones(1,length(datasets.stim)),ones(1,length(datasets.nostim))];
placefields_corr = [];
units_corr = [];
trialsToAverage = 20;
trialstype = [];
trialstype2 = [];
z = 1;
x = 1;
for k = 1:length(datasets.all)
    id = datasets.all(k);
    cd(recordings(id).folder)
    recording = recordings(id);
    load('units.mat')
    load('animal.mat')
    load('trials.mat')
    if exist('optogenetics.mat')
        load('optogenetics.mat')
        optogenetics.pos = interp1(animal.time,animal.pos',optogenetics.peak)';
        optogenetics.polar_rho = interp1(animal.time,animal.polar_rho',optogenetics.peak);
        optogenetics.polar_theta = interp1(animal.time,animal.polar_theta',optogenetics.peak);
        optogenetics.rim = (optogenetics.polar_rho > animal.polar_rho_limits(1) & optogenetics.polar_rho < animal.polar_rho_limits(2) & optogenetics.polar_theta > animal.polar_theta_limits(1) & optogenetics.polar_theta < animal.polar_theta_limits(2));
        optogenetics.arm = (optogenetics.pos(1,:) > animal.pos_x_limits(1)   & optogenetics.pos(2,:) > animal.pos_y_limits(1)/2 & optogenetics.pos(1,:) < animal.pos_x_limits(2) );
        optogenetics.left = (optogenetics.pos(1,:) < animal.pos_x_limits(1)  & optogenetics.pos(2,:) > animal.pos_y_limits(1));
        optogenetics.right = (optogenetics.pos(1,:) > animal.pos_x_limits(2) & optogenetics.pos(2,:) > animal.pos_y_limits(1));
        optogenetics.trials = interp1(animal.time,trials.trials2',optogenetics.peak,'nearest');
    end
    zz = 1; % Trials
    xx = 1; % Units 
    for i = 1:length(units)
        % % % % % % % % % % %
        % All maze
        if isfield(units(i).PhasePrecession,'placefields_polar_theta') || isfield(units(i).PhasePrecession,'placefields_center_arm')
        % Early trials
        maze_offset = 80;
        theta_limit = 50;
        spikesInField01 = find(units(i).polar_theta > -theta_limit &  units(i).polar_theta < theta_limit & units(i).rim == 1 & units(i).trials > trialsToAverage & units(i).trials < trials.total-trialsToAverage);
        spikesInField02 = find(units(i).arm == 1 & units(i).trials > trialsToAverage & units(i).trials < trials.total-trialsToAverage);
        
        timeInField1 = find(animal.polar_theta > -theta_limit &  animal.polar_theta < theta_limit & animal.rim == 1 & trials.trials2 <= trialsToAverage);
        timeInField2 = find(animal.arm == 1 & trials.trials2 <= trialsToAverage);
        spikesInField11 = find(units(i).polar_theta > -theta_limit &  units(i).polar_theta < theta_limit & units(i).rim == 1 & units(i).trials <= trialsToAverage);
        spikesInField12 = find(units(i).arm == 1 & units(i).trials <= trialsToAverage);
        
        bins = [-60:5:110];
        test1 = histcounts([units(i).polar_theta(spikesInField11), units(i).pos(2,spikesInField12)+maze_offset] ,bins)./histcounts([animal.polar_theta(timeInField1),animal.pos(2,timeInField2)+maze_offset],bins);
        test1(isnan(test1)) = 0;
        % Late trials
        timeInField1 = find(animal.polar_theta > -theta_limit &  animal.polar_theta < theta_limit & animal.rim == 1 & trials.trials2 >= trials.total-trialsToAverage);
        timeInField2 = find(animal.arm == 1 & trials.trials2 >= trials.total-trialsToAverage);
        spikesInField21 = find(units(i).polar_theta > -theta_limit &  units(i).polar_theta < theta_limit & units(i).rim == 1 & units(i).trials >= trials.total-trialsToAverage);
        spikesInField22 = find(units(i).arm == 1 & units(i).trials >= trials.total-trialsToAverage);
        test2 = histcounts([units(i).polar_theta(spikesInField21), units(i).pos(2,spikesInField22)+maze_offset],bins)./histcounts([animal.polar_theta(timeInField1),animal.pos(2,timeInField2)+maze_offset],bins);
        test2(isnan(test2)) = 0;
        units_corr(x) = corr(test1',test2');
        trialstype2(x) = datasets.type(k);
        figure(k+length(datasets.all))
        subplot(6,7,xx)
        plot([units(i).polar_theta(spikesInField01),units(i).pos(2,spikesInField02)+maze_offset],units(i).trials([spikesInField01,spikesInField02]),'.k'), hold on
        plot([units(i).polar_theta(spikesInField11),units(i).pos(2,spikesInField12)+maze_offset],units(i).trials([spikesInField11,spikesInField12]),'.b')
        plot([units(i).polar_theta(spikesInField21),units(i).pos(2,spikesInField22)+maze_offset],units(i).trials([spikesInField21,spikesInField22]),'.r'), axis tight
        if exist('optogenetics.mat')
            plot(optogenetics.polar_theta(optogenetics.rim),optogenetics.trials(optogenetics.rim),'.m')
            plot(optogenetics.pos(2,optogenetics.arm)+maze_offset,optogenetics.trials(optogenetics.arm),'.m')
        end
        xlim([bins(1),bins(end)])
        title(['Unit ', num2str(i)]), ylabel(['Corr ' num2str(units_corr(x),2)])
        figure(k+2*length(datasets.all))
        subplot(6,7,xx)
        stairs(bins(1:end-1),test1), hold on, stairs(bins(1:end-1),test2)
        title(['Unit ', num2str(i)]), ylabel(['Corr ' num2str(units_corr(x),2)]), axis tight
        x=x+1;
        xx = xx+1;
        end
        % % % % % % % % %
        % Rim placefields
        figure(k)
        if isfield(units(i).PhasePrecession,'placefields_polar_theta')
            for m = 1:size(units(i).PhasePrecession.placefields_polar_theta,1)
                % Early trials
                spikesInField = find(units(i).polar_theta > units(i).PhasePrecession.placefields_polar_theta(m,1) & units(i).polar_theta < units(i).PhasePrecession.placefields_polar_theta(m,2) & units(i).rim == 1 & units(i).trials > trialsToAverage & units(i).trials < trials.total-trialsToAverage);
                bins = [units(i).PhasePrecession.placefields_polar_theta(m,1):3:units(i).PhasePrecession.placefields_polar_theta(m,2)+3];
                timeInField = find(animal.polar_theta > units(i).PhasePrecession.placefields_polar_theta(m,1) & animal.polar_theta < units(i).PhasePrecession.placefields_polar_theta(m,2) & animal.rim == 1 & trials.trials2 <= trialsToAverage);
                spikesInField1 = find(units(i).polar_theta > units(i).PhasePrecession.placefields_polar_theta(m,1) & units(i).polar_theta < units(i).PhasePrecession.placefields_polar_theta(m,2) & units(i).rim == 1 & units(i).trials <= trialsToAverage);
                
                test1 = histcounts(units(i).polar_theta(spikesInField1),bins)./histcounts(animal.polar_theta(timeInField),bins);
                test1(isnan(test1)) = 0;
                % Late trials
                timeInField = find(animal.polar_theta > units(i).PhasePrecession.placefields_polar_theta(m,1) & animal.polar_theta < units(i).PhasePrecession.placefields_polar_theta(m,2) & animal.rim == 1 & trials.trials2 >= trials.total-trialsToAverage);
                spikesInField2 = find(units(i).polar_theta > units(i).PhasePrecession.placefields_polar_theta(m,1) & units(i).polar_theta < units(i).PhasePrecession.placefields_polar_theta(m,2) & units(i).rim == 1 & units(i).trials >= trials.total-trialsToAverage);
                test2 = histcounts(units(i).polar_theta(spikesInField2),bins)./histcounts(animal.polar_theta(timeInField),bins);
                test2(isnan(test2)) = 0;
                placefields_corr(z) = corr(test1',test2');
                trialstype(z) = datasets.type(k);
                subplot(6,6,zz)
                plot(units(i).polar_theta(spikesInField),units(i).trials(spikesInField),'.k'), hold on
                plot(units(i).polar_theta(spikesInField1),units(i).trials(spikesInField1),'.b')
                plot(units(i).polar_theta(spikesInField2),units(i).trials(spikesInField2),'.r'), axis tight
                if exist('optogenetics.mat')
                    plot(optogenetics.polar_theta(optogenetics.rim),optogenetics.trials(optogenetics.rim),'.m')
                end
                xlim([bins(1),bins(end)])
                title(['Unit ', num2str(i), ', rim ', num2str(m)]), ylabel(['Corr ' num2str(placefields_corr(z),2)])
                z=z+1;
                zz = zz + 1;
            end
        end
        
        % Central arm place fields
        if isfield(units(i).PhasePrecession,'placefields_center_arm')
            for m = 1:size(units(i).PhasePrecession.placefields_center_arm,1)
                % Early trials
                spikesInField = find(units(i).pos(2,:) > units(i).PhasePrecession.placefields_center_arm(m,1) & units(i).pos(2,:) < units(i).PhasePrecession.placefields_center_arm(m,2) & units(i).arm == 1 & units(i).trials > trialsToAverage & units(i).trials < trials.total-trialsToAverage);
                bins = [units(i).PhasePrecession.placefields_center_arm(m,1):3:units(i).PhasePrecession.placefields_center_arm(m,2)+3];
                timeInField = find(animal.pos(2,:) > units(i).PhasePrecession.placefields_center_arm(m,1) & animal.pos(2,:) < units(i).PhasePrecession.placefields_center_arm(m,2) & animal.arm == 1 & trials.trials2 <= trialsToAverage);
                spikesInField1 = find(units(i).pos(2,:) > units(i).PhasePrecession.placefields_center_arm(m,1) & units(i).pos(2,:) < units(i).PhasePrecession.placefields_center_arm(m,2) & units(i).arm == 1 & units(i).trials <= trialsToAverage);
                test1 = histcounts(units(i).pos(2,spikesInField1),bins)./histcounts(animal.pos(2,timeInField),bins);
                test1(isnan(test1)) = 0;
                % Late trials
                bins = [units(i).PhasePrecession.placefields_center_arm(m,1):3:units(i).PhasePrecession.placefields_center_arm(m,2)+3];
                timeInField = find(animal.pos(2,:) > units(i).PhasePrecession.placefields_center_arm(m,1) & animal.pos(2,:) < units(i).PhasePrecession.placefields_center_arm(m,2) & animal.arm == 1 & trials.trials2 >= trials.total-trialsToAverage);
                spikesInField2 = find(units(i).pos(2,:) > units(i).PhasePrecession.placefields_center_arm(m,1) & units(i).pos(2,:) < units(i).PhasePrecession.placefields_center_arm(m,2) & units(i).arm == 1 & units(i).trials >= trials.total-trialsToAverage);
                test2 = histcounts(units(i).pos(2,spikesInField2),bins)./histcounts(animal.pos(2,timeInField),bins);
                test2(isnan(test2)) = 0;
                placefields_corr(z) = corr(test1',test2');
                trialstype(z) = datasets.type(k);
                subplot(6,6,zz)
                plot(units(i).pos(2,spikesInField),units(i).trials(spikesInField),'.k'), hold on
                plot(units(i).pos(2,spikesInField1),units(i).trials(spikesInField1),'.b')
                plot(units(i).pos(2,spikesInField2),units(i).trials(spikesInField2),'.r'), axis tight
                if exist('optogenetics.mat')
                    plot(optogenetics.pos(2,optogenetics.arm),optogenetics.trials(optogenetics.arm),'.m')
                end
                title(['Unit ', num2str(i), ', arm ', num2str(m)]), ylabel(['Corr ' num2str(placefields_corr(z),2)])
                xlim([bins(1),bins(end)])
                z = z + 1;
                zz = zz + 1;
            end
        end
    end
    clear units animal trials
end

% Placefields
placefields_corr(isnan(placefields_corr)) = 0;
figure
boxplot(placefields_corr,trialstype)
xticks([1,2])
xticklabels({['sham (' num2str(sum(trialstype==1)) ')'],['stim (' num2str(sum(trialstype==2)) ')']})
title('Placefield stability for sham and stim sessions')

% Units
figure
boxplot(units_corr,trialstype2)
xticks([1,2])
xticklabels({['sham (' num2str(sum(trialstype2==1)) ')'],['stim (' num2str(sum(trialstype2==2)) ')']})
title('Unit stability for sham and stim sessions')



