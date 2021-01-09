MedialSeptum_Recordings;
sessionIDs = [126,140,139,127,78,79,80,81,92,93] % 61,78,79,80,81,92,93
sessionNames = cellstr(vertcat(recordings(sessionIDs).name));
% sessionsMS14  = {'Peter_MS14_180122_163606_concat'};
% sessionsMS18  = {'Peter_MS18_180519_120300_concat'};
% sessionsMS21  = {'Peter_MS21_180808_115125_concat','Peter_MS21_180718_103455_concat', 'Peter_MS21_180629_110332_concat','Peter_MS21_180627_143449_concat','Peter_MS21_180719_155941_concat'};
% sessionsMS22  = {'Peter_MS22_180628_120341_concat','Peter_MS22_180629_110319_concat','Peter_MS22_180719_122813_concat'};
% sessionNames = [sessionsMS14,sessionsMS18,sessionsMS21,sessionsMS22];
cell_metrics = LoadCellMetricBatch('sessions',sessionNames);
% cell_metrics = CellInspector('metrics',cell_metrics);

%%
InterneuronsIndexes = find(contains(cell_metrics.putativeCellType,'Narrow Interneuron'));
% InterneuronsIndexes = find(contains(cell_metrics.PutativeCellType,'Wide Interneuron'));
recording_sessions = cell_metrics.sessionName(InterneuronsIndexes);
Interneurons_rate = [];
m = 1;
CCGs_interneurons = [];
mm = 1;
ISI_hist = [];
for j = 1:length(sessionNames)
    temp = InterneuronsIndexes(find(strcmp(recording_sessions,sessionNames{j})));
    temp = cell_metrics.UID(temp);
    
    [session, basename, basepath, clusteringpath] = db_set_path('session',sessionNames{j});
    spikes = loadClusteringData(sessionNames{j},session.spikeSorting.format{1},clusteringpath);
    load('animal.mat')
%     load('trials.mat')
    sr = session.extracellular.sr;
    
    for i = 1:size(spikes.ts,2)
        spikes.ts{i} = spikes.ts{i}; % (spikes.times{i} < animal.time(end))
        spikes.times{i} = spikes.ts{i}/sr;
        spikes.total(i) = length(spikes.ts{i});
        spikes.ts_eeg{i} = ceil(spikes.ts{i}/16);        
        spikes.speed{i} = interp1(animal.time,animal.speed,spikes.times{i});
        spikes.pos_linearized{i} = interp1(animal.time,animal.pos_linearized,spikes.times{i});
        spikes.state{i} = interp1(animal.time,animal.state,spikes.times{i},'nearest');
    end

    for k = 1:length(temp)
        for kk = 1:length(animal.state_labels)
            Interneurons_rate(1,m) = j;
            Interneurons_rate(2,m) = temp(k);
            Interneurons_rate(kk+2,m) = sum(~isnan(spikes.pos_linearized{temp(k)}) & spikes.state{temp(k)}==kk)/sum(~isnan(animal.pos_linearized) & animal.state==kk)*animal.sr;
            indx = find(~isnan(spikes.pos_linearized{temp(k)}) & spikes.state{temp(k)}==kk);
            ISI_hist(:,m,kk) = histcounts(diff(spikes.times{temp(k)}(indx)*1000),[1:100]);
        end
        m = m + 1;
    end

    for kk = 1:length(animal.state_labels)
        spikes2 = [];
        for k = 1:length(spikes.UID)
            indx = find(~isnan(spikes.pos_linearized{k}) & spikes.state{k}==kk);
            spikes2.times{k} = spikes.times{k}(indx);
        end
        spikes2.UID = spikes.UID;
        spikes2 = generateSpinDices(spikes2);
        [ccg,time] = CCG(spikes2.spindices(:,1),spikes2.spindices(:,2),'binSize',0.001,'duration',1,'norm','rate');
        for k = 1:length(temp)
            CCGs_interneurons(:,mm,kk) = nanconv(ccg(:,temp(k),temp(k))',gausswin(60)','edge');
            mm = mm + 1;
        end
    end

end

figure, 
subplot(3,1,1)
imagesc(zscore(ISI_hist(:,:,1))')
subplot(3,1,2)
imagesc(zscore(ISI_hist(:,:,2))')
subplot(3,1,3)
imagesc(zscore(ISI_hist(:,:,3))')

figure,
id = 26
plot(zscore(ISI_hist(:,id,1))), hold on
plot(zscore(ISI_hist(:,id,2)))
plot(zscore(ISI_hist(:,id,3)))

%%
colors = {'g','b','r'};

figure, subplot(2,3,1)
AverageRate = Interneurons_rate(3:5,:);
AverageRate(:,AverageRate(1,:)<0.5) = [];
plot(AverageRate), ylabel('Average rate (Hz)')
xticks([1:3]), xticklabels(animal.state_labels)

subplot(2,3,2), boxplot(AverageRate')
xticks([1:3])
xticklabels(animal.state_labels)
ratio0 = AverageRate;
ratio1 = [ratio0;ratio0(2,:)]; % [ratio0(1,:)./ratio0(2,:);ratio0(3,:)./ratio0(2,:)];
x = ratio1(1,:); y1 = ratio1(2,:); 

subplot(2,3,3), plot(ratio1(1,:),ratio1(2,:),'o'), hold on 
P = polyfit(x,y1,1); yfit = P(1)*x+P(2); plot(x,yfit,'k-');
[R,P] = corrcoef(x,y1); text(max(x),max(yfit),['R = ' num2str(R(2,1),3),', P = ', num2str(P(2,1),3)],'Color','k')
axis tight, grid on, xlabel('Pre'), ylabel('Cooling'), axis equal, xlim([0,80]), ylim([0,80]), plot([0,80],[0,80])

subplot(2,3,6), plot(ratio1(3,:),ratio1(2,:),'o'), hold on 
P = polyfit(x,y1,1); yfit = P(1)*x+P(2); plot(x,yfit,'k-');
[R,P] = corrcoef(x,y1); text(max(x),max(yfit),['R = ' num2str(R(2,1),3),', P = ', num2str(P(2,1),3)],'Color','k')
axis tight, grid on, xlabel('Post'), ylabel('Cooling'), axis equal, xlim([0,80]), ylim([0,80]), plot([0,80],[0,80])

subplot(2,3,4)
colors = {'k','b','r'};
conditions = {'Pre','Cooling','Post'}
x_data = AverageRate;
x_bins = [0:10:100];
for i = 1:3
    x1 = x_data(i,:); 
    [N,edges] = histcounts(x1,x_bins, 'Normalization', 'probability')
    x_bins_diff = diff(x_bins); x_bins_diff = x_bins_diff(1);
    x2 = [x_bins(1:end-1)+x_bins_diff;x_bins(1:end-1)+x_bins_diff];
    y = [N;N];
    area(x2([2:end end]),y(1:end),'FaceColor',colors{i}), hold on, xlim([0,80]), xlabel('Oscillation frequency (Hz)')
end

subplot(2,3,5)
plot(AverageRate(2,:)./AverageRate(1,:),AverageRate(2,:)./AverageRate(3,:),'.'), xlabel('Cooling/Pre'), ylabel('Cooling/Post'), grid on, axis equal

%% % ACG of interneurons
figure
for j = 1:3
    
    temp = CCGs_interneurons(:,:,j);
    temp = temp(:,find(sum(temp)>0));
    temp2 = temp-min(temp);
    subplot(2,3,j)
%     plot(CCGs_interneurons(:,:,i))
    
    Arowmax = max(temp2', [], 2);
    [~,idx] = sort(Arowmax, 'descend'); % Amplitude
%     [~,idx] = sort(idx_offset, 'descend'); % By offset
    temp2 = temp2(:,idx);
    
%     temp2 = sort(temp2,2);
    
    temp2 = temp2./max(temp2);
    subplot(2,3,j)
    imagesc([-500:500],[1:size(temp2,2)],temp2'),
    title(conditions{j}), xlabel('Time (ms)'), xlim([-500,500])
    subplot(2,1,2)
    plot([-500:500],nanmean(temp2')/max(nanmean(temp2'))), hold on, xlim([-500,500])
    title('ACGs - time'), xlabel('Time (ms)')
end
