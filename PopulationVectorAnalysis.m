function PopulationVectorAnalysisLength = PopulationVectorAnalysis(id)
%%  Population vector analysis
% [~,~,PyramidalIndexes1] = intersect(find(cellfun(@isempty,recording.SpikeSorting.center_arm_placecells)==0),spikes.cluID);
% PyramidalIndexes = PyramidalIndexes1';
% 
% [~,~,PyramidalIndexes2] = intersect(find(cellfun(@isempty,recording.SpikeSorting.polar_theta_placecells)==0),spikes.cluID);
% PyramidalIndexes = PyramidalIndexes2';
MedialSeptum_Recordings;
% id = 140;

recording = recordings(id);

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

PyramidalIndexes = get_CellMetrics('session',recording.name,'putativeCellType',{'Pyramidal'});
colormap2 = jet(length(PyramidalIndexes));
bin_step_size = 4;
bins_position1 = [0:bin_step_size:350];
states = {'Pre','Cooling','Post'};
pos_interval = [85,215,350];
r = [1:length(bins_position1)-1];
diagonalValues = toeplitz(r);
figure('position',[50,50,1000,800])
for jj = 1:3
    placefields = nan(length(bins_position1)-1,length(PyramidalIndexes));
    
    trials_state = intersect(find(trials.cooling==jj),setdiff([1:length(trials.cooling)],trials.error));
    idx = find(ismember(trials.trials2,trials_state) & animal.pos_linearized>0);
    N_animal = histcounts(animal.pos_linearized(idx),bins_position1);
    N_animal = N_animal/animal.sr;
    subplot(3,1,jj)
    for ii = 1:length(PyramidalIndexes)
        i = PyramidalIndexes(ii);
        idx2 = find(ismember(spikes.trials{i},trials_state) & spikes.pos_linearized{i}>0);
        N = histcounts(spikes.pos_linearized{i}(idx2),bins_position1);
%         placefields(:,ii) = nanconv(N./N_animal,gausswin(4)'/sum(gausswin(4)));
        placefields(:,ii) = N./N_animal;
        plot(bins_position1(1:end-1),placefields(:,ii),'-','color',colormap2(ii,:)), hold on
    end
    [R,P] = corrcoef(placefields');
    R_all{jj} = R;
    P_all{jj} = P;
end
path = 'K:\Dropbox\Buzsakilab Postdoc\MatlabFigures\PopulationVector';
print(gcf, [path,'\rateMaps_' recording.name],'-dpdf')

figure('position',[50,50,1000,800])
PopulationVectorAnalysisLength = [];
for jj = 1:3
    mean_data = [];
    std_data = [];
    subplot(3,2,jj*2-1)
    imagesc(bins_position1(1:end-1),bins_position1(1:end-1),R_all{jj}), hold on
    xlabel('Position (cm)'), ylabel('Position (cm)')
    title(states{jj})
    plot([pos_interval;pos_interval]-bin_step_size,[0,max(pos_interval)]'*[1,1,1],'w')
    plot([0,max(pos_interval)]'*[1,1,1],[pos_interval;pos_interval]-bin_step_size,'w')
    subplot(3,2,jj)
    x_data = diagonalValues(:);
    y_data = R_all{jj}(:);
    for i = 1:length(bins_position1)-1
        idx = find(x_data == i);
        mean_data(i) = nanmean(y_data(idx));
        std_data(i) = nanstd(y_data(idx));
    end
    subplot(3,2,jj*2)
    plot(x_data,y_data,'.b'), hold on
    plot(mean_data,'k','linewidth',2)
    plot(mean_data+std_data,'k')
    plot(mean_data-std_data,'k')
    idx = find(mean_data<0.5,1);
    PopulationVectorAnalysisLength(jj) = bins_position1(idx)+1;
    plot(idx,0.5,'or'), hold on
    plot([pos_interval;pos_interval]/bin_step_size-bin_step_size,[-0.5,1.5]'*[1,1,1],'k')
    xlim([0,max(x_data)]), ylim([-0.5,1]), xlabel('Position (bins)'), ylabel('Correlation'), title(states{jj})
end

print(gcf, [path,'\populationVector_' recording.name],'-dpdf');
save('PopulationVectorAnalysisLength.mat','PopulationVectorAnalysisLength');
