% function MedialSeptum_TrialAverageAcrossSessions
% Time dependent metrics across sessions for MS cooling data
MedialSeptum_Recordings;
sessionIDs = [126,140,139,127,78,79,80,81,92,93]; % 61

% figure
out = {};
tri_all = [];
for i = 1:length(sessionIDs)
    disp([num2str(i),'/' num2str(length(sessionIDs))])
    recording = recordings(sessionIDs(i));
    [session, baseName, basepath, clusteringpath] = db_set_path('session',recording.name);
    sr = session.extracellular.sr;
    srLfp = session.extracellular.srLfp;
    load('animal.mat');
    load('trials.mat');
    spikes = loadClusteringData(baseName,session.spikeSorting.format{1},clusteringpath);
    
    out{i} = plotPlaceFieldStability(spikes,trials);
    temp2 = [];
    kkk = 1;
    cooling_temp = vertcat(spikes.temperature{:});
    cooling_trial = vertcat(spikes.trials{:});
    for jj = 1:trials.total
        cooling_trial2(jj) = mean(cooling_temp(find(cooling_trial==jj)));
    end
    cooling_start_trial = find(diff(cooling_trial2<34)==1,1);
    trials_axis = [1:trials.total]-cooling_start_trial;
    
    for j = 1:length(spikes.ts)
        if length(spikes.PhasePrecession)>=j && ~isempty(spikes.PhasePrecession{j})
            PhasePrecession1 = spikes.PhasePrecession{j};
            if isfield(PhasePrecession1,'placefields_polar_theta')
                for jjj = 1:size(PhasePrecession1.placefields_polar_theta,1)
                    temp = find(spikes.rim{j}' & spikes.polar_theta{j} > PhasePrecession1.placefields_polar_theta(jjj,1) & spikes.polar_theta{j} < PhasePrecession1.placefields_polar_theta(jjj,2));
                    phasepresession_sign = PhasePrecession1.placefields_polar_theta(jjj,3);
                    phasepresession_data = [];
                    phasepresession_data(1,:) = spikes.polar_theta{j};
                    phasepresession_data(2,:) = spikes.theta_phase{j};
                    [tri,out] = calc_trial_averages(spikes,j,temp,phasepresession_sign,phasepresession_data);
                    PlotFieldnames = fieldnames(tri);

                    figure
                    tri_all.trials_axis{kkk} = out.trials_axis+out.trial_step_size-cooling_start_trial;
                    for k = 1:length(PlotFieldnames)
                        tri_all.(PlotFieldnames{k}){kkk} = tri.(PlotFieldnames{k});
                        subplot(3,3,k),plot(tri_all.trials_axis{kkk},tri.(PlotFieldnames{k})), title(PlotFieldnames{k}), axis tight
                    end
                    drawnow
                    kkk = kkk + 1;
                end
            end
            if isfield(PhasePrecession1,'placefields_center_arm')
                for jjj = 1:size(PhasePrecession1.placefields_center_arm,1)
                    temp = find(spikes.arm{j}' & spikes.pos{j}(2,:)' > PhasePrecession1.placefields_center_arm(jjj,1) & spikes.pos{j}(2,:)' < PhasePrecession1.placefields_center_arm(jjj,2));
                    phasepresession_sign = PhasePrecession1.placefields_center_arm(jjj,3);
                    phasepresession_data = [];
                    phasepresession_data(1,:) = spikes.polar_theta{j};
                    phasepresession_data(2,:) = spikes.theta_phase{j};
                    [tri,out] = calc_trial_averages(spikes,j,temp,phasepresession_sign,phasepresession_data);
                    
                    figure
                    PlotFieldnames = fieldnames(tri);
                    tri_all.trials_axis{kkk} = out.trials_axis+out.trial_step_size-cooling_start_trial;
                    for k = 1:length(PlotFieldnames)
                        tri_all.(PlotFieldnames{k}){kkk} = tri.(PlotFieldnames{k});
                        subplot(3,3,k),plot(tri_all.trials_axis{kkk},tri.(PlotFieldnames{k})), title(PlotFieldnames{k})
                    end
                    drawnow
                    kkk = kkk + 1;
                end
            end
        end
    end
end

for k = 1:length(PlotFieldnames)-1
    figure
    for i = 1:length(tri_all.(PlotFieldnames{k}))
        plot(tri_all.trials_axis{k},tri_all.(PlotFieldnames{k}){i}), hold on, axis tight
    end
    title(PlotFieldnames{k})
end
