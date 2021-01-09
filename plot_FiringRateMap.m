function PhasePrecession = plot_FiringRateMap(varargin)
% PhasePrecession = plot_FiringRateMap('animal',animal,'spikes',spikes,'trials',trials,'binsize',5)
% 
% INPUTS
% animal
% spikes
% trials
% units2plot
% bin_size
% textlabel
%
% OUTPUTS
% firing_rate_map
%
% By Peter Petersen
% petersen.peter@gmail.com

p = inputParser;

addParameter(p,'animal',[],@isstruct);
addParameter(p,'spikes',[],@isstruct);
addParameter(p,'trials',[],@isstruct);
addParameter(p,'bin_size',5,@isnumeric);
addParameter(p,'units2plot',[],@isnumeric);
addParameter(p,'theta',[],@isstruct);
addParameter(p,'textlabel','none',@isstr);

parse(p,varargin{:})

animal = p.Results.animal;
spikes = p.Results.spikes;
trials = p.Results.trials;
bin_size = p.Results.bin_size;
units2plot = p.Results.units2plot;
textlabel = p.Results.textlabel;
theta = p.Results.theta;


PhasePrecession = [];
if isfield(trials,'temperature')
    trial_colors = winter(ceil(10*(max(trials.temperature)-min(trials.temperature))));
    trials.colors = trial_colors(1+floor((trials.temperature-min(trials.temperature))/(max(trials.temperature)-min(trials.temperature))*(size(trial_colors,1)-1)),:);
end

% Plotting 
x_hist2 = [animal.pos_limits(1):bin_size:animal.pos_limits(2)];
colors = [0.8500, 0.3250, 0.0980; 0, 0.4470, 0.7410; 0.9290, 0.6940, 0.1250; 0.4940, 0.1840, 0.5560; 0.4660, 0.6740, 0.1880; 0.3010, 0.7450, 0.9330; 0.6350, 0.0780, 0.1840; 0.4660, 0.6740, 0.1880; 0.3010, 0.7450, 0.9330; 0.6350, 0.0780, 0.1840];
if ~exist('units2plot')
    units2plot = [];
end
if isempty(units2plot)
    units2plot = 1:size(spikes.ts,2);
end
if ~isfield(spikes,'PhasePrecession')
    spikes.PhasePrecession(units2plot) = {[]};
end

ii = 0;
for i = units2plot
    ii = ii + 1;
    clear unit;
    unit.ts = spikes.ts{i};
    unit.times = spikes.times{i};
    unit.total = spikes.total(i);
    unit.PhasePrecession = spikes.PhasePrecession{i};
    unit.trials = spikes.trials{i};
    unit.theta_freq = spikes.theta_freq{i};
    unit.speed = spikes.speed{i};
    if ~isfield(spikes,'pos')
        unit.pos = interp1(animal.time,animal.pos,unit.times);
    else
        unit.pos = spikes.pos{i};
    end
    if ~isfield(spikes,'speed')
        unit.speed = interp1(animal.time,animal.speed,unit.times);
    else
        unit.speed = spikes.speed{i};
    end
    if ~isfield(spikes,'theta_phase')
        unit.theta_phase = interp1([1:length(theta.phase)]/theta.sr,theta.phase,unit.times);
    else
        unit.theta_phase = spikes.theta_phase{i};
    end
    if ~isfield(spikes,'state')
        unit.state = interp1(animal.time,animal.state,unit.times,'nearest');
    else
        unit.state = spikes.state{i};
    end
    if ~isempty(spikes.PhasePrecession)
        unit.PhasePrecession = spikes.PhasePrecession{i};
        polar_theta_placecells = 1;
    else
        unit.PhasePrecession = [];
        polar_theta_placecells = 0;
    end
    for k = 1:sum(unique(animal.state)>0)
        % Phase precession
        subplot_Peter(5,sum(unique(animal.state)>0)+2,ii,k)
        if isfield(unit,'rim')
            indexes = find(unit.state == k & unit.speed > animal.speed_th & unit.rim > 0); 
        elseif isfield(animal,'rim')
            unit.rim = interp1(animal.time,animal.rim,unit.times,'nearest');
            indexes = find(unit.state == k & unit.speed > animal.speed_th & unit.rim > 0); 
        else
            indexes = find(unit.state == k);
        end
        unit2 = []; 
        unit2.pos = unit.pos(indexes);
        unit2.theta_phase = unit.theta_phase(indexes);
        unit2.trials = unit.trials(indexes);
        unit2.times = unit.times(indexes);
        unit2.theta_freq = unit.theta_freq(indexes);
        unit2.speed = unit.speed(indexes);
        if ~isfield(spikes,'trials')
            unit2.trials = interp1(animal.time,trials.trials{k},unit.times(indexes),'nearest');
        end
        if polar_theta_placecells
            if ~isempty(unit.PhasePrecession)
                unit2.PhasePrecession = unit.PhasePrecession;
                indexes2 = find(animal.state == k & animal.speed > animal.speed_th  & animal.rim);
                animal2 = [];
                animal2.time = animal.time(indexes2);
                animal2.pos = animal.pos(indexes2);
                animal2.sr = animal.sr;
                [PhasePrecession(i).Slope{k},temp,temp2] = plot_PhasePrecession(unit2,colors(k,:),animal2);
                
                PhasePrecession(i).placefield = unit2.PhasePrecession;
                if ~isfield(PhasePrecession(i),'spikesperTrials')
                    PhasePrecession(i).spikesperTrials = zeros(size(temp,1),trials.total);
                    PhasePrecession(i).PhasePrecessionTrials.slope2 = zeros(size(temp,1),trials.total);
                    PhasePrecession(i).PhasePrecessionTrials.offset2 = zeros(size(temp,1),trials.total);
                    PhasePrecession(i).PhasePrecessionTrials.R2 = zeros(size(temp,1),trials.total);
                elseif isempty(PhasePrecession(i).spikesperTrials)
                    PhasePrecession(i).spikesperTrials = zeros(size(temp,1),trials.total);
                    PhasePrecession(i).PhasePrecessionTrials.slope2 = zeros(size(temp,1),trials.total);
                    PhasePrecession(i).PhasePrecessionTrials.offset2 = zeros(size(temp,1),trials.total);
                    PhasePrecession(i).PhasePrecessionTrials.R2 = zeros(size(temp,1),trials.total);
                end
                trials_all = unique(unit2.trials);
                trials_all = min(trials_all):max(trials_all);
                if ~isempty(temp)
                    PhasePrecession(i).spikesperTrials(:,trials_all) = PhasePrecession(i).spikesperTrials(:,trials_all)+temp(:,trials_all);
                end
                if ~isempty(temp2.slope2)
                    PhasePrecession(i).PhasePrecessionTrials.slope2(:,trials_all) = PhasePrecession(i).PhasePrecessionTrials.slope2(:,trials_all)+temp2.slope2(:,trials_all);
                    PhasePrecession(i).PhasePrecessionTrials.offset2(:,trials_all) = PhasePrecession(i).PhasePrecessionTrials.offset2(:,trials_all)+temp2.offset2(:,trials_all);
                    PhasePrecession(i).PhasePrecessionTrials.R2(:,trials_all) = PhasePrecession(i).PhasePrecessionTrials.R2(:,trials_all)+temp2.R2(:,trials_all);
                end
                
                % Oscillation frequency of the place field
                timestamps = unit2.times;
                timestamps = timestamps - timestamps(1);
                interval = zeros(1,ceil(timestamps(end)*1000+50));
                interval(1+round(timestamps*1000)) = 1;
                interval = nanconv(interval,gausswin(80)','edge');
                
                xcorr_spikes =xcorr(interval,180);
                [~,locs] = findpeaks(xcorr_spikes(181+50:181+150),'SortStr','descend');
                
                if length(locs)>0
                   PhasePrecession(i).OscillationFreq(k) = 1/abs(locs(1)+50)*1000;
                else
                   PhasePrecession(i).OscillationFreq(k) = nan;
                end
                PhasePrecession(i).theta_freq(k) = mean(unit2.theta_freq);
                PhasePrecession(i).speed(k) = mean(unit2.speed);
                
            else
                plot_PhasePrecession(unit2,colors(k,:));
            end
        else
            plot_PhasePrecession(unit2,colors(k,:));
        end
        grid on
        if isfield(animal,'maze')
            if isfield(animal.maze,'reward_points')
                plot(animal.maze.reward_points,-pi,'sm')
                plot(animal.maze.reward_points,3*pi,'sm')
            end
        end
        xlim([x_hist2(1),x_hist2(end)]),ylim([-pi,3*pi]),
        if k == 1; 
            ylabel(['Unit ' num2str(i) ' (id ' num2str(spikes.cluID(i)) ')']); 
        end
        % if mod(i,5) == 0; xlabel('Position on rim (cm)'); end
        if isfield(animal,'state_labels') && mod(i,5) == 1; title(animal.state_labels{k}); end
        if isfield(animal,'state_labels') && strcmp(animal.state_labels{k},'Stim')
            plot(animal.optogenetics.pos,-pi,'^k')
            % hist_opto_count = histogram(animal.optogenetics.pos,x_hist2,'Normalization','probability','facecolor','none','edgecolor','k');
            % stairs(x_hist2,hist_opto_count.,'color',colors(k,:)), hold on
        end
        if isfield(animal,'maze')
            if isfield(animal.maze,'boundaries')
                gridxy(animal.maze.boundaries)
            end
        end
    end
    
    % Trials
    subplot_Peter(5,sum(unique(animal.state)>0)+2,ii,sum(unique(animal.state)>0)+1)
    for k = 1:sum(unique(trials.state)>0)
        if isfield(unit,'rim')
            indexes = find(unit.state == k & unit.speed > animal.speed_th & unit.rim > 0); 
        elseif isfield(animal,'rim') & ~isfield(unit,'rim')
            unit.rim = interp1(animal.time,animal.rim,unit.times,'nearest');
            indexes = find(unit.state == k & unit.speed > animal.speed_th & unit.rim > 0); 
        else
            indexes = find(unit.state == k);
        end
        if ~isfield(spikes,'trials')
            unit.trials = interp1(animal.time,trials.trials{k},unit.times,'nearest');
        end
        if isfield(trials,'temperature')
            units2plot1 = unit.pos(indexes);
            trials2plot = unit.trials(indexes);
            for kk = min(trials2plot):max(trials2plot)
            indexes3 = find( trials2plot == kk );
            plot(units2plot1(indexes3),trials2plot(indexes3),'.','color',trials.colors(kk,:),'markersize',5), hold on
            %scatter(units2plot(indexes3),trials2plot(indexes3),8,'filled', 'MarkerFaceAlpha',3/10,'MarkerFaceColor',trials.colors(kk,:))
            end
        else
            plot(unit.pos(indexes),unit.trials(indexes),'.','color',colors(k,:),'markersize',5), hold on
            
            %scatter(unit.pos(indexes),unit.trials(indexes),8,'filled', 'MarkerFaceAlpha',3/10,'MarkerFaceColor',colors(k,:))
        end
        
    end
%     colors22 = [0,0,0;0,1,0];
%     for k = 1:length(unique(trials.stat))
%         if isfield(unit,'rim')
%             indexes = find(ismember(unit.trials, find(trials.stat==k)) & unit.speed > animal.speed_th & unit.rim > 0); 
%         elseif isfield(animal,'rim') & ~isfield(unit,'rim')
%             unit.rim = interp1(animal.time,animal.rim,unit.ts/sr,'nearest');
%             indexes = find(ismember(unit.trials, find(trials.stat==k)) & unit.speed > animal.speed_th & unit.rim > 0); 
%         else
%             indexes = find(ismember(unit.trials, find(trials.stat==k)));
%         end
%     	plot(unit.pos(indexes),unit.trials(indexes),'.','color',colors22(k,:),'markersize',5), hold on
%     end
    if isfield(animal,'maze')
        if isfield(animal.maze,'reward_points')
            plot(animal.maze.reward_points,0,'sm')
            plot(animal.maze.reward_points,trials.total,'sm')
        end
    end
    if isfield(animal,'optogenetics')
        plot([animal.optogenetics.pos;animal.optogenetics.pos],[animal.optogenetics.trial-0.5;animal.optogenetics.trial+0.5],'-m','linewidth',2) % ,'color',colors(5,:)
    end
    xlim(animal.pos_limits),ylim([0,trials.total]),
    if isfield(animal,'maze')
            if isfield(animal.maze,'boundaries')
                gridxy(animal.maze.boundaries)
            end
        end
    if mod(i,5) == 1; title('Trials'); end
    if mod(i,5) == 0 | i == size(spikes.ts,2); xlabel('Position (cm)'); end

    % Firing rate map
    for k = 1:sum(unique(animal.state)>0)
        indexes2 = find(animal.state == k & animal.speed > animal.speed_th  & animal.rim);
        polar_theta_counts = histc(animal.pos(indexes2),x_hist2);
        subplot_Peter(5,sum(unique(animal.state)>0)+2,ii,sum(unique(animal.state)>0)+2)
        if isfield(unit,'rim')
            indexes = find(unit.state == k & unit.speed > animal.speed_th & unit.rim > 0);
        elseif isfield(animal,'rim')
            unit.rim = interp1(animal.time,animal.rim,unit.times,'nearest');
            indexes = find(unit.state == k & unit.speed > animal.speed_th & unit.rim > 0); 
        else
            indexes = find(unit.state == k);
        end
        % for ii = trials.error
        %     indexes(units.times{i}(indexes) > animal.time(trials.start(ii)) & units.times{i}(indexes) < animal.time(trials.end(ii))) = [];
        % end
        lin1 = unit.pos(indexes);
        hist_polar_count = histc(lin1,x_hist2);
        stairs(x_hist2,animal.sr*hist_polar_count./polar_theta_counts','color',colors(k,:)), hold on
        if isfield(animal,'maze')
            if isfield(animal.maze,'reward_points')
                plot(animal.maze.reward_points,0,'sm')
            end
        end
        if isfield(animal,'optogenetics') && ~isempty(animal.optogenetics.pos)
            plot(animal.optogenetics.pos,0,'^k')
            % hist_opto_count = histogram(animal.optogenetics.pos,x_hist2,'Normalization','probability','facecolor','none','edgecolor','k');
            % stairs(x_hist2,hist_opto_count.,'color',colors(k,:)), hold on
        end
        if isfield(animal,'maze')
            if isfield(animal.maze,'boundaries')
                gridxy(animal.maze.boundaries)
            end
        end
        axis tight, xlim([x_hist2(1),x_hist2(end)]),
        if isempty(unit.ts)
            title('No Spikes')
        else
            title([num2str(unit.total/((unit.times(end)-unit.times(1))),2) 'Hz']),
        end
        if mod(i,5) == 1; title(['Rate ' num2str(unit.total/((unit.ts(end)-unit.times(1))),2) 'Hz']); end
        
    end
    if mod(i,5) == 0
%         print(['Units', num2str(ii), '_', textlabel],'-dpdf');
    end
end
OscillationFreq = [];
theta_freq = [];
m = 1;
slope1 = [];
for i = 1:length(PhasePrecession)
    if ~isempty(PhasePrecession(i).Slope)
        for j = 1:size(PhasePrecession(i).Slope,2)
            if isfield(PhasePrecession(i).Slope{j},'slope1')
                elements = length(PhasePrecession(i).Slope{j}.slope1);
                slope1(j,m:m+elements-1) = PhasePrecession(i).Slope{j}.slope1;
                OscillationFreq(j,m:m+elements-1) = PhasePrecession(i).OscillationFreq(j);
                theta_freq(j,m:m+elements-1) = PhasePrecession(i).theta_freq(j);
            else
                elements = 0;
            end
        end
        m = m+elements;
    end
end

if ~isempty(PhasePrecession)
figure,
subplot(2,3,1), plot(abs(slope1)), 
title('Phase precession'), 
xticks([1:size(PhasePrecession(i).Slope,2)])
if isfield(animal,'state_labels')
    xticklabels(animal.state_labels)
end
subplot(2,3,2),
if size(slope1,1) ==3
    plot(slope1(1,:)./slope1(2,:),slope1(3,:)./slope1(2,:),'o'), axis equal, hold on, plot([0,2],[0,2]), plot([1,1],[0,2]), plot([0,2],[1,1])
    title('Ratio'), xlabel('Pre/Cooling')
else
    plot(slope1(1,:),slope1(2,:),'o'), axis equal, %hold on, plot([0,2],[0,2]), plot([1,1],[0,2]), plot([0,2],[1,1])
    title('Slopes'), 
    if isfield(animal,'state_labels')
        xlabel(animal.state_labels{1}), ylabel(animal.state_labels{2})
    end
end
hold on, plot([-0.1,0],[-0.1,0],'-','color',[0.5,0.5,0.5]), xlim([-0.1,0]),ylim([-0.1,0])
title('Ratio'), axis tight
subplot(2,3,4), boxplot(-abs(slope1)')
title('Phase precession boxplot'),
xticks([1:size(PhasePrecession(i).Slope,2)])
if isfield(animal,'state_labels') 
    xticklabels(animal.state_labels)
end
subplot(2,3,5),
if size(slope1,1) ==3
    boxplot([slope1(1,:)./slope1(2,:);slope1(3,:)./slope1(2,:)]')
    title('Ratios'), 
    xticks([1,2])
    xticklabels({'Pre/Cooling','Post/Cooling'})
else
    boxplot([slope1(1,:);slope1(2,:)]')
    title('Slopes'), 
    xticks([1,2])
    if isfield(animal,'state_labels') 
        xticklabels(animal.state_labels)
    end
end
subplot(2,3,3),
plot(OscillationFreq), title('Oscillation Frequency'), hold on
plot(theta_freq,'k')
subplot(2,3,6),
plot(OscillationFreq-theta_freq), title('Oscillation Frequency-theta')
end
