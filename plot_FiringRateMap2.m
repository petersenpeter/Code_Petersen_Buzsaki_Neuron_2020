function firing_rate_map = plot_FiringRateMap2(varargin)
% firing_rate_map = plot_FiringRateMap('animal',animal,'spikes',spikes,'trials',trials,'binsize',5)
% 
% INPUTS
% animal
% spikes
% trials
% units2plot
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

parse(p,varargin{:})

animal = p.Results.animal;
spikes = p.Results.spikes;
trials = p.Results.trials;
bin_size = p.Results.bin_size;
units2plot = p.Results.units2plot;

firing_rate_map = [];
if isfield(trials,'temperature')
    trial_colors = winter(ceil(10*(max(trials.temperature)-min(trials.temperature))));
    trials.colors = trial_colors(1+floor((trials.temperature-min(trials.temperature))/(max(trials.temperature)-min(trials.temperature))*(size(trial_colors,1)-1)),:);
end

% Plotting
x_bins = [animal.pos_limits(1):bin_size:animal.pos_limits(2)];
firing_rate_map.x_bins = x_bins;
colors = [0.8500, 0.3250, 0.0980; 0, 0.4470, 0.7410; 0.9290, 0.6940, 0.1250; 0.4940, 0.1840, 0.5560; 0.4660, 0.6740, 0.1880; 0.3010, 0.7450, 0.9330; 0.6350, 0.0780, 0.1840; 0.4660, 0.6740, 0.1880; 0.3010, 0.7450, 0.9330; 0.6350, 0.0780, 0.1840];
if ~exist('units2plot')
    units2plot = [];
end
if isempty(units2plot)
    units2plot = 1:size(spikes.times,2);
end
if ~isfield(spikes,'PhasePrecession')
    spikes.PhasePrecession(units2plot) = {[]};
end

ii = 0;
for i = units2plot
    ii = ii + 1;
    clear unit;
    unit.times = spikes.times{i};
    unit.total = spikes.total(i);
    unit.cluID = spikes.cluID(i);
%     unit.PhasePrecession = spikes.PhasePrecession{i};
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
    if ~isfield(spikes,'state')
        unit.state = interp1(animal.time,animal.state,unit.times,'nearest');
    else
        unit.state = spikes.state{i};
    end
    
    % Firing rate map
    for k = 1:sum(unique(animal.state)>0)
        if isfield(animal,'state_labels')
            firing_rate_map.state_labels = animal.state_labels;
        end
        indexes2 = find(animal.state == k & animal.speed > animal.speed_th);
        polar_theta_counts = histc(animal.pos(indexes2),x_bins);
        subplot_Peter(5,sum(unique(animal.state)>0),ii,k)
        if isfield(unit,'rim')
            indexes = find(unit.state == k & unit.speed > animal.speed_th & unit.rim > 0);
        elseif isfield(animal,'rim')
            unit.rim = interp1(animal.time,animal.rim,unit.times,'nearest');
            indexes = find(unit.state == k & unit.speed > animal.speed_th & unit.rim > 0);
        else
            indexes = find(unit.state == k);
        end
        
        lin1 = unit.pos(indexes);
        hist_polar_count = histc(lin1,x_bins);
        firing_rate_map.map{i}(:,k) = (1/mean(diff(animal.time)))*hist_polar_count(:)./polar_theta_counts(:);
        stairs(x_bins,(1/mean(diff(animal.time)))*hist_polar_count(:)./polar_theta_counts(:),'color',colors(k,:)), hold on
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
        axis tight, xlim([x_bins(1),x_bins(end)]),
        if isempty(unit.times)
            title('No Spikes')
        else
            title([num2str(unit.total/((unit.times(end)-unit.times(1))),2) 'Hz']),
        end
        if k == 1
            ylabel(['Unit ' num2str(i) ' (id ' num2str(unit.cluID) ')']);
        end
    end
end
