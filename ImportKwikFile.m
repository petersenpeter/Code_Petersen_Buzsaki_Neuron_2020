function [units,indexes] = ImportKwikFile(dataset,folder,shanks,plots,units_to_exclude)
% Imports data from the new Kwik format (Klustasuite 0.3.0.beta4)
%
% Inputs: 
% dataset:  Filename of the kwik file (without kwik-extension)
% folder:   Path of the dataset
% shanks:   Select the shanks to import
% plots:    Defines if you want to plot a rastergram with the imported
%        spiketrains (1 or 0)
% units_to_exclude: excluded specific units to import
%
% Outputs:
% units: structure fule containing spike_times (ts), waveforms, total number of spikes, kwik_id
% indexes: Indexes for all spikes, used for easy rastergram plots.
%
% By Peter Petersen
% petersen.peter@gmail.com

dataset = 'ham5_769_batch';
folder = 'D:\IntanData\ViktorsData\ham5_769_batch\spikes';
shanks = [1,2,4];
plots = 0;
units_to_exclude = [];

if nargin == 3; plots = 0; units_to_exclude = []; 
elseif nargin == 4; units_to_exclude = []; 
end

indexes = [];
units = [];
shank_nb = 1;
for shank = 1:shanks
    spike_times = double(hdf5read([folder,'/', dataset, '.kwik'], ['/channel_groups/' num2str(shank-1) '/spikes/time_samples']));
    recording_nb = double(hdf5read([folder,'/', dataset, '.kwik'], ['/channel_groups/' num2str(shank-1) '/spikes/recording']));
    cluster_index = double(hdf5read([folder,'/', dataset, '.kwik'], ['/channel_groups/' num2str(shank-1) '/spikes/clusters/main']));
    % waveforms = double(hdf5read([folder, dataset, '.kwx'], ['/channel_groups/' num2str(shank-1) '/waveforms_filtered']));
    clusters = unique(cluster_index);
    for i = 1:length(clusters(:))
        cluster_type = double(hdf5read([folder,'/', dataset, '.kwik'], ['/channel_groups/' num2str(shank-1) '/clusters/main/' num2str(clusters(i)),'/'],'cluster_group'));
        if cluster_type == 2
            indexes{shank_nb} = shank_nb*ones(sum(cluster_index == clusters(i)),1);
            units.ts{shank_nb} = spike_times(cluster_index == clusters(i))+recording_nb(cluster_index == clusters(i))*40*40000;
            units.total(shank_nb) = sum(cluster_index == clusters(i));
            units.shank(shank_nb) = shank-1;
            units.kwik_id(shank_nb) = clusters(i);
            % units.waveforms{shank_nb} = mean(waveforms(:,:,cluster_index == clusters(i)),3);
            % units.waveforms_std{shank_nb} = permute(std(permute(waveforms(:,:,cluster_index == clusters(i)),[3,1,2])),[2,3,1]); % [3,1,2])),[2,3,1]
            shank_nb = shank_nb+1;
        end
    end
end
if ~isempty(units_to_exclude)
    disp('Excluding units...')
    indexes(units_to_exclude) = [];
    units.ts(units_to_exclude) = [];
    units.total(units_to_exclude) = [];
    units.shank(units_to_exclude) = [];
    units.kwik_id(units_to_exclude) = [];
    % units.waveforms(units_to_exclude) = [];
    % units.waveforms_std(units_to_exclude) = [];
    for i = 1:size(indexes,2)
        indexes{i} = ones(length(indexes{i}),1)*i;
    end
end
if plots
    figure
    plot([vertcat(units.ts{:}), vertcat(units.ts{:})]'/40000,[vertcat(indexes{:})-1,vertcat(indexes{:})-0.05]','k'), hold on
end
