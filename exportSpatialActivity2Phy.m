% Exporting pny files for phy
disp('Exporting pny files for phy')
savepath =  pwd;

% Exporting theta phase
[signal_phase,signal_phase2] = calcInstantaneousTheta(recording);
ThetaTime = [1:length(signal_phase)]/(sr/16);
writeNPY(ThetaTime, fullfile(savepath, 'ThetaTime.npy'));

ThetaPhase = signal_phase;
writeNPY(ThetaPhase, fullfile(savepath, 'ThetaPhase.npy'));

% recording_length = session.General.Duration;
if ~isfield(recording,'arena')
    recording.arena = session.SubSessions.MazeType{1};
end

if any(strcmp(recording.arena,{'linear track','Linear track'}))
    load('animal.mat')
    posTime = animal.time;
    writeNPY(posTime, fullfile(savepath, 'posTime.npy'));
    
    posX = animal.pos(1,:);
    writeNPY(posX, fullfile(savepath, 'posX.npy'));
    
    posY = animal.pos(3,:);
    writeNPY(posY, fullfile(savepath, 'posY.npy'));
    
    posLin1 = animal.pos(1,:);
    posLin1(animal.ab==0)=nan;
    writeNPY(posLin1, fullfile(savepath, 'posLin1.npy'));
    
    posLin2 = animal.pos(1,:);
    posLin2(animal.ba==0)=nan;
    writeNPY(posLin2, fullfile(savepath, 'posLin2.npy'));

    % Exporting trials
    trialsphy = NaN(1,length(animal.time));
    for i = 1:length(trials.ab.start)
        trialsphy(trials.ab.start(i):trials.ab.end(i))=i;
    end
    writeNPY(trialsphy, fullfile(savepath, 'trialsphy1.npy'));
    
    % Exporting trials
    trialsphy = NaN(1,length(animal.time));
    for i = 1:length(trials.ba.start)
        trialsphy(trials.ba.start(i):trials.ba.end(i))=i;
    end
    writeNPY(trialsphy, fullfile(savepath, 'trialsphy2.npy'));
    
    % Exporting temperature
    if exist('temperature.mat')
        load('temperature.mat');
        phy_temp = interp1(temperature.time,temperature.temp,[1:floor(temperature.time(end))])./37;
        writeNPY(phy_temp, fullfile(savepath, 'temperature.npy'));
        trials_temperature = [];
        for i = 1:length(trials.ba.start)
            trials_temperature(i) = mean(animal.temperature(trials.ba.start(i):trials.ba.end(i)));
        end
        writeNPY(trials_temperature, fullfile(savepath, 'trials_temperature.npy'));
    end
    
elseif strcmp(recording.arena,'circular track')
    load('animal.mat')
    posTime = animal.time;
    writeNPY(posTime, fullfile(savepath, 'posTime.npy'));
    
    posX = animal.pos(1,:);
    writeNPY(posX, fullfile(savepath, 'posX.npy'));
    
    posY = animal.pos(2,:);
    writeNPY(posY, fullfile(savepath, 'posY.npy'));
    
    posLin1 = animal.polar_theta;
    posLin1(animal.rim==0)=nan;
    writeNPY(posLin1, fullfile(savepath, 'posLin1.npy'));
    
    posLin2 = animal.pos(2,:);
    posLin2(animal.arm==0)=nan;
    writeNPY(posLin2, fullfile(savepath, 'posLin2.npy'));
    
    % % Exporting trials
    trialsphy1 = NaN(1,length(animal.time));
    for i = 1:length(trials.start)
        trialsphy1(trials.start(i):trials.end(i))=i;
    end
    writeNPY(trialsphy1, fullfile(savepath, 'trialsphy1.npy'));
    writeNPY(trialsphy1, fullfile(savepath, 'trialsphy2.npy'));
    
%     % Exporting temperature
%     if exist('temperature.mat')
%         load('temperature.mat');
%         phy_temp = interp1(temperature.time,temperature.temp,[1:floor(temperature.time(end))])./37;
%         writeNPY(phy_temp, fullfile(savepath, 'temperature.npy'));
%         trials_temperature = [];
%         for i = 1:length(trials.ba.start)
%             trials_temperature(i) = mean(animal.temperature(trials.ab.start(i):trials.ab.end(i)));
%         end
%         writeNPY(trials_temperature, fullfile(savepath, 'trials_temperature.npy'));
%     end
end

% Exporting optogenetics
if exist('optogenetics.mat')
    load('optogenetics.mat');
    opto = optogenetics.peak;
    writeNPY(opto, fullfile(savepath, 'optogenetics.npy'));
end
% Exporting behavior types
disp('Export to phy complete')
% behavior_types = [0,2,prebehaviortime/sr,2,prebehaviortime/sr,4,(behaviortime+prebehaviortime)/sr,4,(behaviortime+prebehaviortime)/sr,2,recording_length,2];
% writeNPY(behavior_types, fullfile(savepath, 'behavior_types.npy'));
