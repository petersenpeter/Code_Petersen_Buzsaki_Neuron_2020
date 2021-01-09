ids = [61,64,78,79,80,81,92,93,126,140,139,127,83,91,149,153,151,166,94,168];
for i = 1:length(ids)
    disp([num2str(i), '/', num2str(length(ids))])
    calc_phaseprecessionslopes(ids(i))
%     close all
end

%% Statistics about the place fields across sessions, including oscillation freq, field size
clear all, close all
MedialSeptum_Recordings

colors = {'g','b','r'};
conditions = {'Pre','Cooling','Post'};

maze = [];
maze.radius_in = 96.5/2;
maze.radius_out =  116.5/2;
maze.arm_half_width = 4;
maze.cross_radii = 47.9;
maze.polar_rho_limits = [44,65];
maze.polar_theta_limits = [-2.8,2.8]*maze.radius_in;
maze.pos_x_limits = [-10,10]; % -15
maze.pos_y_limits = [-40,45];

maze.boundary{1} = [0,40];
maze.boundary{2} = [0,25];
maze.boundary{3} = [maze.pos_x_limits(1),40];
maze.boundary{4} = [maze.pos_x_limits(2),40];
maze.boundary{5} = [maze.radius_in-3.25,maze.polar_theta_limits(2)];

slope1 = [];
mean1 = [];
std1 = [];
skewness1 = [];
mean2 = [];
std2 = [];
skewness2 = [];
OscillationFreq = [];
theta_freq = [];
speed = [];
PeakRate = [];
AverageRate = [];
FieldSize = [];
FieldWidth = [];
recordingID = [];
armRim = [];
fieldCenter = [];
datasets = [61,64,78,79,80,81,83,91,92,93,126,140,149,153,151,139,127,166]; % 62 63, 94,168
animalID = [1,1,  2,2,2,2,2,  3,3,3  4,4,4,4,4,  5,5,5];

for kkk = [61,64,78,79,80,81,92,93,126,140,139,127,83,91,149,153,151,166,94,168] %   [61,78,79,126,140,139,127,80,81,92,93]% [61,78,79,80,81,92,93,126,139,140,127]
    id = kkk; %subset(kkk);
    recording = recordings(id);
    disp([num2str(kkk) '. Loading session ', recording.name, ' with id ', num2str(id)])
    cd(fullfile('Z:\peterp03\IntanData\',recording.animal_id,recording.name))
    
    PhasePrecessionSlope = load('PhasePrecessionSlope.mat');
    
    slope1 = [slope1,PhasePrecessionSlope.slope1];
    mean1 = [mean1,PhasePrecessionSlope.mean1];
    std1 = [std1,PhasePrecessionSlope.std1];
    skewness1 = [skewness1,PhasePrecessionSlope.skewness1];
    mean2 = [mean2,PhasePrecessionSlope.mean2];
    std2 = [std2,PhasePrecessionSlope.std2];
    skewness2 = [skewness2,PhasePrecessionSlope.skewness2];
    OscillationFreq = [OscillationFreq,PhasePrecessionSlope.OscillationFreq];
    theta_freq = [theta_freq,PhasePrecessionSlope.theta_freq];
    speed = [speed,PhasePrecessionSlope.speed];
    
    PeakRate = [PeakRate,PhasePrecessionSlope.PeakRate];
    AverageRate = [AverageRate,PhasePrecessionSlope.AverageRate];
    FieldSize = [FieldSize,PhasePrecessionSlope.FieldSize];
    FieldWidth = [FieldWidth,PhasePrecessionSlope.FieldWidth];
    recordingID = [recordingID,kkk*ones(1,size(PhasePrecessionSlope.FieldWidth,2))];
    
    for i = 1:size(PhasePrecessionSlope.PhasePrecessionSlope1,2)
        if ~isempty(PhasePrecessionSlope.PhasePrecessionSlope1(i).placefield)
            temp = reshape(PhasePrecessionSlope.PhasePrecessionSlope1(i).placefield,3,[]);
            if size(temp,1)==1
                temp = temp';
            end
            temp = mean(temp([1,2],:));
            fieldCenter = [fieldCenter,temp]; %  + diff(maze.pos_y_limits) - maze.pos_y_limits(1)-maze.polar_theta_limits(1)
            armRim = [armRim,ones(1,length(temp))];
        end
    end
    for i = 1:size(PhasePrecessionSlope.PhasePrecessionSlope2,2)
        if ~isempty(PhasePrecessionSlope.PhasePrecessionSlope2(i).placefield)
            temp = reshape(PhasePrecessionSlope.PhasePrecessionSlope2(i).placefield,3,[]);
            if size(temp,1)==1
                temp = temp';
            end
            temp = mean(temp([1,2],:));
            fieldCenter = [fieldCenter,temp];
            armRim = [armRim,2*ones(1,length(temp))];
        end
    end
end

load('animal.mat')

% redefining mean phase
mean2(mean2<-pi/4) = mean2(mean2<-pi/4)+2*pi;
% redefining field std to 5std as an estimate of the size
std1 = std1*5;
% % % % % % % % % % % % % % % % % % % % % % % % % % % %
figure(100),
subplot(2,3,1), plot(slope1),
[p1,h1] = signrank(slope1(1,:),slope1(2,:));
[p2,h2] = signrank(slope1(2,:),slope1(3,:));
[p3,h3] = signrank(slope1(1,:),slope1(3,:));
text(1.2,0.055,[num2str(p1),',  ',num2str(h1)],'Rotation',45);
text(2.8,0.055,[num2str(p2),'  ,',num2str(h2)],'Rotation',45);
text(2,0.055,[num2str(p3),',  ',num2str(h3)],'Rotation',45);
title('Phase precession'),
xticks([1:size(PhasePrecessionSlope.speed,1)])
xticklabels({'Pre','Cooling','Post'})
subplot(2,3,2),
plot(slope1(1,:)./slope1(2,:),slope1(3,:)./slope1(2,:),'o'), axis equal, hold on, plot([0,2],[0,2]), plot([1,1],[0,2]), plot([0,2],[1,1])
title('Ratio'), xlabel('Pre/Cooling'), ylabel('Post/Cooling')
subplot(2,3,4),
boxplot((abs(slope1)-mean(abs(slope1)))')
title('Phase precession boxplot'),
xticks([1:size(PhasePrecessionSlope.speed,1)])
xticklabels(animal.state_labels)
subplot(2,3,5),
boxplot([slope1(1,:)./slope1(2,:);slope1(3,:)./slope1(2,:)]')
title('Ratios'),
xticks([1,2])
xticklabels({'Pre/Cooling','Post/Cooling'})

subplot(2,3,3),
x_data = slope1;
x_bins = [0:0.0025:0.04];
for i = 1:3
    %     indices = find(~isnan(x_data(i,:)));
    y1 = x_data(i,:);
    [N,edges] = histcounts(y1,x_bins, 'Normalization', 'probability')
    x_bin_diff = diff(x_bins);
    plot(x_bins(1:end-1)+x_bin_diff(1),N,colors{i}), hold on
    
    %     x2 = [x_bins(1:end-1)+x_bin_diff;x_bins(1:end-1)+x_bin_diff];
    %     y = [N;N];
    %     area(x2([2:end end]),y(1:end),'FaceColor',colors{i}),
end
axis tight, hold on, xlim([0,0.04]), xlabel('Phase precession slope')


% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
figure(101),
subplot(2,4,1), plot(OscillationFreq), hold on, plot(theta_freq,'k')
[p1,h1] = signrank(OscillationFreq(1,:),OscillationFreq(2,:));
[p2,h2] = signrank(OscillationFreq(2,:),OscillationFreq(3,:));
[p3,h3] = signrank(OscillationFreq(1,:),OscillationFreq(3,:));
text(1.2,16,[num2str(p1),',  ',num2str(h1)],'Rotation',45);
text(2.8,16,[num2str(p2),'  ,',num2str(h2)],'Rotation',45);
text(2,16,[num2str(p3),',  ',num2str(h3)],'Rotation',45);
title('Oscillation Frequency'),
xticks([1:size(PhasePrecessionSlope.speed,1)])
xticklabels(animal.state_labels)
subplot(2,4,2),
plot(OscillationFreq-theta_freq), title('Oscillation Frequency-theta'), hold on
x_data = OscillationFreq-theta_freq;
[p1,h1] = signrank(x_data(1,:),x_data(2,:));
[p2,h2] = signrank(x_data(2,:),x_data(3,:));
[p3,h3] = signrank(x_data(1,:),x_data(3,:));
text(1.2,7,[num2str(p1),',  ',num2str(h1)],'Rotation',45);
text(2.8,7,[num2str(p2),'  ,',num2str(h2)],'Rotation',45);
text(2,7,[num2str(p3),',  ',num2str(h3)],'Rotation',45);
xticks([1:size(PhasePrecessionSlope.speed,1)])
xticklabels(animal.state_labels)
subplot(2,4,3),
plot(speed),title('Animal speed'), hold on
x_data = speed;
[p1,h1] = signrank(x_data(1,:),x_data(2,:));
[p2,h2] = signrank(x_data(2,:),x_data(3,:));
[p3,h3] = signrank(x_data(1,:),x_data(3,:));
text(1.2,150,[num2str(p1),',  ',num2str(h1)],'Rotation',45);
text(2.8,150,[num2str(p2),'  ,',num2str(h2)],'Rotation',45);
text(2,150,[num2str(p3),',  ',num2str(h3)],'Rotation',45);
subplot(2,4,4),
plot(OscillationFreq(1,:)./OscillationFreq(2,:),OscillationFreq(3,:)./OscillationFreq(2,:),'o'), axis equal, hold on, plot([0,2],[0,2]), plot([1,1],[0,2]), plot([0,2],[1,1])
title('Ratio'), xlabel('Pre/Cooling'), ylabel('Post/Cooling')
subplot(2,4,5),
boxplot(OscillationFreq'), hold on
[p1,h1] = signrank(theta_freq(1,:),theta_freq(2,:));
[p2,h2] = signrank(theta_freq(2,:),theta_freq(3,:));
[p3,h3] = signrank(theta_freq(1,:),theta_freq(3,:));
text(1.2,15,[num2str(p1),',  ',num2str(h1)],'Rotation',45);
text(2.8,15,[num2str(p2),'  ,',num2str(h2)],'Rotation',45);
text(2,15,[num2str(p3),',  ',num2str(h3)],'Rotation',45);
subplot(2,4,6),
boxplot((OscillationFreq-theta_freq)')
subplot(2,4,7),
boxplot(speed')
subplot(2,4,8),
boxplot([OscillationFreq(1,:)./OscillationFreq(2,:);OscillationFreq(3,:)./OscillationFreq(2,:)]')
title('Ratios'),
xticks([1,2])
xticklabels({'Pre/Cooling','Post/Cooling'})

% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
figure(102),
subplot(2,2,1)
x_data = speed;
y_data = OscillationFreq;
for i = 1:3
    plt1(i) = scatter(x_data(i,:),y_data(i,:), 'MarkerFaceColor',colors{i}, 'MarkerEdgeColor','none','MarkerFaceAlpha',.3); hold on
    indices = intersect(find(~isnan(x_data(i,:))),find(~isnan(y_data(i,:))));
    x = x_data(i,indices); y1 = y_data(i,indices); P = polyfit(x,y1,1); yfit = P(1)*x+P(2); plot(x,yfit,[colors{i},'-']);
    [R,P] = corrcoef(x,y1);
    text(max(x),max(yfit),[conditions{i},': R = ' num2str(R(2,1),3),', P = ', num2str(P(2,1),3)],'Color',colors{i})
end
xlabel('Animal speed (cm)'), ylabel('Oscillation Frequency (Hz)')
x1 = nanmean(y_data(1,:) - y_data(2,:)) ./ nanmean(x_data(1,:) - x_data(2,:))
x2 = nanmean(y_data(3,:) - y_data(2,:)) ./ nanmean(x_data(3,:) - x_data(2,:))'
xx = [ nanmean(x_data(1,:)), nanmean(x_data(2,:)), nanmean(x_data(3,:))];
yy = [ nanmean(y_data(1,:)), nanmean(y_data(2,:)), nanmean(y_data(3,:))];
% plot(xx,yy,'o-k','linewidth',2),plot([6,10],[6,10],'--k')
legend(plt1,conditions,'Location','northeast'), %xlim([6,9.5]), ylim([6,13.5])

subplot(2,2,2)
colors = {'g','b','r'};
conditions = {'Pre','Cooling','Post'};
x_data = speed;
y_data = slope1;
for i = 1:3
    plt1(i) = scatter(x_data(i,:),y_data(i,:), 'MarkerFaceColor',colors{i}, 'MarkerEdgeColor','none','MarkerFaceAlpha',.3); hold on
    indices = intersect(find(~isnan(x_data(i,:))),find(~isnan(y_data(i,:))));
    x = x_data(i,indices); y1 = y_data(i,indices); P = polyfit(x,y1,1); yfit = P(1)*x+P(2); plot(x,yfit,[colors{i},'-']);
    [R,P] = corrcoef(x,y1);
    text(max(x),max(yfit),[conditions{i},': R = ' num2str(R(2,1),3),', P = ', num2str(P(2,1),3)],'Color',colors{i})
end
xlabel('Animal speed (cm)'), ylabel('Phase precession')
x1 = nanmean(y_data(1,:) - y_data(2,:)) ./ nanmean(x_data(1,:) - x_data(2,:))
x2 = nanmean(y_data(3,:) - y_data(2,:)) ./ nanmean(x_data(3,:) - x_data(2,:))'
xx = [ nanmean(x_data(1,:)), nanmean(x_data(2,:)), nanmean(x_data(3,:))];
yy = [ nanmean(y_data(1,:)), nanmean(y_data(2,:)), nanmean(y_data(3,:))];
% plot(xx,yy,'o-k','linewidth',2),plot([6,10],[6,10],'--k')
legend(plt1,conditions,'Location','northeast'), %xlim([6,9.5]), ylim([6,13.5])

subplot(2,2,3)
colors = {'g','b','r'};
conditions = {'Pre','Cooling','Post'};
x_data = speed;
y_data = theta_freq;
for i = 1:3
    plt1(i) = scatter(x_data(i,:),y_data(i,:), 'MarkerFaceColor',colors{i}, 'MarkerEdgeColor','none','MarkerFaceAlpha',.3); hold on
    indices = intersect(find(~isnan(x_data(i,:))),find(~isnan(y_data(i,:))));
    x = x_data(i,indices); y1 = y_data(i,indices); P = polyfit(x,y1,1); yfit = P(1)*x+P(2); plot(x,yfit,[colors{i},'-']);
    [R,P] = corrcoef(x,y1);
    text(max(x),max(yfit),[conditions{i},': R = ' num2str(R(2,1),3),', P = ', num2str(P(2,1),3)],'Color',colors{i})
end
xlabel('Animal speed (cm)'), ylabel('Theta frequency (Hz)')
x1 = nanmean(y_data(1,:) - y_data(2,:)) ./ nanmean(x_data(1,:) - x_data(2,:))
x2 = nanmean(y_data(3,:) - y_data(2,:)) ./ nanmean(x_data(3,:) - x_data(2,:))'
xx = [ nanmean(x_data(1,:)), nanmean(x_data(2,:)), nanmean(x_data(3,:))];
yy = [ nanmean(y_data(1,:)), nanmean(y_data(2,:)), nanmean(y_data(3,:))];
% plot(xx,yy,'o-k','linewidth',2),plot([6,10],[6,10],'--k')
legend(plt1,conditions,'Location','northeast'), %xlim([6,9.5]), ylim([6,13.5])

% % % % % % % % % % % % % % % % % % % % % % % % % % % %
figure(103)
colors = {'k','b','r'};
conditions = {'Pre','Cooling','Post'}
x_data = theta_freq;
y_data = OscillationFreq;
x_bins = [6:0.25:9.5];
y_bins = [6:0.25:13.5];
for i = 1:3
    subplot(1,3,1)
    plt1(i) = scatter(x_data(i,:),y_data(i,:), 'MarkerFaceColor',colors{i}, 'MarkerEdgeColor','none','MarkerFaceAlpha',.3); hold on
    indices = intersect(find(~isnan(x_data(i,:))),find(~isnan(y_data(i,:))));
    y1 = x_data(i,indices); x = y_data(i,indices); P = polyfit(x,y1,1); yfit = P(1)*x+P(2); plot(yfit,x,[colors{i},'-']);
    [R,P] = corrcoef(x,y1);
    text(max(yfit),max(x),[conditions{i},': R = ' num2str(R(2,1),3),', P = ', num2str(P(2,1),3)],'Color',colors{i})
    
    subplot(1,3,2)
    [N,edges] = histcounts(y1,x_bins, 'Normalization', 'probability')
    %     plot(x_bins(1:end-1)+0.125,N,colors{i},'linewidth',2)
    %     area(x_bins(1:end-1)+0.125,N,'FaceColor',colors{i}), hold on, xlim([6,9.5]), xlabel('LFP Theta frequency (Hz)')
    x2 = [x_bins(1:end-1)+0.125;x_bins(1:end-1)+0.125];
    y = [N;N];
    area(x2([2:end end]),y(1:end),'FaceColor',colors{i}), hold on, xlim([6,9.5]), xlabel('LFP Theta frequency (Hz)')
    
    subplot(1,3,3)
    [N,edges] = histcounts(x,y_bins, 'Normalization', 'probability')
    %     plot(y_bins(1:end-1)+0.125,N,colors{i},'linewidth',2)
    x = [y_bins(1:end-1)+0.125;y_bins(1:end-1)+0.125];
    y = [N;N];
    area(x([2:end end]),y(1:end),'FaceColor',colors{i}), hold on, xlim([6,13.5]), xlabel('Oscillation frequency (Hz)')
end
subplot(1,3,1)
xlabel('Theta Frequency'), ylabel('Oscillation Frequency'), axis equal
x1 = nanmean(y_data(1,:) - y_data(2,:)) ./ nanmean(x_data(1,:) - x_data(2,:))
x2 = nanmean(y_data(3,:) - y_data(2,:)) ./ nanmean(x_data(3,:) - x_data(2,:))'
xx = [ nanmean(x_data(1,:)), nanmean(x_data(2,:)), nanmean(x_data(3,:))];
yy = [ nanmean(y_data(1,:)), nanmean(y_data(2,:)), nanmean(y_data(3,:))];
plot(xx,yy,'o-k','linewidth',2),plot([6,10],[6,10],'--k')
legend(plt1,conditions,'Location','southeast'), xlim([6,9.5]), ylim([6,13.5])



% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
% Position
figure(104)
subplot(3,6,1), plot(mean1-mean(mean1)),
[p1,h1] = signrank(mean1(1,:),mean1(2,:));
[p2,h2] = signrank(mean1(2,:),mean1(3,:));
[p3,h3] = signrank(mean1(1,:),mean1(3,:));
text(1.2,12,[num2str(p1),',  ',num2str(h1)],'Rotation',45);
text(2.8,12,[num2str(p2),'  ,',num2str(h2)],'Rotation',45);
text(2,12,[num2str(p3),',  ',num2str(h3)],'Rotation',45);
title('Mean'),
xticks([1:size(PhasePrecessionSlope.speed,1)])
xticklabels(animal.state_labels)
subplot(3,6,7), boxplot((mean1-mean(mean1))')
xticks([1:size(PhasePrecessionSlope.speed,1)])
xticklabels(animal.state_labels)
subplot(3,6,2), plot(std1),
[p1,h1] = signrank(std1(1,:),std1(2,:));
[p2,h2] = signrank(std1(2,:),std1(3,:));
[p3,h3] = signrank(std1(1,:),std1(3,:));
text(1.2,170,[num2str(p1),',  ',num2str(h1)],'Rotation',45);
text(2.8,170,[num2str(p2),'  ,',num2str(h2)],'Rotation',45);
text(2,170,[num2str(p3),',  ',num2str(h3)],'Rotation',45);
title('Field size (5 std)'),
xticks([1:size(PhasePrecessionSlope.speed,1)])
xticklabels(animal.state_labels)
subplot(3,6,8), boxplot((std1)')
xticks([1:size(PhasePrecessionSlope.speed,1)])
xticklabels(animal.state_labels)
subplot(3,6,3), plot(skewness1),
[p1,h1] = signrank(skewness1(1,:),skewness1(2,:));
[p2,h2] = signrank(skewness1(2,:),skewness1(3,:));
[p3,h3] = signrank(skewness1(1,:),skewness1(3,:));
text(1.2,2.5,[num2str(p1),',  ',num2str(h1)],'Rotation',45);
text(2.8,2.5,[num2str(p2),'  ,',num2str(h2)],'Rotation',45);
text(2,2.5,[num2str(p3),',  ',num2str(h3)],'Rotation',45);
title('Skewness'),
xticks([1:size(PhasePrecessionSlope.speed,1)])
xticklabels(animal.state_labels)
subplot(3,6,9), boxplot((skewness1)')
xticks([1:size(PhasePrecessionSlope.speed,1)])
xticklabels(animal.state_labels)

% PHASE
subplot(3,6,4), plot(mean2),
title('Mean phase'),
xticks([1:size(PhasePrecessionSlope.speed,1)])
xticklabels(animal.state_labels)

subplot(3,6,10), boxplot((mean2)')
xticks([1:size(PhasePrecessionSlope.speed,1)])
xticklabels(animal.state_labels)
x = mean2;
[p1,h1] = signrank(x(1,:),x(2,:))
text(1.4,6,[num2str(p1),',',num2str(h1)],'Rotation',45);
[p2,h2] = signrank(x(2,:),x(3,:))
text(2.4,6,[num2str(p2),',',num2str(h2)],'Rotation',45);
[p3,h3] = signrank(x(1,:),x(3,:))
text(2,6,[num2str(p3),',',num2str(h3)],'Rotation',45);

subplot(3,6,5), plot(std2),
title('Std phase'),
xticks([1:size(PhasePrecessionSlope.speed,1)])
xticklabels(animal.state_labels)
subplot(3,6,11), boxplot((std2)')
xticks([1:size(PhasePrecessionSlope.speed,1)])
xticklabels(animal.state_labels)
x = std2;
[p1,h1] = signrank(x(1,:),x(2,:))
text(1.4,1.5,[num2str(p1),',',num2str(h1)],'Rotation',45);
[p2,h2] = signrank(x(2,:),x(3,:))
text(2.4,1.5,[num2str(p2),',',num2str(h2)],'Rotation',45);
[p3,h3] = signrank(x(1,:),x(3,:))
text(2,1.5,[num2str(p3),',',num2str(h3)],'Rotation',45);

subplot(3,6,6), plot(skewness2),
title('Skewness phase'),
xticks([1:size(PhasePrecessionSlope.speed,1)])
xticklabels(animal.state_labels)
subplot(3,6,12), boxplot((skewness2)')
xticks([1:size(PhasePrecessionSlope.speed,1)])
xticklabels(animal.state_labels)
x = skewness2;
[p1,h1] = signrank(x(1,:),x(2,:))
text(1.2,0.3,[num2str(p1),',',num2str(h1)],'Rotation',45);
[p2,h2] = signrank(x(2,:),x(3,:))
text(2.8,0.3,[num2str(p2),',',num2str(h2)],'Rotation',45);
[p3,h3] = signrank(x(1,:),x(3,:))
text(2,0.3,[num2str(p3),',',num2str(h3)],'Rotation',45);

% Ratios
ratio0 = mean1-mean(mean1);
ratio1 = [ratio0(1,:)-ratio0(2,:);ratio0(3,:)-ratio0(2,:)];
subplot(3,6,13), plot(ratio1(1,:),ratio1(2,:),'.'), hold on
x = ratio1(1,:); y1 = ratio1(2,:); P = polyfit(x,y1,1); yfit = P(1)*x+P(2); plot(x,yfit,'k-');
[R,P] = corrcoef(x,y1); text(max(x),max(yfit),['R = ' num2str(R(2,1),3),', P = ', num2str(P(2,1),3)],'Color','k')
axis equal, grid on

ratio0 = std1;
ratio1 = [ratio0(1,:)-ratio0(2,:);ratio0(3,:)-ratio0(2,:)];
subplot(3,6,14), plot(ratio1(1,:),ratio1(2,:),'.'), hold on
x = ratio1(1,:); y1 = ratio1(2,:); P = polyfit(x,y1,1); yfit = P(1)*x+P(2); plot(x,yfit,'k-');
[R,P] = corrcoef(x,y1); text(max(x),max(yfit),['R = ' num2str(R(2,1),3),', P = ', num2str(P(2,1),3)],'Color','k')
axis equal, grid on

ratio0 = skewness1;
ratio1 = [ratio0(1,:)-ratio0(2,:);ratio0(3,:)-ratio0(2,:)];
subplot(3,6,15), plot(ratio1(1,:),ratio1(2,:),'.'), hold on
x = ratio1(1,:); y1 = ratio1(2,:); y1(find(x>2))= []; x(x>2)= [];
P = polyfit(x,y1,1); yfit = P(1)*x+P(2); plot(x,yfit,'k-');
[R,P] = corrcoef(x,y1); text(max(x),max(yfit),['R = ' num2str(R(2,1),3),', P = ', num2str(P(2,1),3)],'Color','k')
axis equal, grid on

ratio0 = mean2;
ratio1 = [ratio0(1,:)-ratio0(2,:);ratio0(3,:)-ratio0(2,:)];
subplot(3,6,16), plot(ratio1(1,:),ratio1(2,:),'.'), hold on
x = ratio1(1,:); y1 = ratio1(2,:); P = polyfit(x,y1,1); yfit = P(1)*x+P(2); plot(x,yfit,'k-');
[R,P] = corrcoef(x,y1); text(max(x),max(yfit),['R = ' num2str(R(2,1),3),', P = ', num2str(P(2,1),3)],'Color','k')
axis equal, grid on

ratio0 = std2;
ratio1 = [ratio0(1,:)-ratio0(2,:);ratio0(3,:)-ratio0(2,:)];
subplot(3,6,17), plot(ratio1(1,:),ratio1(2,:),'.'), hold on
x = ratio1(1,:); y1 = ratio1(2,:); P = polyfit(x,y1,1); yfit = P(1)*x+P(2); plot(x,yfit,'k-');
[R,P] = corrcoef(x,y1); text(max(x),max(yfit),['R = ' num2str(R(2,1),3),', P = ', num2str(P(2,1),3)],'Color','k')
axis equal, grid on

ratio0 = skewness2;
ratio1 = [ratio0(1,:)-ratio0(2,:);ratio0(3,:)-ratio0(2,:)];
subplot(3,6,18), plot(ratio1(1,:),ratio1(2,:),'.'), hold on
x = ratio1(1,:); y1 = ratio1(2,:); P = polyfit(x,y1,1); yfit = P(1)*x+P(2); plot(x,yfit,'k-');
[R,P] = corrcoef(x,y1); text(max(x),max(yfit),['R = ' num2str(R(2,1),3),', P = ', num2str(P(2,1),3)],'Color','k')
axis equal, grid on


x1 = mean2;
x2 = std2;
x3 = skewness2;
figure,
subplot(2,2,1)
colors = {'g','b','r'};
conditions = {'Pre','Cooling','Post'};
x_data = mean2;
y_data = std2;
for i = 1:3
    plt1(i) = scatter(x_data(i,:),y_data(i,:), 'MarkerFaceColor',colors{i}, 'MarkerEdgeColor','none','MarkerFaceAlpha',.3); hold on
    indices = intersect(find(~isnan(x_data(i,:))),find(~isnan(y_data(i,:))));
    x = x_data(i,indices); y1 = y_data(i,indices); P = polyfit(x,y1,1); yfit = P(1)*x+P(2); plot(x,yfit,[colors{i},'-']);
    [R,P] = corrcoef(x,y1);
    text(max(x),max(yfit),[conditions{i},': R = ' num2str(R(2,1),3),', P = ', num2str(P(2,1),3)],'Color',colors{i})
end
% plot(x_data,y_data,'-k')
xlabel('std'), ylabel('skewness')
x1 = nanmean(y_data(1,:) - y_data(2,:)) ./ nanmean(x_data(1,:) - x_data(2,:))
x2 = nanmean(y_data(3,:) - y_data(2,:)) ./ nanmean(x_data(3,:) - x_data(2,:))'
xx = [ nanmean(x_data(1,:)), nanmean(x_data(2,:)), nanmean(x_data(3,:))];
yy = [ nanmean(y_data(1,:)), nanmean(y_data(2,:)), nanmean(y_data(3,:))];
% plot(xx,yy,'o-k','linewidth',2),plot([6,10],[6,10],'--k')
legend(plt1,conditions,'Location','northeast'), axis tight%xlim([0,140]),ylim([0,140]), grid on

subplot(2,2,2)


% % % % % % % % % % % % % % % % % % % % % % % % % % % %
figure(105)
subplot(1,3,1)
colors = {'g','b','r'};
conditions = {'Pre','Cooling','Post'};
x_data = std1;
y_data = OscillationFreq;
for i = 1:3
    plt1(i) = scatter(x_data(i,:),y_data(i,:), 'MarkerFaceColor',colors{i}, 'MarkerEdgeColor','none','MarkerFaceAlpha',.3); hold on
    indices = intersect(find(~isnan(x_data(i,:))),find(~isnan(y_data(i,:))));
    x = x_data(i,indices); y1 = y_data(i,indices); P = polyfit(x,y1,1); yfit = P(1)*x+P(2); plot(x,yfit,[colors{i},'-']);
    [R,P] = corrcoef(x,y1);
    text(max(x),min(yfit),[conditions{i},': R = ' num2str(R(2,1),3),', P = ', num2str(P(2,1),3)],'Color',colors{i})
end
xlabel('Place field size (std)'), ylabel('Oscillation Frequency (Hz)')
x1 = nanmean(y_data(1,:) - y_data(2,:)) ./ nanmean(x_data(1,:) - x_data(2,:))
x2 = nanmean(y_data(3,:) - y_data(2,:)) ./ nanmean(x_data(3,:) - x_data(2,:))'
xx = [ nanmean(x_data(1,:)), nanmean(x_data(2,:)), nanmean(x_data(3,:))];
yy = [ nanmean(y_data(1,:)), nanmean(y_data(2,:)), nanmean(y_data(3,:))];
% plot(xx,yy,'o-k','linewidth',2),plot([6,10],[6,10],'--k')
legend(plt1,conditions,'Location','northeast'), %xlim([6,9.5]), ylim([6,13.5])

subplot(1,3,2)
colors = {'g','b','r'};
conditions = {'Pre','Cooling','Post'};
x_data = std1;
y_data = theta_freq;
for i = 1:3
    plt1(i) = scatter(x_data(i,:),y_data(i,:), 'MarkerFaceColor',colors{i}, 'MarkerEdgeColor','none','MarkerFaceAlpha',.3); hold on
    indices = intersect(find(~isnan(x_data(i,:))),find(~isnan(y_data(i,:))));
    x = x_data(i,indices); y1 = y_data(i,indices); P = polyfit(x,y1,1); yfit = P(1)*x+P(2); plot(x,yfit,[colors{i},'-']);
    [R,P] = corrcoef(x,y1);
    text(max(x),min(yfit),[conditions{i},': R = ' num2str(R(2,1),3),', P = ', num2str(P(2,1),3)],'Color',colors{i})
end
xlabel('Place field size(std)'), ylabel('Theta Frequency (Hz)')
x1 = nanmean(y_data(1,:) - y_data(2,:)) ./ nanmean(x_data(1,:) - x_data(2,:))
x2 = nanmean(y_data(3,:) - y_data(2,:)) ./ nanmean(x_data(3,:) - x_data(2,:))'
xx = [ nanmean(x_data(1,:)), nanmean(x_data(2,:)), nanmean(x_data(3,:))];
yy = [ nanmean(y_data(1,:)), nanmean(y_data(2,:)), nanmean(y_data(3,:))];
% plot(xx,yy,'o-k','linewidth',2),plot([6,10],[6,10],'--k')
legend(plt1,conditions,'Location','northeast'), %xlim([6,9.5]), ylim([6,13.5])

subplot(1,3,3)
colors = {'g','b','r'};
conditions = {'Pre','Cooling','Post'};
x_data = std1;
y_data = speed;
for i = 1:3
    plt1(i) = scatter(x_data(i,:),y_data(i,:), 'MarkerFaceColor',colors{i}, 'MarkerEdgeColor','none','MarkerFaceAlpha',.3); hold on
    indices = intersect(find(~isnan(x_data(i,:))),find(~isnan(y_data(i,:))));
    x = x_data(i,indices); y1 = y_data(i,indices); P = polyfit(x,y1,1); yfit = P(1)*x+P(2); plot(x,yfit,[colors{i},'-']);
    [R,P] = corrcoef(x,y1);
    text(max(x),max(yfit),[conditions{i},': R = ' num2str(R(2,1),3),', P = ', num2str(P(2,1),3)],'Color',colors{i})
end
xlabel('Place field size (std)'), ylabel('Animal speed (cm)')
x1 = nanmean(y_data(1,:) - y_data(2,:)) ./ nanmean(x_data(1,:) - x_data(2,:))
x2 = nanmean(y_data(3,:) - y_data(2,:)) ./ nanmean(x_data(3,:) - x_data(2,:))'
xx = [ nanmean(x_data(1,:)), nanmean(x_data(2,:)), nanmean(x_data(3,:))];
yy = [ nanmean(y_data(1,:)), nanmean(y_data(2,:)), nanmean(y_data(3,:))];
% plot(xx,yy,'o-k','linewidth',2),plot([6,10],[6,10],'--k')
legend(plt1,conditions,'Location','northeast'), %xlim([6,9.5]), ylim([6,13.5])

%% % % % % % % % % % % % % %
figure(106),
subplot(3,3,1)
plot(AverageRate), ylabel('Average rate (Hz)')
[p1,h1] = signrank(AverageRate(1,:),AverageRate(2,:));
[p2,h2] = signrank(AverageRate(2,:),AverageRate(3,:));
[p3,h3] = signrank(AverageRate(1,:),AverageRate(3,:));
text(1.2,80,[num2str(p1),',  ',num2str(h1)],'Rotation',45);
text(2.8,80,[num2str(p2),'  ,',num2str(h2)],'Rotation',45);
text(2,80,[num2str(p3),',  ',num2str(h3)],'Rotation',45);
xticks([1:size(PhasePrecessionSlope.speed,1)]), xticklabels(animal.state_labels)
subplot(3,3,4), boxplot(AverageRate')
xticks([1:size(PhasePrecessionSlope.speed,1)])
xticklabels(animal.state_labels)
% ratio0 = AverageRate;
% ratio1 = [ratio0;ratio0(2,:)]; % [ratio0(1,:)./ratio0(2,:);ratio0(3,:)./ratio0(2,:)];

x = AverageRate(1,:); y1 = AverageRate(2,:);
subplot(3,3,7), plot(x,y1,'o'), hold on
P = polyfit(x,y1,1); yfit = P(1)*x+P(2); plot(x,yfit,'k-');
[R,P] = corrcoef(x,y1); text(max(x),max(yfit),['R = ' num2str(R(2,1),3),', P = ', num2str(P(2,1),3)],'Color','k')
axis equal, xlabel('Pre'), ylabel('Cooling'), xlim([0,80]), ylim([0,80]), refline(1,0)

subplot(3,3,2)
plot(PeakRate), ylabel('Peak rate (Hz)')
[p1,h1] = signrank(PeakRate(1,:),PeakRate(2,:));
[p2,h2] = signrank(PeakRate(2,:),PeakRate(3,:));
[p3,h3] = signrank(PeakRate(1,:),PeakRate(3,:));
text(1.2,100,[num2str(p1),',  ',num2str(h1)],'Rotation',45);
text(2.8,100,[num2str(p2),'  ,',num2str(h2)],'Rotation',45);
text(2,100,[num2str(p3),',  ',num2str(h3)],'Rotation',45);
xticks([1:size(PhasePrecessionSlope.speed,1)]), xticklabels(animal.state_labels)
subplot(3,3,5), boxplot(PeakRate')
xticks([1:size(PhasePrecessionSlope.speed,1)])
xticklabels(animal.state_labels)
ratio0 = PeakRate;
ratio1 = [ratio0;ratio0(2,:)]; %[ratio0(1,:)./ratio0(2,:);ratio0(3,:)./ratio0(2,:)];
x = ratio1(1,:); y1 = ratio1(2,:);
subplot(3,3,8), plot(ratio1(1,:),ratio1(2,:),'o'), hold on
P = polyfit(x,y1,1); yfit = P(1)*x+P(2); plot(x,yfit,'k-');
[R,P] = corrcoef(x,y1); text(max(x),max(yfit),['R = ' num2str(R(2,1),3),', P = ', num2str(P(2,1),3)],'Color','k')
axis equal, xlabel('Pre'), ylabel('Cooling'), xlim([0,100]), ylim([0,100]), refline(1,0)

subplot(3,3,3)
plot(FieldSize), ylabel('FieldSize (cm)')
[p1,h1] = signrank(FieldSize(1,:),FieldSize(2,:));
[p2,h2] = signrank(FieldSize(2,:),FieldSize(3,:));
[p3,h3] = signrank(FieldSize(1,:),FieldSize(3,:));
text(1.2,120,[num2str(p1),',  ',num2str(h1)],'Rotation',45);
text(2.8,120,[num2str(p2),'  ,',num2str(h2)],'Rotation',45);
text(2,120,[num2str(p3),',  ',num2str(h3)],'Rotation',45);
xticks([1:size(PhasePrecessionSlope.speed,1)]), xticklabels(animal.state_labels)
subplot(3,3,6), boxplot(FieldSize')
xticks([1:size(PhasePrecessionSlope.speed,1)])
xticklabels(animal.state_labels)
ratio0 = FieldSize;
ratio1 = [ratio0;ratio0(2,:)]; %[ratio0(1,:)./ratio0(2,:);ratio0(3,:)./ratio0(2,:)];
x = ratio1(1,:); y1 = ratio1(2,:);
subplot(3,3,9), plot(ratio1(1,:),ratio1(2,:),'o'), hold on
P = polyfit(x,y1,1); yfit = P(1)*x+P(2); plot(x,yfit,'k-');
[R,P] = corrcoef(x,y1); text(max(x),max(yfit),['R = ' num2str(R(2,1),3),', P = ', num2str(P(2,1),3)],'Color','k')
axis equal, xlabel('Pre'), ylabel('Cooling'), xlim([0,100]), ylim([0,100]), refline(1,0)

median(AverageRate(2,:)./AverageRate(1,:))
median(AverageRate(2,:)./AverageRate(3,:))
median(AverageRate(3,:)./AverageRate(1,:))
[p1,h1] = signrank(AverageRate(1,:),AverageRate(2,:));

median(PeakRate(2,:)./PeakRate(1,:))
median(PeakRate(2,:)./PeakRate(3,:))
median(PeakRate(3,:)./PeakRate(1,:))
[p1,h1] = signrank(PeakRate(1,:),PeakRate(2,:));

%%
figure(107),
colors = {'g','b','r'};
conditions = {'Pre','Cooling','Post'};
x_data = (std1);%FieldSize;
y_data = speed./(OscillationFreq-theta_freq);
for i = 1:3
    plt1(i) = scatter(x_data(i,:),y_data(i,:), 'MarkerFaceColor',colors{i}, 'MarkerEdgeColor','none','MarkerFaceAlpha',.3); hold on
    indices = intersect(find(~isnan(x_data(i,:))),find(~isnan(y_data(i,:))));
    x = x_data(i,indices); y1 = y_data(i,indices); P = polyfit(x,y1,1); yfit = P(1)*x+P(2); plot(x,yfit,[colors{i},'-']);
    [R,P] = corrcoef(x,y1);
    text(max(x),max(yfit),[conditions{i},': R = ' num2str(R(2,1),3),', P = ', num2str(P(2,1),3)],'Color',colors{i})
end
xlabel('Place field size (5*std)'), ylabel('Predicted field size (Speed / (f_0-f_{LFP})')
x1 = nanmean(y_data(1,:) - y_data(2,:)) ./ nanmean(x_data(1,:) - x_data(2,:));
x2 = nanmean(y_data(3,:) - y_data(2,:)) ./ nanmean(x_data(3,:) - x_data(2,:))';
xx = [ nanmean(x_data(1,:)), nanmean(x_data(2,:)), nanmean(x_data(3,:))];
yy = [ nanmean(y_data(1,:)), nanmean(y_data(2,:)), nanmean(y_data(3,:))];
% plot(xx,yy,'o-k','linewidth',2),plot([6,10],[6,10],'--k')
legend(plt1,conditions,'Location','northeast'),xlim([0,140]),ylim([0,140]), refline(1,0)



figure(109),
colors = {'g','b','r'};
conditions = {'Pre','Cooling','Post'};
x_data = slope1;
y_data = (OscillationFreq-theta_freq)./(std1);%FieldSize;
for i = 1:3
    plt1(i) = scatter(x_data(i,:),y_data(i,:), 'MarkerFaceColor',colors{i}, 'MarkerEdgeColor','none','MarkerFaceAlpha',.3); hold on
    indices = intersect(find(~isnan(x_data(i,:))),find(~isnan(y_data(i,:))));
    x = x_data(i,indices); y1 = y_data(i,indices); P = polyfit(x,y1,1); yfit = P(1)*x+P(2); plot(x,yfit,[colors{i},'-']);
    [R,P] = corrcoef(x,y1);
    text(max(x),max(yfit),[conditions{i},': R = ' num2str(R(2,1),3),', P = ', num2str(P(2,1),3)],'Color',colors{i})
end
% plot(x_data,y_data,'-k')
xlabel('Phase precession'), ylabel('(f_0-f_{LFP})/ field size')
x1 = nanmean(y_data(1,:) - y_data(2,:)) ./ nanmean(x_data(1,:) - x_data(2,:));
x2 = nanmean(y_data(3,:) - y_data(2,:)) ./ nanmean(x_data(3,:) - x_data(2,:))';
xx = [ nanmean(x_data(1,:)), nanmean(x_data(2,:)), nanmean(x_data(3,:))];
yy = [ nanmean(y_data(1,:)), nanmean(y_data(2,:)), nanmean(y_data(3,:))];
% plot(xx,yy,'o-k','linewidth',2),plot([6,10],[6,10],'--k')
legend(plt1,conditions,'Location','northeast'), axis tight, refline(1,0) %xlim([0,140]),ylim([0,140]), grid on

figure(110)%, subplot(1,2,1)
markersize = (mean(FieldSize))-20;
r = mean(FieldSize)./2;
plot_ThetaMaze(maze), hold on
for i =1:length(markersize)
    if armRim(i)==2
%         plot1 = scatter(0,fieldCenter(i),20*markersize(i),'filled');
        DrawCircle(0,fieldCenter(i),r(i))
    else
        [x,y] = pol2cart(-pi*fieldCenter(i)/(2*(maze.radius_out+5))+pi/2,(maze.radius_out-5));
        DrawCircle(x,y,r(i));
%         plot1 = scatter(x,y,20*markersize(i),'filled');
    end
%     alpha(plot1,.2)
end
axis equal
title('Place field sizes on maze')
figure
subplot(2,2,2), hold on
for i =1:length(markersize)
    if armRim(i)==2
        plot1 = scatter(fieldCenter(i)-maze.pos_y_limits(1),markersize(i),20*markersize(i),'filled');
    else
        if fieldCenter(i)<0
            plot1 = scatter(-fieldCenter(i)+diff(maze.pos_y_limits),markersize(i),20*markersize(i),'filled');
            
        else
            plot1 = scatter(fieldCenter(i)+diff(maze.pos_y_limits)-maze.polar_theta_limits(1),markersize(i),20*markersize(i),'filled');
        end
    end
    alpha(plot1,.2);
end
axis tight
gridxy([diff(maze.pos_y_limits),diff(maze.pos_y_limits)-maze.polar_theta_limits(1),diff(maze.pos_y_limits)+diff(maze.polar_theta_limits)])
title('Place field sizes linearized'), xlabel('Position (cm)'), ylabel('Field size')

subplot(2,2,4), hold on
slope11 = mean(slope1);
for i =1:length(markersize)
    if armRim(i)==2
        plot1 = scatter(fieldCenter(i)-maze.pos_y_limits(1),slope11(i),20*markersize(i),'filled');
    else
        if fieldCenter(i)<0
            plot1 = scatter(-fieldCenter(i)+diff(maze.pos_y_limits),slope11(i),20*markersize(i),'filled');
            
        else
            plot1 = scatter(fieldCenter(i)+diff(maze.pos_y_limits)-maze.polar_theta_limits(1),slope11(i),20*markersize(i),'filled');
        end
    end
    alpha(plot1,.2)
end
axis tight
gridxy([diff(maze.pos_y_limits),diff(maze.pos_y_limits)-maze.polar_theta_limits(1),diff(maze.pos_y_limits)+diff(maze.polar_theta_limits)])
title('Phase precession linearized'), xlabel('Position (cm)'), ylabel('Precession slope')