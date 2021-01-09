function SpikeSortingQualityTest(units_set1,units_set2)
% 
% clear all
% datapath = 'G:\IntanData\';
% Recordings_MedialSeptum
% id = 51;
% recording= recordings(id).name;
% data_path = [datapath, recordings(id).name(1:6) recordings(id).rat_id '\' recording, '\']
% cd(data_path)
% shanks = 1:3;
% Loading units from the three sorting methods

units_all = [units_set1.ts,units_set2.ts];
sr = 20000;
t_bins = [0:5:max(cellfun(@max,units_all))./20]/1000;
corr_matrix = zeros(size(units_all,2));
corr_matrix(1:size(units_set1.ts,2),1:size(units_set1.ts,2))= 0.05;
% corr_matrix(size(units_klustakwik.ts,2)+1:size([units_klustakwik.ts,units_Klusta.ts],2),size(units_klustakwik.ts,2)+1:size([units_klustakwik.ts,units_Klusta.ts],2))= 0.1;
% corr_matrix(size([units_klustakwik.ts,units_Klusta.ts],2)+1:end,size([units_klustakwik.ts,units_Klusta.ts],2)+1:end)= 0.15;

corr_matrix = corr_matrix.*tril(ones(size(units_all,2)),0);
disp('Calculating the correlation matrix')
size(units_all,2)
tic
parfor i = 1:size(units_all,2)
    i
    hist1 = hist(units_all{i}./20000,t_bins);
    v = zeros(1,size(units_all,2));
    for j = min(i+1,size(units_all,2)):size(units_all,2)
        hist2 = hist(units_all{j}/20000,t_bins);
        temp = corrcoef(hist1,hist2);
        v(j) = temp(2,1);
    end
    corr_matrix(i,:) = v;
    %disp((sum(corr_matrix~=0)/(size(units_all,2)*size(units_all,1))*100))
end
toc
corr_matrix(end,end) = 0;
corr_matrix1 = corr_matrix(1:size(units_set1.ts,2),size(units_set1.ts,2)+1:end);
[~,I1] = sort(max(corr_matrix1),'descend');
corr_matrix2 = corr_matrix1(:,I1)';
[~,I2] = sort(max(corr_matrix2),'descend');
figure
imagesc(corr_matrix2(:,I2)), hold on, colormap jet, colorbar, axis equal, axis tight, hold on
ylabel({units_set2.SpikeSorting_method; ['(' units_set2.SpikeSorting_path ')']}),xlabel({units_set1.SpikeSorting_method; ['(' units_set1.SpikeSorting_path ')']}),
xticks([1:size(units_set1.ts,2)])
xticklabels(num2str(I2'))
yticks([1:size(units_set2.ts,2)])
yticklabels(num2str(I1'))
title('Confusion matrix')
save('corr_matrix2.mat','corr_matrix2','corr_matrix')