function [trans,prob,prob_uncor,pred] = GetTransProb(rawCCG,n,binSize,conv_win,varargin)
% Extract the baseline corrected CCG + spike trans probe from raw CCG

%rawCCG = spike count between reference and target spike train
% n = number of reference spiks
% bin size = the binning of the CCG
% conv_win = slow, network comodulation time scale
% (optional input) = intwin = time bins in which synapse should inject
% excess synchrony


% define integration window
if ~isempty(varargin)
    intwin = varargin{1};
else
    intwin = round(length(rawCCG)/2) + round([.0008:binSize:.0028]/binSize);
end

%get CCG normalized by number of reference spikes
prob_uncor = rawCCG/n;

%calculate low freq. network comodulation
[ ~, pred ] = bz_cch_conv( rawCCG, round(conv_win/binSize));

%baseline subtract
prob = (rawCCG(:) - pred)/n;


%prob(prob<0) = 0;

%integrate
trans = sum(prob(intwin));
end