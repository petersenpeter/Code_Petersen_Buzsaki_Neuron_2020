function pos_linearized = linearize_pos(animal,arena)
% Linearize a 2D trajectory into one 1D.
% 
% Peter Petersen
% petersen.peter@gmail.com

if any(strcmp(arena, {'theta','CircularTrack','circular track'}))
    % This is a circular track as used by Peter and Viktor
    % The origin of the track has to be centralized with the origin at the center of the maze
    % First the central arm is linearized
    pos_linearized = nan(size(animal.time));
    pos_linearized(find(animal.arm)) = animal.pos(2,find(animal.arm))-animal.pos_y_limits(1);
    % Next the left(?) return side-arm
    boundary = diff(animal.pos_y_limits)-5;
    pos_linearized(find(animal.rim & animal.polar_theta < -5)) = -animal.polar_theta(find(animal.rim & animal.polar_theta < -5)) + boundary;
    % Finally the right return side-arm is linearized.
    boundary = boundary + abs(animal.polar_theta_limits(1))-5;
    pos_linearized(find(animal.rim & animal.polar_theta > 5)) = animal.polar_theta(find(animal.rim & animal.polar_theta > 5)) + boundary;
end
