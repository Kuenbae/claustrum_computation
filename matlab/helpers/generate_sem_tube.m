function [tubeX, tubeY, tubeZ] = generate_sem_tube(avgTraj, semTraj, nTheta)
%GENERATE_SEM_TUBE Generate a simple SEM tube around a 3-D trajectory.

if nargin < 3
    nTheta = 20;
end
T = size(avgTraj, 1);
theta = linspace(0, 2*pi, nTheta);
tubeX = zeros(T, nTheta);
tubeY = zeros(T, nTheta);
tubeZ = zeros(T, nTheta);

for i = 1:T
    pt = avgTraj(i, :);
    r = norm(semTraj(i, :));
    if i == 1
        tangent = avgTraj(min(2, T), :) - avgTraj(1, :);
    else
        tangent = avgTraj(i, :) - avgTraj(i-1, :);
    end
    if norm(tangent) == 0
        tangent = [1 0 0];
    else
        tangent = tangent ./ norm(tangent);
    end

    arbitrary = [0 0 1];
    if abs(dot(tangent, arbitrary)) >= 0.9
        arbitrary = [0 1 0];
    end
    u = cross(tangent, arbitrary);
    if norm(u) == 0
        u = [1 0 0];
    else
        u = u ./ norm(u);
    end
    v = cross(tangent, u);

    for j = 1:nTheta
        offset = r .* (cos(theta(j)) .* u + sin(theta(j)) .* v);
        tubeX(i, j) = pt(1) + offset(1);
        tubeY(i, j) = pt(2) + offset(2);
        tubeZ(i, j) = pt(3) + offset(3);
    end
end
end
