function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%

% first find the indices of the students admitted ('admit') or not ('not_admit')
admit = find(y==1);
not_admit = find(y==0);

% plot
plot(X(admit,1),X(admit,2),'b+',X(not_admit,1),X(not_admit,2),'ro');







% =========================================================================



hold off;

end
