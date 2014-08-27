function [  ] = plot_pca( plot_title, varargin )

pca_fname = varargin{1}
delimiter = ' ';
data = dlmread(pca_fname,delimiter);

row_idx_pos = (data(:,1) == 1);
X_pos = data(row_idx_pos,2)
Y_pos = data(row_idx_pos,3)

row_idx_neg = (data(:,1) == -1);
X_neg = data(row_idx_neg,2)
Y_neg = data(row_idx_neg,3)

figure

plot(X_pos,Y_pos,'b*');
hold on;
%figure;
plot(X_neg,Y_neg,'r*');

title(plot_title)
legend('Positive Samples','Negative Samples');
legend('boxoff');

end

