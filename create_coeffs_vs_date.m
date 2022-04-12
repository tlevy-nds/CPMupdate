%% from auc_vs_date
temp = findobj(gca, 'Type', 'Line'); myx = unique(temp(1).XData);

%% from coefficiens vs patient
temp = findobj(gca, 'Type', 'Line');

figure;
hold on;
arrayfun(@(x) plot(myx(1:end), x.YData, '.-', 'DisplayName', x.DisplayName, 'Color', x.Color), flip(temp));
hold off;
legend('show');
grid on;
xlim(myx([1 end-0]));
ylim([-.3 .35]);

%% coefficient bar plots
temp = findobj(gca, 'Type', 'Line');
y = arrayfun(@(x) x.YData(1), flip(temp));
dn = arrayfun(@(x) x.DisplayName, flip(temp), 'UniformOutput', false);
figure;bar(1:length(y), y);set(gca, 'XTick', 1:length(y), 'XTickLabel', dn);grid on;