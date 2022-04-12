function window_button_down_fcn(obj, src, evt)

[~, mi] = min(sum(([obj.fpr obj.tpr obj.thresh] - evt.IntersectionPoint).^2, 2));

col = src.Color;
ax = get(src, 'Parent');
hold(ax, 'on');
plot3(ax, evt.IntersectionPoint(1), evt.IntersectionPoint(2), evt.IntersectionPoint(3), obj.markers(obj.markerInd), ...
    'MarkerSize', obj.markerSizes(obj.markerInd), 'Color', col);
hold(ax, 'off');
myTable = table({'Expired'; 'Survived'}, obj.opPts(:, 1, mi), obj.opPts(:, 2, mi), 'VariableNames', {'Actual', 'Predicted Expired', 'Predicted Survived'});
disp(myTable);
fprintf(1, 'Prob = %2.1f\nPPV = %2.1f\nNPV = %2.1f\n', 100 * obj.thresh(mi), ...
    100 * obj.opPts(2, 2, mi) / sum(obj.opPts(:, 2, mi)), 100 * obj.opPts(1, 1, mi) / sum(obj.opPts(:, 1, mi)));

if isvalid(obj.axpr)
    if isempty(obj.P)
        thr = obj.thresh;
    else
        thr = obj.P;
    end
    hold(obj.axpr, 'on');
    plot3(obj.axpr, obj.recall(mi), obj.precision(mi), thr(mi), obj.markers(obj.markerInd), ...
        'MarkerSize', obj.markerSizes(obj.markerInd), 'Color', col);
    hold(obj.axpr, 'off');    
    obj.markerInd = mod(obj.markerInd, length(obj.markers)) + 1;
end


