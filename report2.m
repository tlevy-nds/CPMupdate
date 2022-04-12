import mlreportgen.ppt.*

pngfileAucIciByPatients = fullfile('figs', sprintf('pic_auc_ici_%s_byPatients_%d.png', modelType, horizon));
pngfileAucIciByDate = fullfile('figs', sprintf('pic_auc_ici_%s_byDate_%d.png', modelType, horizon));
pngfileNB = fullfile('figs', sprintf('pic_nb_%s_%d.png', modelType, horizon));
% pngFileImportance = fullfile('figs', sprintf('pic_importance_%s_%d.png', modelType, horizon));

figure(hFigVsPatients); print(pngfileAucIciByPatients, '-dpng');
figure(hFigVsDate); print(pngfileAucIciByDate, '-dpng');
figure(hFigNB); print(pngfileNB, '-dpng');
% figure(hFigCoeffs); print(pngFileImportance, '-dpng');

haserrors = true;
% I think the print function can return before the operating system has finished creating the image file
errorCt = 0;
while haserrors
    try
        if horizon == 28
            switch modelType
                case 'nocos'
                    replace(slides.Children(3), 'Top Left', Picture(pngfileAucIciByPatients));
                    replace(slides.Children(4), 'Top Left', Picture(pngfileAucIciByDate));
                    replace(slides.Children(8), 'Bottom Left', Picture(pngfileNB));
                    % replace(slides.Children(9), 'Bottom Left', Picture(pngFileImportance));
                case 'LR'
                    replace(slides.Children(3), 'Top Center', Picture(pngfileAucIciByPatients));
                    replace(slides.Children(4), 'Top Center', Picture(pngfileAucIciByDate));
                    replace(slides.Children(8), 'Bottom Center', Picture(pngfileNB));
                    % replace(slides.Children(9), 'Bottom Center', Picture(pngFileImportance));
                case 'xgboost'
                    replace(slides.Children(3), 'Top Right', Picture(pngfileAucIciByPatients));
                    replace(slides.Children(4), 'Top Right', Picture(pngfileAucIciByDate));
                    replace(slides.Children(8), 'Bottom Right', Picture(pngfileNB));
                    % replace(slides.Children(9), 'Bottom Right', Picture(pngFileImportance));
            end
        elseif horizon == 7
            switch modelType
                case 'nocos'
                    replace(slides.Children(6), 'Top Left', Picture(pngfileAucIciByPatients));
                    replace(slides.Children(7), 'Top Left', Picture(pngfileAucIciByDate));
                    replace(slides.Children(8), 'Top Left', Picture(pngfileNB));
                    % replace(slides.Children(9), 'Top Left', Picture(pngFileImportance));
                case 'LR'
                    replace(slides.Children(6), 'Top Center', Picture(pngfileAucIciByPatients));
                    replace(slides.Children(7), 'Top Center', Picture(pngfileAucIciByDate));
                    replace(slides.Children(8), 'Top Center', Picture(pngfileNB));
                    % replace(slides.Children(9), 'Top Center', Picture(pngFileImportance));
                case 'xgboost'
                    replace(slides.Children(6), 'Top Right', Picture(pngfileAucIciByPatients));
                    replace(slides.Children(7), 'Top Right', Picture(pngfileAucIciByDate));
                    replace(slides.Children(8), 'Top Right', Picture(pngfileNB));
                    % replace(slides.Children(9), 'Top Right', Picture(pngFileImportance));
            end
        end
        haserrors = false;
    catch
        haserrors = true;
        errorCt = errorCt + 1;
        if errorCt > 30            
            disp('infinite loop');            
        end
        pause(2);
    end
end
