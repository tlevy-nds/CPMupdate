import mlreportgen.ppt.*

if isequal(updateMethod, chosenUpdateMethod)
    hFigRocChosen = figure(); axRocChosen =  gca(); plot_roc(axRocChosen, allPs, allys, myColors(iupdateMethod, :), updateMethod);
    hFigPrChosen = figure(); axPrChosen = gca(); plot_pr(axPrChosen, allPs, allys, myColors(iupdateMethod, :), updateMethod); ylim(axPrChosen, [0.75 1]);
    
    pngfileRocChosen = fullfile('figs', sprintf('pic_roc_%s_%s_%d.png', modelType, chosenUpdateMethod, horizon));
    pngfilePrChosen = fullfile('figs', sprintf('pic_pr_%s_%s_%d.png', modelType, chosenUpdateMethod, horizon));
    pngfileCalChosen = fullfile('figs', sprintf('pic_cal_%s_%s_%d.png', modelType, chosenUpdateMethod, horizon));
    pngFileImportance = fullfile('figs', sprintf('pic_importance_%s_%d.png', modelType, horizon));
    
    figure(hFigRocChosen);print(pngfileRocChosen, '-dpng');
    figure(hFigPrChosen);print(pngfilePrChosen, '-dpng');
    figure(hFigCoeffs); print(pngFileImportance, '-dpng');
    figure(hCal); print(pngfileCalChosen, '-dpng');
    
    if horizon == 28
        slideNum = 5;
        slideRow = 'Bottom';
    elseif horizon == 7
        slideNum = 11;
        slideRow = 'Top';
    end
    
    % I think the print function can return before the operating system has finished creating the image file
    errorCt = 0;
    haserrors = true;
    while haserrors
        try
            switch modelType
                case 'nocos'
                    replace(slides.Children(slideNum), 'Top Left Left', Picture(pngfileRocChosen));
                    replace(slides.Children(slideNum), 'Top Left Right', Picture(pngfilePrChosen));
                    replace(slides.Children(slideNum), 'Bottom Left', Picture(pngfileCalChosen));
                    replace(slides.Children(9), sprintf('%s Left', slideRow), Picture(pngFileImportance));
                case 'LR'
                    replace(slides.Children(slideNum), 'Top Center Left', Picture(pngfileRocChosen));
                    replace(slides.Children(slideNum), 'Top Center Right', Picture(pngfilePrChosen));
                    replace(slides.Children(slideNum), 'Bottom Center', Picture(pngfileCalChosen));
                    replace(slides.Children(9), sprintf('%s Center', slideRow), Picture(pngFileImportance));
                case 'xgboost'
                    replace(slides.Children(slideNum), 'Top Right Left', Picture(pngfileRocChosen));
                    replace(slides.Children(slideNum), 'Top Right Right', Picture(pngfilePrChosen));
                    replace(slides.Children(slideNum), 'Bottom Right', Picture(pngfileCalChosen));
                    replace(slides.Children(9), sprintf('%s Right', slideRow), Picture(pngFileImportance));
            end
            haserrors = false;
        catch
            haserrors = true;
            errorCt = errorCt + 1;
            if errorCt > 30
                disp('infinite Loop');
            end
            pause(2);
        end
    end
end

% retro and no update slides
if isequal(updateMethod, 'no updates')
    pngfileRocRetroNoUpdates = fullfile('figs', sprintf('pic_roc_%s_retro_noUpdates_%d.png', modelType, horizon));
    pngfilePrRetroNoUpdates = fullfile('figs', sprintf('pic_pr_%s_retro_noUpdates_%d.png', modelType, horizon));
    pngfileCalRetro = fullfile('figs', sprintf('pic_cal_%s_retro_%d.png', modelType, horizon));
    pngfileCalNoUpdates = fullfile('figs', sprintf('pic_cal_%s_noUpdates_%d.png', modelType, horizon));
    
    figure(hFigRoc);print(pngfileRocRetroNoUpdates, '-dpng');
    figure(hFigPR);print(pngfilePrRetroNoUpdates, '-dpng');
    figure(hCalRetro); print(pngfileCalRetro, '-dpng');
    figure(hCal); print(pngfileCalNoUpdates, '-dpng');
    
    if horizon == 28
        slideNum = 2;
    elseif horizon == 7
        slideNum = 10;
    end
    
    % I think the print function can return before the operating system has finished creating the image file
    errorCt = 0;
    haserrors = true;
    while haserrors
        try
            switch modelType
                case 'nocos'
                    replace(slides.Children(slideNum), 'Top Left Left', Picture(pngfileRocRetroNoUpdates));
                    replace(slides.Children(slideNum), 'Top Left Right', Picture(pngfilePrRetroNoUpdates));
                    replace(slides.Children(slideNum), 'Middle Left', Picture(pngfileCalRetro));
                    replace(slides.Children(slideNum), 'Bottom Left', Picture(pngfileCalNoUpdates));
                case 'LR'
                    replace(slides.Children(slideNum), 'Top Center Left', Picture(pngfileRocRetroNoUpdates));
                    replace(slides.Children(slideNum), 'Top Center Right', Picture(pngfilePrRetroNoUpdates));
                    replace(slides.Children(slideNum), 'Middle Center', Picture(pngfileCalRetro));
                    replace(slides.Children(slideNum), 'Bottom Center', Picture(pngfileCalNoUpdates));
                case 'xgboost'
                    replace(slides.Children(slideNum), 'Top Right Left', Picture(pngfileRocRetroNoUpdates));
                    replace(slides.Children(slideNum), 'Top Right Right', Picture(pngfilePrRetroNoUpdates));
                    replace(slides.Children(slideNum), 'Middle Right', Picture(pngfileCalRetro));
                    replace(slides.Children(slideNum), 'Bottom Right', Picture(pngfileCalNoUpdates));
            end
            haserrors = false;
        catch
            haserrors = true;
            errorCt = errorCt + 1;
            if errorCt > 30
                disp('infinite Loop');
            end
            pause(2);
        end
    end
end